import os
import shutil
import hydra
import wandb
import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger, MLFlowLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Callback
from lightning.pytorch.utilities import rank_zero_only

from loguru import logger
from torch.utils.data import DataLoader
from typing import Any, Dict, List, Optional, Tuple, Callable
from omegaconf import DictConfig, OmegaConf

from transformers import  AutoTokenizer
from hf_ehr.data.datasets import FEMRDataset
from hf_ehr.data.tokenization import CookbookTokenizer, CLMBRTokenizer, DescTokenizer
from hf_ehr.models.bert import BERTLanguageModel
from hf_ehr.models.gpt import GPTLanguageModel
from hf_ehr.models.hyena import HyenaLanguageModel
from hf_ehr.models.mamba import MambaLanguageModel
from hf_ehr.models.t5 import T5LanguageModel
from hf_ehr.trainer.loaders import load_datasets, load_dataloaders
from hf_ehr.config import rewrite_paths_for_carina_from_config, PATH_TO_TOKENIZER_CLMBR_v8_CONFIG, PATH_TO_TOKENIZER_DESC_v8_CONFIG
from hf_ehr.logger.reloggers import WandbRelogger

class GradNormCallback(Callback):
    """
    Source: https://github.com/Lightning-AI/lightning/issues/1462
    """

    def gradient_norm(self, model):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm

    def on_before_optimizer_step(self, trainer, model, optimizer):
        model.log("optim/grad_norm_raw", self.gradient_norm(model))


class StartTrainingCheckpoint(ModelCheckpoint):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_train_start(self, trainer, pl_module):
        if rank_zero_only.rank != 0:
            return
        checkpoint_file = os.path.join(self.dirpath, f"{self.filename}.ckpt")
        if not os.path.exists(checkpoint_file):
            # Save a checkpoint at the beginning of training
            self._save_checkpoint(trainer, checkpoint_file)

class MetricBasedCheckpoint(pl.callbacks.Callback):
    def __init__(self, metric_name: str, is_valid_metric_func: Callable, dirpath: str, is_run_val: bool = False):
        """
            is_valid_metric_func: Callable
                Inputs: 
                    current metric value: Any
                    metric vale at last ckpt: Any
                Returns:
                    is_ckpt: boolean -- True if create ckpt at this step
                    val: Any -- cleaned version of current metric to associate with ckpt
        """
        super().__init__()
        self.metric_name: str = metric_name
        self.is_valid_metric_func: Callable = is_valid_metric_func
        self.dirpath: str = dirpath
        self.is_run_val = is_run_val
        self.last_ckpt_metric_value: Optional[Any] = None
        self.is_run_val: bool = is_run_val
        self.is_validating = False # TODO

    def on_train_batch_end(self, trainer, *args, **kwargs):
        if rank_zero_only.rank != 0:
            return
        
        metrics = trainer.callback_metrics
        metric_value = metrics.get(self.metric_name)
        
        if metric_value is not None:
            is_ckpt, true_val, ckpt_val = self.is_valid_metric_func(metric_value, self.last_ckpt_metric_value)
            print(f"Metric: {self.metric_name} | Metric Value: {metric_value}, Interval: {true_val}, Checkpoint Value: {ckpt_val}")
            if is_ckpt:
                filepath = os.path.join(self.dirpath, f"{self.metric_name.replace('/', '-')}-true_val={true_val}-ckpt_val={ckpt_val}-persist.ckpt")
                trainer.save_checkpoint(filepath)
                logger.info(f"Checkpoint saved at {filepath} with {self.metric_name}={metric_value}")
                self.last_ckpt_metric_value = metric_value
                if self.is_run_val:
                    logger.info(f"Triggering validation at step {trainer.global_step} due to metric {self.metric_name} with value ckpt_val={ckpt_val}")
                    # Synchronize GPUs
                    # torch.cuda.synchronize()
                    # # Pausing other GPUs
                    # current_device = torch.cuda.current_device()
                    # for i in range(torch.cuda.device_count()):
                    #     if i != current_device:
                    #         torch.cuda.set_device(i)
                    #         torch.cuda.synchronize()
                    # # Run validation on the primary GPU
                    # torch.cuda.set_device(current_device)
                    # Run validation
                    breakpoint()
                    _first_loop_iter: Optional[bool] = trainer._logger_connector._first_loop_iter
                    _results = trainer._logger_connector.trainer._results
                    trainer.validate(model=trainer.model.module, ckpt_path=None, dataloaders=trainer.val_dataloaders)
                    trainer._logger_connector._first_loop_iter = _first_loop_iter
                    trainer._logger_connector.trainer._results = _results
                    # # Resume other GPUs
                    # for i in range(torch.cuda.device_count()):
                    #     if i != current_device:
                    #         torch.cuda.set_device(i)
                    #         torch.cuda.synchronize()
                    
                    # # Set back to the original device
                    # torch.cuda.set_device(current_device)

    @property
    def state_key(self) -> str:
        kwargs = { 
            'metric_name' : self.metric_name,
        }
        return f"{self.__class__.__qualname__}{repr(kwargs)}"

def train_flops_metric_func(val: int, last_val: Optional[int], config) -> Tuple[bool, int, int]:
    interval: int = int(config.callbacks.model_checkpointing.every_n_flops)
    current: int = int(val // interval)
    if last_val is None:
        return True, int(val), current * interval
    else:
        last: int = int(last_val // interval)
        return last < current, int(val), current * interval

def train_token_metric_func(val: int, last_val: Optional[int], config) -> Tuple[bool, int, int]:
    interval: int = int(config.callbacks.model_checkpointing.every_n_train_nonPAD_tokens)
    current: int = int(val // interval)
    if last_val is None:
        return True, int(val), current * interval
    else:
        last: int = int(last_val // interval)
        return last < current, int(val), current * interval

@hydra.main(version_base=None, config_path='../configs/', config_name="config")
def main(config: DictConfig) -> None:
    # Rewrite paths for /local-scratch on certain partitions
    config = rewrite_paths_for_carina_from_config(config)

    # Load config
    print(config)
    path_to_output_dir: str = config.main.path_to_output_dir
    is_wandb: bool = config.logging.wandb.is_wandb
    is_mlflow: bool = config.logging.mlflow.is_mlflow
    is_log_grad_norm: bool = config.logging.is_log_grad_norm
    model_name: str = config.model.name
    path_to_tokenizer_config: str = config.data.tokenizer.path_to_config
    tokenizer_metadata: Dict[str, Any] = getattr(config.data.tokenizer, 'metadata', {})
    seed: int = config.main.seed
    is_force_restart: bool = config.main.is_force_restart

    # Random seed
    pl.seed_everything(seed, workers=True)

    # Check if resuming from checkpoint
    is_resume_from_ckpt: bool = os.path.exists(os.path.join(path_to_output_dir, 'ckpts/last.ckpt'))
    path_to_resume_ckpt: Optional[str] = os.path.join(path_to_output_dir, 'ckpts/last.ckpt') if is_resume_from_ckpt else None
    if is_force_restart:
        print("====================================")
        print("====================================")
        print("!!!! Force restart !!!!")
        print(f"!!!! Deleting folder at `{path_to_output_dir}` !!!!")
        print("====================================")
        print("====================================")
        is_resume_from_ckpt = False
        path_to_resume_ckpt = None
        if os.path.exists(path_to_output_dir):
            shutil.rmtree(path_to_output_dir)

    # Paths
    path_to_log_dir: str = os.path.join(path_to_output_dir, 'logs/')
    path_to_ckpt_dir: str = os.path.join(path_to_output_dir, 'ckpts/')
    path_to_log_file: str = os.path.join(path_to_log_dir, 'info.log')
    path_to_artifacts_dir: str = os.path.join(path_to_log_dir, 'artifacts/')
    os.makedirs(path_to_output_dir, exist_ok=True)
    os.makedirs(path_to_log_dir, exist_ok=True)
    os.makedirs(path_to_ckpt_dir, exist_ok=True)
    os.makedirs(path_to_artifacts_dir, exist_ok=True)
    
    # Logging
    logger.add(path_to_log_file, enqueue=True, mode='a')
    logger.info(config)
    loggers: List = [ TensorBoardLogger(save_dir=path_to_log_dir) ]

    # IMPORTANT! Keep this key (i.e. a pointer to the `tokenizer` object) out of the config, otherwise stuff is slow / logging breaks
    if hasattr(config, 'tokenizer'):
        config.__delattr__('tokenizer')

    ## MLFlow
    if is_mlflow:
        if is_resume_from_ckpt:
            # Load existing mlflow run ID
            with open(os.path.join(path_to_log_dir, 'mlflow_run_id.txt'), 'r') as f:
                mlflow_run_id: str = f.read()
            logger.info(f"Found existing mlflow run: `{mlflow_run_id}`")
            loggers += [ 
                    MLFlowLogger(experiment_name='hf_ehr',
                                    run_id=mlflow_run_id,
                                    log_model='all',
                                    save_dir=f"{path_to_log_dir}",
                                    tracking_uri=f"file:{path_to_log_dir}") 
            ]
        else:
            loggers += [ 
                    MLFlowLogger(experiment_name='hf_ehr',
                                    run_name=config.logging.mlflow.name,
                                    log_model='all',
                                    save_dir=f"{path_to_log_dir}",
                                    tracking_uri=f"file:{path_to_log_dir}") 
            ]
            if rank_zero_only.rank == 0:
                # Save mlflow run ID
                mlflow_run_id: str = loggers[-1].run_id
                with open(os.path.join(path_to_log_dir, 'mlflow_run_id.txt'), 'w') as f:
                    f.write(mlflow_run_id)
        if rank_zero_only.rank == 0:
            if not is_resume_from_ckpt:
                mlflow_config = OmegaConf.to_container(config, resolve=True)
                mlflow_config.pop('config', None)
                loggers[-1].log_hyperparams(mlflow_config)

    ## Wandb
    # NOTE: There's a lot of `init()` calls below. Idk why they are all necessary, but they seem to be. Don't remove any!
    run = None
    if is_wandb:
        if is_resume_from_ckpt:
            # Load existing wandb run ID
            with open(os.path.join(path_to_log_dir, 'wandb_run_id.txt'), 'r') as f:
                wandb_run_id: str = f.read()
                
            logger.info(f"Found existing wandb run: `{wandb_run_id}`")

            if rank_zero_only.rank == 0:
                if config.logging.wandb.is_force_create_wandb_run_from_scratch:
                    logger.info(f"Creating new wandb run from scratch")
                    run = wandb.init(
                        entity='ehr-fm',
                        project='hf_ehr', 
                        dir=path_to_log_dir, 
                        name=config.logging.wandb.name,
                        resume='never',
                    )
                    wandb_run_id = run.id
                else:
                    logger.info(f"Restarting wandb run from prior run with id=`{wandb_run_id}`")
                    wandb_relogger = WandbRelogger('hf_ehr', 'ehr-fm')
                    run = wandb_relogger.relog_metrics(path_to_resume_ckpt, path_to_log_dir)
                    wandb_run_id = run.id
            
            loggers += [ 
                WandbLogger(project='hf_ehr',
                            log_model=False,
                            save_dir=path_to_log_dir,
                            resume='allow',
                            id=wandb_run_id)
            ]
        else:
            if rank_zero_only.rank == 0:
                run = wandb.init(
                    entity='ehr-fm',
                    project='hf_ehr', 
                    dir=path_to_log_dir, 
                    name=config.logging.wandb.name
                )
            loggers += [ 
                        WandbLogger(project='hf_ehr',
                                    log_model=False,
                                    save_dir=path_to_log_dir,
                                    name=config.logging.wandb.name)
            ]
            if rank_zero_only.rank == 0:
                # Save wandb run ID
                wandb_run_id: str = run.id
                with open(os.path.join(path_to_log_dir, 'wandb_run_id.txt'), 'w') as f:
                    f.write(wandb_run_id)
        if rank_zero_only.rank == 0:
            if not is_resume_from_ckpt:
                wandb_config = OmegaConf.to_container(config, resolve=True)
                run.config.update(wandb_config)
            run.define_metric('train/loss', summary='min')
            run.define_metric('val/loss', summary='min')

    logger.info("========================== Starting main ==========================")
    logger.info(f">>>> Resuming from CHECKPOINT | Loading from: `{path_to_resume_ckpt}` <<<<" if is_resume_from_ckpt else f">>>> Training from SCRATCH | Saving to: `{path_to_output_dir}` <<<<")

    # Tokenizer
    if config.data.tokenizer.name == 'DescTokenizer':
        # DescEmb
        logger.info(f"Loading DescTokenizer: `{PATH_TO_TOKENIZER_DESC_v8_CONFIG}` using base tokenizer `{config.data.tokenizer.desc_emb_tokenizer}`")
        tokenizer = DescTokenizer( PATH_TO_TOKENIZER_DESC_v8_CONFIG, metadata=tokenizer_metadata)
    elif config.data.tokenizer.name == 'CLMBRTokenizer':
        # CLMBR
        logger.info(f"Loading CLMBRTokenizer: `{PATH_TO_TOKENIZER_CLMBR_v8_CONFIG}`")
        tokenizer = CLMBRTokenizer( PATH_TO_TOKENIZER_CLMBR_v8_CONFIG )
    elif config.data.tokenizer.name == 'CookbookTokenizer':
        # Custom cookbook
        raise ValueError("CookbookTokenizer is not supported in this script yet.")
        logger.info(f"Loading CookbookTokenizer: `{path_to_tokenizer_config}`")
        tokenizer = CookbookTokenizer( path_to_tokenizer_config, metadata=tokenizer_metadata, )
    logger.info(f"Vocab size: all={tokenizer.vocab_size} | non_special_tokens={len(tokenizer.non_special_tokens)} | special_tokens={len(tokenizer.special_tokens)}")

    # Model
    logger.info(f"Loading model: `{model_name}`")
    if 'gpt2' in model_name:
        model = GPTLanguageModel(config, tokenizer.vocab_size, tokenizer.pad_token_id)
    elif 'bert' in model_name:
        model = BERTLanguageModel(config, tokenizer.vocab_size, tokenizer.pad_token_id)
    elif 'hyena' in model_name:
        model = HyenaLanguageModel(config, tokenizer.vocab_size, tokenizer.pad_token_id)
    elif 'mamba' in model_name:
        model = MambaLanguageModel(config, tokenizer.vocab_size, tokenizer.pad_token_id)
    elif 't5' in model_name:
        model = T5LanguageModel(config, tokenizer.vocab_size, tokenizer.pad_token_id)
    else:
        raise ValueError(f"Model `{config.model.name}` not supported.")

    logger.info(f"FLOPs per token of model = {model.flops_per_token}")
    logger.info(f"Parameter count of model = {model.get_param_count()}")
    
    # Datasets
    logger.info(f"Loading FEMR datasets...")
    datasets: Dict[str, FEMRDataset] = load_datasets(config)

    for key, val in datasets.items():
        logger.info(f"{key} dataset size: {len(val)}")
    
    # Dataloaders
    logger.info(f"Loading FEMR dataloaders...")
    dataloaders: Dict[str, DataLoader] = load_dataloaders(config, datasets, tokenizer)

    # Callbacks
    callbacks: List = []
    callbacks += [ 
        # Save top-K checkpoints based on `val/loss`; overwrites old models
        ModelCheckpoint(
            dirpath=path_to_ckpt_dir,
            filename='{epoch}-{step}-val_loss',
            save_top_k=config.callbacks.model_checkpointing.save_top_k,
            every_n_train_steps=config.callbacks.model_checkpointing.most_recent_every_n_train_steps,
            save_weights_only=False, # If False, then save optimizer + scheduler states as well
            monitor='val/loss',
            mode='min',
            verbose=True,
        ),
        # Save checkpoint at end of every epoch
        ModelCheckpoint(
            dirpath=path_to_ckpt_dir,
            filename='{epoch}-epoch-persist',
            save_top_k=-1,
            every_n_epochs=1,
            save_weights_only=False, # If False, then save optimizer + scheduler states as well
            verbose=True,
        ),
        # Save checkpoint every `every_n_train_steps` steps; persists all models
        ModelCheckpoint(
            dirpath=path_to_ckpt_dir,
            filename='{epoch}-{step}-persist',
            save_top_k=-1,
            every_n_train_steps=config.callbacks.model_checkpointing.every_n_train_steps,
            save_weights_only=False, # If False, then save optimizer + scheduler states as well
            verbose=True,
        ),
        # Save most recent `n = save_most_recent_k` checkpoints; overwrites old models
        ModelCheckpoint(
            dirpath=path_to_ckpt_dir,
            filename='{epoch}-{step}-recent',
            save_top_k=config.callbacks.model_checkpointing.save_most_recent_k,
            every_n_train_steps=config.callbacks.model_checkpointing.most_recent_every_n_train_steps,
            save_last=True, # When True, saves an exact copy of the checkpoint to a file last.ckpt whenever a checkpoint file gets saved.
            save_weights_only=False, # If False, then save optimizer + scheduler states as well
            monitor='step',
            mode='max',
            verbose=True,
        ),
        StartTrainingCheckpoint(
            dirpath=path_to_ckpt_dir,
            filename='first',
            save_top_k=1,
            verbose=True,
        )
    ]
    if getattr(config.callbacks.model_checkpointing, 'every_n_train_nonPAD_tokens', None) not in [None, "None"]:
        # Save checkpoint every `every_n_train_nonPAD_tokens` steps; persists all models
        print("Adding MetricBasedCheckpoint for non-PAD tokens...")
        callbacks += [ 
            MetricBasedCheckpoint(
                dirpath=path_to_ckpt_dir,
                metric_name="train/tokens/total_nonPAD",
                is_valid_metric_func=lambda x,y: train_token_metric_func(x, y, config),
                is_run_val=True,
            ),
        ]
    if getattr(config.callbacks.model_checkpointing, 'every_n_flops', None) not in [None, "None"]:
        # Save checkpoint every `every_n_flops` FLOPs; persists all models
        callbacks += [ 
            MetricBasedCheckpoint(
                dirpath=path_to_ckpt_dir,
                metric_name="train/tokens/total_flops",
                is_valid_metric_func=lambda x,y: train_flops_metric_func(x, y, config),
                is_run_val=True,
            ),
        ]
        
    if is_log_grad_norm:
        callbacks += [ GradNormCallback() ]
    
    # Copy artifacts into output directory for reproducibility
    shutil.copy(path_to_tokenizer_config, path_to_artifacts_dir) # save tokenizer code_2_detail
    with open(os.path.join(path_to_artifacts_dir, 'config.yaml'), 'w') as fd: # save config
        OmegaConf.save(config=config, f=fd)

    # Trainer
    trainer = pl.Trainer(
        # profiler='advanced',
        logger=loggers,
        callbacks=callbacks,
        accelerator='gpu',
        devices=config.trainer.devices,
        strategy=config.trainer.distributed_backend,
        limit_train_batches=config.trainer.limit_train_batches,
        limit_val_batches=config.trainer.limit_val_batches,
        log_every_n_steps=config.logging.log_every_n_steps,
        precision="bf16" if torch.cuda.is_bf16_supported() else 16,
        val_check_interval=config.trainer.val_check_interval, # check val set every 10% of training batches (useful for large training datasets, rather than wait for full epoch to finish)
        check_val_every_n_epoch=config.trainer.check_val_every_n_epoch, # log val PPL at end of every epoch
        max_epochs=config.trainer.max_epochs,
        min_epochs=config.trainer.min_epochs,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        gradient_clip_val=config.trainer.gradient_clip_value,
        gradient_clip_algorithm=config.trainer.gradient_clip_algorithm,
        use_distributed_sampler=False if getattr(config.data.dataloader, 'mode', 'batch') == 'approx' else True
    )

    # Run
    try:
        trainer.fit(model, 
                    train_dataloaders=dataloaders['train'],
                    val_dataloaders=dataloaders['val'],
                    ckpt_path=path_to_resume_ckpt)
    except Exception as e:
        print("Exception during trainer.fit:")
        print(e)
        import traceback
        traceback.print_exc()
        exit()
    if is_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
