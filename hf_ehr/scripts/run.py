import time
start = time.time()
import os
import json
import hydra
import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger, MLFlowLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Callback
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.profilers import PyTorchProfiler

from loguru import logger
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from omegaconf import DictConfig, OmegaConf

from hf_ehr.data.datasets import FEMRDataset, FEMRTokenizer
from hf_ehr.models.bert import BERTLanguageModel
from hf_ehr.models.gpt import GPTLanguageModel
from hf_ehr.models.hyena import HyenaLanguageModel
from hf_ehr.models.mamba import MambaLanguageModel
from hf_ehr.models.t5 import T5LanguageModel
from hf_ehr.trainer.loaders import load_datasets, load_dataloaders
print("====> Done loading imports: ", time.time() - start, "s")

V100_BASE_DIR: str = '/local-scratch-nvme/nigam/hf_ehr/'
A100_BASE_DIR: str = '/local-scratch/nigam/hf_ehr/'
GPU_BASE_DIR: str = '/home/hf_ehr/'

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

def rewrite_paths_for_carina_from_config(config: DictConfig) -> DictConfig:
    """Rewrite paths for Carina partitions to use local-scratch directories."""
    if os.environ.get('SLURM_JOB_PARTITION') == 'nigam-v100':
        if not os.path.exists(V100_BASE_DIR):
            os.makedirs(V100_BASE_DIR, exist_ok=True)
            os.system(f'cp -r /share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_08_13_extract_v9_lite {V100_BASE_DIR}')
            os.system(f'cp /share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9_lite/code_2_int.json {V100_BASE_DIR}')
            os.system(f'cp /share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9_lite/code_2_count.json {V100_BASE_DIR}')
        config.data.tokenizer.path_to_code_2_int = config.data.tokenizer.path_to_code_2_int.replace('/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9_lite/', V100_BASE_DIR)
        config.data.tokenizer.path_to_code_2_count = config.data.tokenizer.path_to_code_2_count.replace('/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9_lite/', V100_BASE_DIR)
        config.data.dataset.path_to_femr_extract = config.data.dataset.path_to_femr_extract.replace('/share/pi/nigam/data/', V100_BASE_DIR)
        print(f"Loading data from local-scratch: `{V100_BASE_DIR}`.")
    elif os.environ.get('SLURM_JOB_PARTITION') == 'nigam-a100':
        if not os.path.exists(A100_BASE_DIR):
            # Copy over the cache files
            os.makedirs(A100_BASE_DIR, exist_ok=True)
            os.system(f'cp -r /share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_08_13_extract_v9_lite {A100_BASE_DIR}')
            os.system(f'cp /share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9_lite/code_2_int.json {A100_BASE_DIR}')
            os.system(f'cp /share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9_lite/code_2_count.json {A100_BASE_DIR}')
        config.data.tokenizer.path_to_code_2_int = config.data.tokenizer.path_to_code_2_int.replace('/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9_lite/', A100_BASE_DIR)
        config.data.tokenizer.path_to_code_2_count = config.data.tokenizer.path_to_code_2_count.replace('/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9_lite/', A100_BASE_DIR)
        config.data.dataset.path_to_femr_extract = config.data.dataset.path_to_femr_extract.replace('/share/pi/nigam/data/', A100_BASE_DIR)
        print(f"Loading data from local-scratch: `{A100_BASE_DIR}`.")
    elif os.environ.get('SLURM_JOB_PARTITION') == 'gpu':
        if not os.path.exists(GPU_BASE_DIR):
            os.makedirs(GPU_BASE_DIR, exist_ok=True)
            os.system(f'cp -r /share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_08_13_extract_v9_lite {GPU_BASE_DIR}')
            os.system(f'cp /share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9_lite/code_2_int.json {GPU_BASE_DIR}')
            os.system(f'cp /share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9_lite/code_2_count.json {GPU_BASE_DIR}')
        config.data.tokenizer.path_to_code_2_int = config.data.tokenizer.path_to_code_2_int.replace('/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9_lite/', GPU_BASE_DIR)
        config.data.tokenizer.path_to_code_2_count = config.data.tokenizer.path_to_code_2_count.replace('/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9_lite/', GPU_BASE_DIR)
        config.data.dataset.path_to_femr_extract = config.data.dataset.path_to_femr_extract.replace('/share/pi/nigam/data/', GPU_BASE_DIR)
        print(f"Loading data from local-scratch: `{GPU_BASE_DIR}`.")
    else:
        print("No local-scratch directory found. Using default `/share/pi/` paths.")
    return config

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
    path_to_tokenizer_code_2_int: str = config.data.tokenizer.path_to_code_2_int
    path_to_tokenizer_code_2_count: str = config.data.tokenizer.path_to_code_2_count
    tokenizer_min_code_count: Optional[int] = config.data.tokenizer.min_code_count
    seed: int = config.main.seed

    # Random seed
    pl.seed_everything(seed, workers=True)
    
    # Check if resuming from checkpoint
    is_resume_from_ckpt: bool = os.path.exists(os.path.join(path_to_output_dir, 'ckpts/last.ckpt'))
    path_to_resume_ckpt: Optional[str] = os.path.join(path_to_output_dir, 'ckpts/last.ckpt') if is_resume_from_ckpt else None

    # Paths
    path_to_log_dir: str = os.path.join(path_to_output_dir, 'logs/')
    path_to_ckpt_dir: str = os.path.join(path_to_output_dir, 'ckpts/')
    path_to_log_file: str = os.path.join(path_to_log_dir, 'info.log')
    os.makedirs(path_to_output_dir, exist_ok=True)
    os.makedirs(path_to_log_dir, exist_ok=True)
    os.makedirs(path_to_ckpt_dir, exist_ok=True)
    
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
                                    #log_checkpoint=True,
                                    log_model='all',
                                    save_dir=f"{path_to_log_dir}",
                                    tracking_uri=f"file:{path_to_log_dir}") 
            ]
        else:
            loggers += [ 
                    MLFlowLogger(experiment_name='hf_ehr',
                                    run_name=config.logging.mlflow.name,
                                    #log_checkpoint=True,
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
    if is_wandb:
        import wandb
        if is_resume_from_ckpt:
            # Load existing wandb run ID
            with open(os.path.join(path_to_log_dir, 'wandb_run_id.txt'), 'r') as f:
                wandb_run_id: str = f.read()
            logger.info(f"Found existing wandb run: `{wandb_run_id}`")
            loggers += [ 
                        WandbLogger(project='hf_ehr',
                                    log_model=False,
                                    save_dir=path_to_log_dir,
                                    resume='allow',
                                    id=wandb_run_id)
            ]
        else:
            wandb.init()
            loggers += [ 
                        WandbLogger(project='hf_ehr',
                                    log_model=False,
                                    save_dir=path_to_log_dir,
                                    name=config.logging.wandb.name)
            ]
            if rank_zero_only.rank == 0:
                # Save wandb run ID
                wandb_run_id: str = wandb.run.id
                with open(os.path.join(path_to_log_dir, 'wandb_run_id.txt'), 'w') as f:
                    f.write(wandb_run_id)
        if rank_zero_only.rank == 0:
            if not is_resume_from_ckpt:
                wandb_config = OmegaConf.to_container(config, resolve=True)
                # wandb_config.pop('config', None)
                wandb.config.update(wandb_config)
            wandb.init()
            wandb.define_metric('train/loss', summary='min')
            wandb.define_metric('val/loss', summary='min')

    logger.info("========================== Starting main ==========================")
    logger.info(f">>>> Resuming from CHECKPOINT | Loading from: `{path_to_resume_ckpt}` <<<<" if is_resume_from_ckpt else f">>>> Training from SCRATCH | Saving to: `{path_to_output_dir}` <<<<")
    
    # Tokenizer
    logger.info(f"Loading tokenizer: `{path_to_tokenizer_code_2_int}`")
    femr_vocab_count: Dict[str, int] = json.load(open(path_to_tokenizer_code_2_count, 'r'))
    tokenizer = FEMRTokenizer(femr_vocab_count, min_code_count=tokenizer_min_code_count)
    logger.info(f"Vocab size: `{tokenizer.vocab_size}`")

    # Model
    logger.info(f"Loading model: `{model_name}`")
    if 'gpt2' in model_name:
        model = GPTLanguageModel(config, tokenizer)
    elif 'bert' in model_name:
        model = BERTLanguageModel(config, tokenizer)
    elif 'hyena' in model_name:
        model = HyenaLanguageModel(config, tokenizer)
    elif 'mamba' in model_name:
        model = MambaLanguageModel(config, tokenizer)
    elif 't5' in model_name:
        model = T5LanguageModel(config, tokenizer)
    else:
        raise ValueError(f"Model `{config.model.name}` not supported.")
    #logger.info(f"Parameter count of model = {model.get_param_count()}")
    
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
            filename='{epoch}-epoch',
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
        )
    ]
    if is_log_grad_norm:
        callbacks += [ GradNormCallback() ]
    
    profiler = PyTorchProfiler(
        dirpath=os.path.join(path_to_output_dir),
        filename='pytorch_profiler',
        emit_nvtx=True,
        # on_trace_ready = torch.profiler.tensorboard_trace_handler(os.path.join(path_to_output_dir, "profiler/tb_logs")),
        trace_memory=True,
        export_to_chrome=True,
        # schedule = torch.profiler.schedule(skip_first=10, wait=1, warmup=1, active=20)
    )
    """
    profiler = PyTorchProfiler(
        # You can specify various options here, such as:
        profile_memory=True,  # Whether to report tensor memory allocation/deallocation.
        with_stack=True,  # Whether to record source information.
        record_shapes=True,  # Whether to record tensor shapes.
        profile_mlops=True, # Whether to profile model operations.
        export_to_chrome=True,
        # There are other parameters available depending on your needs.
    )
    """

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
        precision="bf16" if config.trainer.is_use_bf16 else (16 if config.trainer.is_use_fp16 else 32),
        val_check_interval=config.trainer.val_check_interval, # check val set every 10% of training batches (useful for large training datasets, rather than wait for full epoch to finish)
        max_epochs=config.trainer.max_epochs,
        min_epochs=config.trainer.min_epochs,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        gradient_clip_val=config.trainer.gradient_clip_value,
        gradient_clip_algorithm=config.trainer.gradient_clip_algorithm,
    )

    # Run
    trainer.fit(model, 
                train_dataloaders=dataloaders['train'],
                val_dataloaders=dataloaders['val'],
                ckpt_path=path_to_resume_ckpt)
    if is_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()