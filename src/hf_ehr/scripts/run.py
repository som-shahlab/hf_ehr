import os
import wandb
import json
import hydra
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
from pytorch_lightning.utilities import rank_zero_only

from loguru import logger
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from omegaconf import DictConfig, OmegaConf

from hf_ehr.data.datasets import FEMRDataset, FEMRTokenizer
from hf_ehr.models.bert import BERTLanguageModel
from hf_ehr.models.gpt import GPTLanguageModel
from hf_ehr.trainer.loaders import load_datasets, load_dataloaders

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
    
@hydra.main(version_base=None, config_path='../configs/', config_name="config")
def main(config: DictConfig) -> None:
    print(config)
    path_to_output_dir: str = config.main.path_to_output_dir
    is_wandb: bool = config.logging.wandb.is_wandb
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
    if is_wandb:
        if is_resume_from_ckpt:
            # Load existing wandb run ID
            with open(os.path.join(path_to_log_dir, 'wandb_run_id.txt'), 'r') as f:
                wandb_run_id: str = f.read()
            logger.info(f"Found existing wandb run: `{wandb_run_id}`")
            loggers += [ 
                        WandbLogger(project='hf_ehr',
                                    log_model=False,
                                    save_dir=path_to_log_dir,
                                    resume='must',
                                    id=wandb_run_id)
            ]
        else:
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
                wandb_config.pop('config', None)
                wandb.config.update(wandb_config)
            wandb.define_metric('train/loss', summary='min')
            wandb.define_metric('val/loss', summary='min')

    logger.info("========================== Starting main ==========================")
    logger.info(f">>>> Resuming from CHECKPOINT | Loading from: `{path_to_resume_ckpt}` <<<<" if is_resume_from_ckpt else f">>>> Training from SCRATCH | Saving to: `{path_to_output_dir}` <<<<")
    
    # Tokenizer
    logger.info(f"Loading tokenizer: `{path_to_tokenizer_code_2_int}`")
    femr_vocab_atoi: Dict[str, int] = json.load(open(path_to_tokenizer_code_2_int, 'r'))
    femr_vocab_count: Dict[str, int] = json.load(open(path_to_tokenizer_code_2_count, 'r'))
    tokenizer = FEMRTokenizer(femr_vocab_atoi, femr_vocab_count, min_code_count=tokenizer_min_code_count)
    logger.info(f"Vocab size: `{tokenizer.vocab_size}`")

    # Model
    logger.info(f"Loading model: `{model_name}`")
    if 'gpt2' in config.model.name:
        model = GPTLanguageModel(config, tokenizer)
    elif 'bert' in config.model.name:
        model = BERTLanguageModel(config, tokenizer)
    else:
        raise ValueError(f"Model `{config.model.name}` not supported.")

    # Datasets
    logger.info(f"Loading FEMR datasets...")
    datasets: Dict[str, FEMRDataset] = load_datasets(config)
    
    # Dataloaders
    logger.info(f"Loading FEMR dataloaders...")
    dataloaders: Dict[str, DataLoader] = load_dataloaders(config, datasets, tokenizer)

    # Callbacks
    callbacks: List = []
    early_stop_callback = EarlyStopping(
        monitor='val/loss',
        min_delta=0.0,
        patience=config.callbacks.early_stopping.patience,
        verbose=True,
        mode=config.callbacks.early_stopping.metric_mode,
    )
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
            filename='{epoch}-{step}-epoch',
            save_top_k=-1,
            every_n_epochs=1,
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
    
    # from pytorch_lightning.profilers import PyTorchProfiler
    # profiler = PyTorchProfiler(
    #         export_to_chrome=True,
    # )

    # Trainer
    trainer = pl.Trainer(
        # profiler=profiler,
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
    wandb.finish()


if __name__ == "__main__":
    # For Carina to work (otherwise get a ton of Disk space out of memory errors b/c will write to /home/mwornow/.local/ which is space limited)
    os.environ['HF_DATASETS_CACHE'] = "/share/pi/nigam/mwornow/hf_cache/"
    os.environ['TRANSFORMERS_CACHE'] = "/share/pi/nigam/mwornow/hf_cache/"
    os.environ['HUGGINGFACE_HUB_CACHE'] = "/share/pi/nigam/mwornow/hf_cache/"
    os.environ['HF_HOME'] = "/share/pi/nigam/mwornow/hf_cache/"
    os.environ['WANDB_CACHE_DIR'] = "/share/pi/nigam/mwornow/wandb_cache/"
    os.environ['WANDB_CONFIG_DIR'] = "/share/pi/nigam/mwornow/wandb_cache/"
    os.environ['WANDB_DATA_DIR'] = "/share/pi/nigam/mwornow/wandb_cache/"
    os.environ['WANDB_ARTIFACT_DIR'] = "/share/pi/nigam/mwornow/wandb_cache/"
    os.environ['WANDB_DIR'] = "/share/pi/nigam/mwornow/wandb_cache/"
    os.environ['TRITON_CACHE_DIR'] = "/share/pi/nigam/mwornow/triton_cache/"
    os.environ['HF_CACHE_DIR'] = "/share/pi/nigam/mwornow/hf_cache/"

    # Run
    main()