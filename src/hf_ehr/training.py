import os
import wandb
import json
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback

from torch.utils.data import DataLoader
from loguru import logger
from typing import Dict, List, Tuple
import hydra
from omegaconf import DictConfig, OmegaConf

from hf_ehr.data.datasets import FEMRDataset, FEMRTokenizer, collate_femr_timelines
from hf_ehr.models.models import GPTLanguageModel

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
        model.log("optim/grad_norm", self.gradient_norm(model))
        
def load_dataloaders(config: DictConfig, datasets: Dict[str, FEMRDataset], tokenizer: FEMRTokenizer) -> Dict[str, DataLoader]:
    batch_size: int = config.data.dataloader.batch_size
    max_length: int = config.data.dataloader.max_length
    is_truncation_random: bool = config.data.dataloader.is_truncation_random
    n_workers: int = config.data.dataloader.n_workers
    seed: int = config.main.seed

    logger.info(f"Loading FEMR dataloaders...")

    train_loader = DataLoader(
        dataset=datasets['train'],
        batch_size=batch_size,
        collate_fn=lambda x: collate_femr_timelines(x, tokenizer, max_length, is_truncation_random, seed),
        num_workers=n_workers,
    )
    val_loader = DataLoader(
        dataset=datasets['val'],
        batch_size=batch_size,
        collate_fn=lambda x: collate_femr_timelines(x, tokenizer, max_length, is_truncation_random, seed),
        num_workers=n_workers,
    )
    test_loader = DataLoader(
        dataset=datasets['test'],
        batch_size=batch_size,
        collate_fn=lambda x: collate_femr_timelines(x, tokenizer, max_length, is_truncation_random, seed),
        num_workers=n_workers,
    )
    return {
        'train' : train_loader,
        'val' : val_loader,
        'test' : test_loader,
    }

def load_datasets(config: DictConfig) -> Dict[str, FEMRDataset]:
    path_to_femr_extract: str = config.data.dataset.path_to_femr_extract

    logger.info(f"Loading FEMR datasets...")
    train_dataset = FEMRDataset(path_to_femr_extract, split='train')
    val_dataset = FEMRDataset(path_to_femr_extract, split='val')
    test_dataset = FEMRDataset(path_to_femr_extract, split='test')
    
    return { 
            'train' : train_dataset, 
            'val' : val_dataset, 
            'test' : test_dataset,
    }
    
@hydra.main(version_base=None, config_path="./configs/", config_name="config")
def main(config: DictConfig) -> None:
    print(config)
    path_to_output_dir: str = config.main.path_to_output_dir
    is_wandb: bool = config.logging.wandb.is_wandb
    model_name: str = config.model.name
    path_to_tokenizer: str = config.data.tokenizer.path_to_tokenizer
    devices: List[int] = config.trainer.devices
    distributed_backend: str = config.trainer.distributed_backend
    min_epochs: int = config.trainer.min_epochs
    max_epochs: int = config.trainer.max_epochs
    limit_train_batches: float = config.trainer.limit_train_batches
    limit_val_batches: float = config.trainer.limit_val_batches
    is_use_amp: bool = config.trainer.is_use_amp
    accumulate_grad_batches: int = config.trainer.accumulate_grad_batches
    gradient_clip_algorithm: str = config.trainer.gradient_clip_algorithm
    gradient_clip_value: float = config.trainer.gradient_clip_value
    seed: int = config.main.seed

    # Random seed
    pl.seed_everything(seed)

    # Paths
    path_to_log_dir: str = os.path.join(path_to_output_dir, 'logs/')
    path_to_ckpt_dir: str = os.path.join(path_to_output_dir, 'ckpts/')
    path_to_log_file: str = os.path.join(path_to_log_dir, 'info.log')
    os.makedirs(path_to_output_dir, exist_ok=True)
    os.makedirs(path_to_log_dir, exist_ok=True)
    os.makedirs(path_to_ckpt_dir, exist_ok=True)
    
    # Logging
    logger.add(path_to_log_file)
    logger.info("Starting main")
    loggers: List = [ TensorBoardLogger(save_dir=path_to_log_dir) ]
    if is_wandb:
        wandb.init(project='hf_ehr', config=OmegaConf.to_container(config), dir=path_to_log_dir)
        loggers += [ 
                    WandbLogger(project='hf_ehr',
                                log_model=False,
                                save_dir=path_to_log_dir)
        ]
        wandb.define_metric('train/loss', summary='min')
        wandb.define_metric('val/loss', summary='min')

    # Tokenizer
    logger.info(f"Loading tokenizer: `{path_to_tokenizer}`")
    femr_vocab_atoi: Dict[str, int] = json.load(open(path_to_tokenizer, 'r'))
    tokenizer = FEMRTokenizer(femr_vocab_atoi)

    # Model
    logger.info(f"Loading model: `{model_name}`")
    model = GPTLanguageModel(config, tokenizer)
    
    # Datasets
    datasets: Dict[str, FEMRDataset] = load_datasets(config)
    
    # Dataloaders
    dataloaders: Dict[str, DataLoader] = load_dataloaders(config, datasets, tokenizer)

    # Trainer
    early_stop_callback = EarlyStopping(
        monitor='val/loss',
        min_delta=0.0,
        patience=config.callbacks.early_stopping.patience,
        verbose=True,
        mode=config.callbacks.early_stopping.metric_mode,
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=path_to_ckpt_dir,
        filename='{epoch}-{val/loss:.2f}',
        save_top_k=config.callbacks.model_checkpointing.save_top_k,
        save_last=True, # When True, saves an exact copy of the checkpoint to a file last.ckpt whenever a checkpoint file gets saved.
        save_weights_only=False, # If TRUE, then save optimizer + scheduler states as well
        monitor='val/loss',
        mode='min',
        verbose=True,
    )
    grad_norm_callback = GradNormCallback()
    trainer = Trainer(
        logger=loggers,
        callbacks=[ checkpoint_callback, grad_norm_callback, ],
        accelerator='gpu',
        devices=devices,
        strategy=distributed_backend,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        log_every_n_steps=1,
        precision='16-mixed' if is_use_amp else '32-true',
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=gradient_clip_value,
        gradient_clip_algorithm=gradient_clip_algorithm,
    )

    # Run
    trainer.fit(model, 
                train_dataloaders=dataloaders['train'],
                val_dataloaders=dataloaders['val'])
    wandb.finish()


if __name__ == "__main__":
    # For Carina to work
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ['HF_DATASETS_CACHE'] = "/share/pi/nigam/mwornow/hf_cache/"
    os.environ['TRANSFORMERS_CACHE'] = "/share/pi/nigam/mwornow/hf_cache/"
    os.environ['HUGGINGFACE_HUB_CACHE'] = "/share/pi/nigam/mwornow/hf_cache/"
    os.environ['HF_HOME'] = "/share/pi/nigam/mwornow/hf_cache/"
    os.environ['WANDB_CACHE_DIR'] = "/share/pi/nigam/mwornow/wandb_cache/"
    os.environ['WANDB_CONFIG_DIR'] = "/share/pi/nigam/mwornow/wandb_config/"
    os.environ['WANDB_DIR'] = "/share/pi/nigam/mwornow/wandb_dir/"
    os.environ['TRITON_CACHE_DIR'] = "/share/pi/nigam/mwornow/triton_cache/"

    # Run
    main()
