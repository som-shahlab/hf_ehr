import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from transformers import AutoModel, AutoConfig
from jaxtyping import Float
import pytorch_lightning as pl
from typing import Dict, List, Any, Optional, Union, Tuple
from omegaconf import DictConfig
from torchmetrics.aggregation import SumMetric
from hf_ehr.utils import lr_warmup_with_constant_plateau

class BaseModel(pl.LightningModule):
    """
    Base PyTorchLightning model with some common methods.
    """

    def __init__(self, config: DictConfig, tokenizer) -> None:
        super(BaseModel, self).__init__()
        self.save_hyperparameters()
        self.model_name: str = config.model.name
        self.config = config

    def parameters(self):
        return list(self.model.parameters()) + list(self.lm_head.parameters())

    def configure_optimizers(self):
        """ Sets Learning rate for different parameter groups."""
        lr: float = self.config.trainer.optimizer.lr

        # Optimizer
        optimizer = optim.Adam(self.parameters(), lr=lr)
                
        # Scheduler
        if self.config.trainer.scheduler:
            scheduler = lr_warmup_with_constant_plateau(optimizer, 
                                                        num_warmup_steps=self.config.trainer.scheduler.num_warmup_steps, 
                                                        num_decay_steps=self.config.trainer.scheduler.num_decay_steps, 
                                                        initial_lr=self.config.trainer.scheduler.initial_lr, 
                                                        final_lr=self.config.trainer.scheduler.final_lr)


            return [ optimizer ], [ scheduler ]

        return [optimizer]
    
    def on_save_checkpoint(self, checkpoint):
        """Save each metric's state in the checkpoint."""
        for key, metric in self.sum_metrics.items():
            checkpoint[key] = metric.compute()

    def on_load_checkpoint(self, checkpoint):
        """Load each metric's state in the checkpoint."""
        for key in self.sum_metrics.keys():
            if key in checkpoint:
                self.sum_metrics[key] = SumMetric()
                # Need to rescale SumMetric loaded from checkpoint since its saved value is summed across all GPUs,
                # but this `on_load_checkpoint()` gets called per GPU. Need to do this quotient/remainder thing in
                # case the SumMetric's valuvalue is not divisible by the # of GPUs
                remainder: int = checkpoint[key] % (self.trainer.num_devices * self.trainer.num_nodes)
                quotient: int = checkpoint[key] // (self.trainer.num_devices * self.trainer.num_nodes)
                self.sum_metrics[key].update(quotient + (remainder if self.trainer.global_rank == 0 else 0))