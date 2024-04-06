import torch
import torch.nn as nn
from torch import optim
from transformers import AutoModel, AutoConfig
from omegaconf import DictConfig
from typing import Dict, Any, Optional
from hf_ehr.models.modules import BaseModel
from hf_ehr.utils import lr_warmup_with_constant_plateau

class HyenaLanguageModel(BaseModel):
    """
    Hyena with a Language Model head.
    """

    def __init__(self, config: DictConfig, tokenizer) -> None:
        super(HyenaLanguageModel, self).__init__(config, tokenizer)

        # Model specs
        model_config = AutoConfig.from_pretrained(config.model.hf_name, trust_remote_code=True)
        model_config.vocab_size = tokenizer.vocab_size
        for key, val in config.model.config_kwargs.items():
            assert hasattr(model_config, key), f"Config for HF model {self.model_name} does not have attribute {key}"
            setattr(model_config, key, val)
        self.hidden_size = model_config.d_model

        # Model
        self.model = AutoModel.from_config(model_config, trust_remote_code=True)
        self.lm_head = nn.Linear(self.hidden_size, tokenizer.vocab_size, bias=False)

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
    def training_step(self, 
                      batch: Dict[str, Any],
                      batch_idx: int) -> Optional[torch.Tensor]:
        # TODO (@Suhana) -- adapt for Hyena
        tokens: Dict[str, Float[torch.Tensor, 'B L']] = batch['tokens']
        B: int = tokens['input_ids'].shape[0]

        outputs = self.model(**tokens)
        loss: torch.Tensor = outputs.loss
        ppl: torch.Tensor = torch.exp(loss).detach()
        
        # Learning rate scheduler
        lr: float = self.trainer.lr_scheduler_configs[0].scheduler.optimizer.param_groups[0]["lr"]
        sch = self.lr_schedulers()
        sch.step()
        
        # Logging + Metrics
        self.log_training_step(loss.detach(), B, tokens, lr)

        return loss