import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoConfig
from omegaconf import DictConfig
# from hf_ehr.models.modules import CausalModel
from hf_ehr.models.modules import BaseModel
from typing import Union, Tuple, Dict, List, Any, Optional
from jaxtyping import Float

class T5LanguageModel(BaseModel):
    """
    T5 with a Language Model head.
    """

    def __init__(self, config: DictConfig, vocab_size: int, pad_token_id: int) -> None:
        super(T5LanguageModel, self).__init__(config, vocab_size, pad_token_id)

        # Model specs
        model_config = AutoConfig.from_pretrained(config.model.hf_name if hasattr(config.model, 'hf_name') else 't5-base')
        model_config.vocab_size = vocab_size
        for key, val in config.model.config_kwargs.items():
            assert hasattr(model_config, key), f"Config for HF model {self.model_name} does not have attribute {key}"
            setattr(model_config, key, val)
        self.hidden_size = model_config.d_model

        # Model
        self.model = AutoModelForSeq2SeqLM.from_config(model_config)
        
    def training_step(self, 
                      batch: Dict[str, Any],
                      batch_idx: int) -> Optional[torch.Tensor]:
        # TODO (@Miguel) -- adapt for T5
        tokens: Dict[str, Float[torch.Tensor, 'B L']] = batch['tokens']
        del tokens['token_type_ids']
        B: int = tokens['input_ids'].shape[0]
        outputs = self.model(**tokens)
        loss: torch.Tensor = outputs.loss
        lr: float = self.trainer.lr_scheduler_configs[0].scheduler.optimizer.param_groups[0]["lr"]
        sch = self.lr_schedulers()
        sch.step()
        
        # Logging + Metrics
        self.log_training_step(loss.detach(), B, tokens, lr)

        return loss
    
    def validation_step(self, 
                        batch: Dict[str, Any],
                        batch_idx: int) -> Optional[torch.Tensor]:
        tokens: Dict[str, Float[torch.Tensor, 'B L']] = batch['tokens']
        B: int = tokens['input_ids'].shape[0]
        
        # Forward pass
        del tokens['token_type_ids']
        outputs = self.model(**tokens)
        loss: torch.Tensor = outputs.loss

        # Logging
        self.log_validation_step(loss.detach())

        return loss
