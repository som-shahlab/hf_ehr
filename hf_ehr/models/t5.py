import torch
from transformers import AutoModelForSeq2SeqLM, AutoConfig
from omegaconf import DictConfig
from typing import Union, Dict, Any, Optional
from jaxtyping import Float

from hf_ehr.models.base import BaseModel

class T5LanguageModel(BaseModel):
    """
    T5 with a Language Model head.
    """

    def __init__(self, config: DictConfig, vocab_size, pad_token_id) -> None:
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
        
        # Run any post-init handlers from super()
        self.post_init()

    def training_step(self, 
                      batch: Dict[str, Any],
                      batch_idx: int) -> Optional[torch.Tensor]:
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
