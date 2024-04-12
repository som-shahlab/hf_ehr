import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from typing import Dict, List, Any, Optional, Union, Tuple
from omegaconf import DictConfig
from typing import Dict, Any, Optional
from hf_ehr.models.modules import BaseModel
from jaxtyping import Float

class GPTLanguageModel(BaseModel):
    """
    GPT2 with a Language Model head.
    """

    def __init__(self, config: DictConfig, tokenizer) -> None:
        super(GPTLanguageModel, self).__init__(config, tokenizer)

        # Model specs
        model_config = AutoConfig.from_pretrained(config.model.hf_name if hasattr(config.model, 'hf_name') else 'gpt2')
        model_config.vocab_size = tokenizer.vocab_size
        model_config.n_positions = config.data.dataloader.max_length
        for key, val in config.model.config_kwargs.items():
            assert hasattr(model_config, key), f"Config for HF model {config.model.hf_name if hasattr(config.model, 'hf_name') else ''} does not have attribute {key}"
            setattr(model_config, key, val)
        self.hidden_size = model_config.n_embd

        # Model
        self.model = AutoModelForCausalLM.from_config(model_config)

    def training_step(self, 
                      batch: Dict[str, Any],
                      batch_idx: int) -> Optional[torch.Tensor]:
        tokens: Dict[str, Float[torch.Tensor, 'B L']] = batch['tokens']
        B: int = tokens['input_ids'].shape[0]
        
        print(torch.distributed.get_rank(), "|", batch_idx, "|", tokens['input_ids'].shape, tokens['attention_mask'].sum().item(), tokens['input_ids'][0, :10])

        outputs = self.model(**tokens)
        loss: torch.Tensor = outputs.loss
        
        # Learning rate scheduler
        lr: float = self.trainer.lr_scheduler_configs[0].scheduler.optimizer.param_groups[0]["lr"]
        sch = self.lr_schedulers()
        sch.step()
        
        # Logging + Metrics
        self.log_training_step(loss.detach(), B, tokens, lr)

        return loss