import torch
from transformers import AutoConfig, AutoModelForCausalLM
from typing import Dict, List, Any, Optional, Union
from omegaconf import DictConfig
from jaxtyping import Float
from typing import Dict, Any, Optional

from hf_ehr.models.base import BaseModel

class MambaLanguageModel(BaseModel):
    """
   Mamba with a Language Model head.
    """

    def __init__(self, config: DictConfig, vocab_size, pad_token_id) -> None:
        super(MambaLanguageModel, self).__init__(config, vocab_size, pad_token_id)
        
        # Enable faster inference
        if torch.cuda.get_device_capability('cuda')[0] >= 8:
            print("!!!! USING CACHE !!!!")
            kwargs = {
                'torch_dtype': torch.float16,
            }
        else:
            kwargs = {}

        # Model specs
        model_config = AutoConfig.from_pretrained(config.model.hf_name, trust_remote_code=True, **kwargs)
        model_config.vocab_size = vocab_size
        for key, val in config.model.config_kwargs.items():
            assert hasattr(model_config, key), f"Config for HF model {config.model.hf_name if hasattr(config.model, 'hf_name') else ''} does not have attribute {key}"
            setattr(model_config, key, val)
        self.model_config = model_config
        self.hidden_size = model_config.d_model

        # Model
        self.model = AutoModelForCausalLM.from_config(model_config, **kwargs)

        # Run any post-init handlers from super()
        self.post_init()

    def training_step(self, 
                      batch: Dict[str, Any],
                      batch_idx: int) -> Optional[torch.Tensor]:
        tokens: Dict[str, Float[torch.Tensor, 'B L']] = batch['tokens']
        B: int = tokens['input_ids'].shape[0]

        outputs = self.model(**tokens)
        loss: torch.Tensor = outputs.loss
        
        # Learning rate scheduler
        lr: float = self.trainer.lr_scheduler_configs[0].scheduler.optimizer.param_groups[0]["lr"]
        sch = self.lr_schedulers()
        sch.step()
        
        # Logging + Metrics
        self.log_training_step(loss.detach(), B, tokens, lr)

        return loss

if __name__ == '__main__':
    model = MambaLanguageModel()
    
    outputs = model.model(**tokens)
    model.model.backward(outputs.loss)
