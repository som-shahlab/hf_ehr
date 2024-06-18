import torch

from transformers import AutoModelForMaskedLM, AutoConfig
from jaxtyping import Float
from typing import Dict, Any, Optional, Union
from omegaconf import DictConfig

from hf_ehr.models.modules import BaseModel
from hf_ehr.data.datasets import FEMRTokenizer, DescTokenizer


class BERTLanguageModel(BaseModel):
    """
    BERT with a Language Model head.
    """

    def __init__(self, config: DictConfig, tokenizer: Union[FEMRTokenizer, DescTokenizer]) -> None:
        super(BERTLanguageModel, self).__init__(config, tokenizer)

        # Model specs
        model_config = AutoConfig.from_pretrained(config.model.hf_name if hasattr(config.model, 'hf_name') else 'bert-base-uncased')
        model_config.vocab_size = tokenizer.vocab_size
        model_config.n_positions = config.data.dataloader.max_length
        for key, val in config.model.config_kwargs.items():
            assert hasattr(model_config, key), f"Config for HF model {config.model.hf_name if hasattr(config.model, 'hf_name') else ''} does not have attribute {key}"
            setattr(model_config, key, val)
        self.model_config = model_config
        self.hidden_size = model_config.n_embd if hasattr(model_config, 'n_embd') else model_config.hidden_size

        # Model
        self.model = AutoModelForMaskedLM.from_config(model_config)
        self.flops_per_token: Optional[int] = self.calculate_flops_per_token(tokenizer)

    def training_step(self, 
                      batch: Dict[str, Any],
                      batch_idx: int) -> Optional[torch.Tensor]:
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
