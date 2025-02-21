import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from typing import Dict, Any, Optional, Union
from omegaconf import DictConfig
from typing import Dict, Any, Optional
from jaxtyping import Float
from hf_ehr.models.base import BaseModel
from based.models.gpt import GPTLMHeadModel

class BasedLanguageModel(BaseModel):
    """
    Based with a Language Model head.
    """

    def __init__(self, config: DictConfig, vocab_size, pad_token_id) -> None:
        super(BasedLanguageModel, self).__init__(config, vocab_size, pad_token_id)
        
        kwargs = {}

        # Model specs
        model_config = AutoConfig.from_pretrained(config.model.hf_name if hasattr(config.model, 'hf_name') else 'gpt2', **kwargs)
        model_config.vocab_size = vocab_size
        model_config.n_positions = config.data.dataloader.max_length
        for key, val in config.model.config_kwargs.items():
            assert hasattr(model_config, key), f"Config for HF model {config.model.hf_name if hasattr(config.model, 'hf_name') else ''} does not have attribute {key}"
            setattr(model_config, key, val)
        self.hidden_size = model_config.n_embd

        # Model
        if getattr(config.model, 'is_keep_pretrained_weights', False):
            # TODO: Implement loading of pretrained weights
            raise NotImplementedError("Loading of pretrained weights is not yet implemented.")
            self.model = GPTLMHeadModel.from_pretrained(model_config, **kwargs)
        else:
            self.model = GPTLMHeadModel(model_config, **kwargs)

        # Run any post-init handlers from super()
        self.post_init()
    
    def training_step(self, 
                      batch: Dict[str, Any],
                      batch_idx: int) -> Optional[torch.Tensor]:
        self.batch_idx = batch_idx
        tokens: Dict[str, Float[torch.Tensor, 'B L']] = batch['tokens']
        B: int = tokens['input_ids'].shape[0]
        
        outputs = self.model(**tokens)
        logits: Float[torch.Tensor, 'B L V'] = outputs.logits  # shape: [batch_size, seq_len, vocab_size]

        # NOTE: Based doesn't return loss, so manually calculate
        labels: Float[torch.Tensor, 'B L-1'] = tokens['input_ids'][:, 1:]  # shape: [batch_size, seq_len-1]
        shift_logits: Float[torch.Tensor, 'B L-1 V'] = logits[:, :-1, :]  # shape: [batch_size, seq_len-1, vocab_size]
        loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),  # shape: [batch_size * (seq_len-1), vocab_size]
            labels.reshape(-1),  # shape: [batch_size * (seq_len-1)]
            ignore_index=self.pad_token_id  # ignore padding tokens in loss calculation
        )
        
        # Check if loss is NaN and synchronize this information across processes
        if torch.isnan(loss).any():
            nan_detected = torch.tensor([1.0], device=self.device)
        else:
            nan_detected = torch.tensor([0.0], device=self.device)

        if nan_detected.item() == 1:
            print("NaN detected in loss, skipping this batch across all processes.")
            return  # Skip this batch on all processes

        # Learning rate scheduler
        if self.trainer:
            lr: float = self.trainer.lr_scheduler_configs[0].scheduler.optimizer.param_groups[0]["lr"]
            sch = self.lr_schedulers()
            sch.step()
        
        # Logging + Metrics
        self.log_training_step(loss.detach(), B, tokens, lr)

        return loss