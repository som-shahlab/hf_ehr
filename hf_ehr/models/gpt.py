import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoConfig
from typing import Dict, Any, Optional, Union
from omegaconf import DictConfig
from typing import Dict, Any, Optional
from jaxtyping import Float
from lightning.pytorch.utilities import rank_zero_only

from hf_ehr.models.modules import BaseModel

class GPTLanguageModel(BaseModel):
    """
    GPT with a Language Model head.
    """

    def __init__(self, config: DictConfig, vocab_size, pad_token_id) -> None:
        super(GPTLanguageModel, self).__init__(config, vocab_size, pad_token_id)

        # Model specs
        model_config = AutoConfig.from_pretrained(config.model.hf_name if hasattr(config.model, 'hf_name') else 'gpt2')
        model_config.vocab_size = vocab_size
        model_config.n_positions = config.data.dataloader.max_length
        for key, val in config.model.config_kwargs.items():
            assert hasattr(model_config, key), f"Config for HF model {config.model.hf_name if hasattr(config.model, 'hf_name') else ''} does not have attribute {key}"
            setattr(model_config, key, val)
        self.hidden_size = model_config.n_embd

        # Model
        self.model = AutoModelForCausalLM.from_config(model_config)
        
        # Run any post-init handlers from super()
        self.post_init()

    def training_step(self, 
                      batch: Dict[str, Any],
                      batch_idx: int) -> Optional[torch.Tensor]:
        print("TRAINING STEP 1", batch_idx, torch.cuda.current_device())
        tokens: Dict[str, Float[torch.Tensor, 'B L']] = batch['tokens']
        print("TRAINING STEP 2", batch_idx, torch.cuda.current_device())
        B: int = tokens['input_ids'].shape[0]
        print("TRAINING STEP 3", batch_idx, torch.cuda.current_device())

        print("TRAINING STEP RUN MODEL", batch_idx, tokens['input_ids'].shape, torch.cuda.current_device())
        outputs = self.model(**tokens)
        print("MODEL DONE", batch_idx, torch.cuda.current_device())
        loss: torch.Tensor = outputs.loss
        print("LOSS", loss, torch.cuda.current_device())
        
        # Check if loss is NaN and handle it
        # Check if loss is NaN and synchronize this information across processes
        if torch.isnan(loss).any():
            nan_detected = torch.tensor([1.0], device=self.device)
        else:
            nan_detected = torch.tensor([0.0], device=self.device)

        print("START ALL REDUCE", torch.cuda.current_device())
        dist.all_reduce(nan_detected, op=dist.ReduceOp.MAX)
        print("END ALL REDUCE", torch.cuda.current_device())

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
        print("TRAINING STEP LOSS", batch_idx, loss)

        return loss
