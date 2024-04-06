import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoConfig
from jaxtyping import Float
from typing import Dict, List, Any, Optional, Union, Tuple
from omegaconf import DictConfig
from hf_ehr.models.modules import BaseModel
import torch.distributed as torch_dist

class BERTLanguageModel(BaseModel):
    """
    BERT with a Language Model head.
    """

    def __init__(self, config: DictConfig, tokenizer) -> None:
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
        
        # Metrics
        train_batch_examples: int = B
        train_batch_tokens_PAD: torch.Tensor = (1 - tokens['attention_mask']).sum()
        train_batch_tokens_nonPAD: torch.Tensor = tokens['attention_mask'].sum()
        self.sum_metrics['train_total_examples'].update(train_batch_examples)
        self.sum_metrics['train_total_tokens_PAD'].update(train_batch_tokens_PAD)
        self.sum_metrics['train_total_tokens_nonPAD'].update(train_batch_tokens_nonPAD)

        # Logging
        self.log('optim/lr', lr)
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/ppl', torch.clamp(ppl, max=100).to(torch.float32)) # artificially cap to 100 so that charts look prettier
        self.log('train/examples/batch', torch.tensor(B, dtype=torch.float32))
        self.log('train/examples/total', self.sum_metrics['train_total_examples'].compute().to(torch.float32))
        self.log('train/tokens/batch_all', (train_batch_tokens_PAD + train_batch_tokens_nonPAD).to(torch.float32))
        self.log('train/tokens/batch_PAD', train_batch_tokens_PAD.to(torch.float32))
        self.log('train/tokens/batch_nonPAD', train_batch_tokens_nonPAD.to(torch.float32))
        self.log('train/tokens/total_all', (self.sum_metrics['train_total_tokens_PAD'].compute() + self.sum_metrics['train_total_tokens_nonPAD'].compute()).to(torch.float32))
        self.log('train/tokens/total_PAD', self.sum_metrics['train_total_tokens_PAD'].compute().to(torch.float32))
        self.log('train/tokens/total_nonPAD', self.sum_metrics['train_total_tokens_nonPAD'].compute().to(torch.float32))

        return loss

    def validation_step(self, 
                        batch: Dict[str, Any],
                        batch_idx: int) -> Optional[torch.Tensor]:
        tokens: Dict[str, Float[torch.Tensor, 'B L']] = batch['tokens']
        B: int = tokens['input_ids'].shape[0]
        
        # Forward pass
        outputs = self.model(**tokens)
        loss: torch.Tensor = outputs.loss
        ppl: torch.Tensor = torch.exp(loss).detach()

        # Logging
        self.log('val/loss', loss.detach(), prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('val/ppl', torch.clamp(ppl, max=100).to(torch.float32), on_epoch=True, sync_dist=True) # artificially cap to 100 so that charts look prettier
        return loss