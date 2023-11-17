import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from jaxtyping import Float
import pytorch_lightning as pl
from typing import Dict, List, Any, Optional, Union, Tuple
from omegaconf import DictConfig
from hf_ehr.models.modules import BaseModel
from torchmetrics.aggregation import SumMetric

class GPTLanguageModel(BaseModel):
    """
    GPT2 with a Language Model head.
    """

    def __init__(self, config: DictConfig, tokenizer) -> None:
        super(GPTLanguageModel, self).__init__(config, tokenizer)
        self.save_hyperparameters()
        self.model_name: str = config.model.name
        self.config = config

        # Model specs
        model_config = AutoConfig.from_pretrained(self.model_name)
        model_config.vocab_size = tokenizer.vocab_size
        model_config.n_positions = config.data.dataloader.max_length
        for key, val in config.model.config_kwargs.items():
            assert hasattr(model_config, key), f"Config for HF model {self.model_name} does not have attribute {key}"
            setattr(model_config, key, val)
        self.hidden_size = model_config.n_embd

        # Model
        self.model = AutoModel.from_config(model_config)
        self.lm_head = nn.Linear(self.hidden_size, tokenizer.vocab_size, bias=False)

        # Tokenizer
        self.tokenizer = tokenizer
        
        # Metrics
        self.sum_metrics: Dict[str, SumMetric] = torch.nn.ModuleDict({
            'train_total_examples': SumMetric(),
            'train_total_tokens_PAD': SumMetric(),
            'train_total_tokens_MASK': SumMetric(),
            'train_total_tokens_nonPAD': SumMetric(),
        })
    
    def forward(self, tokens: Dict[str, Float[torch.Tensor, 'B L']], is_return_hidden_states: bool = True) -> Union[Tuple[Float[torch.Tensor, 'B L V'], Float[torch.Tensor, 'B L H']], Float[torch.Tensor, 'B L V']]:
        B: int = tokens['input_ids'].shape[0]
        L: int = tokens['input_ids'].shape[1]
        H: int = self.hidden_size
        V: int = self.tokenizer.vocab_size
        
        
        hidden_states: Float[torch.Tensor, 'B L H'] = self.model(tokens['input_ids']).last_hidden_state
        logits: Float[torch.Tensor, 'B L V'] = self.lm_head(hidden_states)
        assert hidden_states.shape == (B, L, H)
        assert logits.shape == (B, L, V)

        if is_return_hidden_states:
            return logits, hidden_states
        else:
            return logits

    def loss(self, logits: Float[torch.Tensor, 'B L V'], targets: Float[torch.Tensor, 'B L']) -> torch.Tensor:
        B: int = logits.shape[0]
        L: int = logits.shape[1]
        V: int = logits.shape[2]
        
        # Shift targets to the right
        targets_shifted: Float[torch.Tensor, 'B L-1'] = targets[:, 1:].contiguous()
        
        # Drop last predicted logit (b/c no target exists for it)
        logits_shifted: Float[torch.Tensor, 'B L-1 V'] = logits[:, :-1, :].contiguous()

        # Calculate loss
        loss = F.cross_entropy(logits_shifted.view(-1, logits_shifted.shape[-1]), 
                               targets_shifted.view(-1), 
                               ignore_index=self.tokenizer.pad_token_id, 
                               reduction='mean')
    
        # Sanity checks
        assert logits.shape == (B, L, V)
        assert targets.shape == (B, L)
        assert logits_shifted.shape == (B, L-1, V)
        assert targets_shifted.shape == (B, L-1)

        return loss
    
    def run_eval(self, tokens: Dict[str, Float[torch.Tensor, 'B L']]) -> torch.Tensor:
        B: int = tokens['input_ids'].shape[0]
        L: int = tokens['input_ids'].shape[1]
        V: int = self.tokenizer.vocab_size
        pred_logits: Float[torch.Tensor, 'B L V'] = self.forward(tokens, is_return_hidden_states=False)
        loss: torch.Tensor = self.loss(pred_logits, tokens['input_ids'])
        assert pred_logits.shape == (B, L, V)
        return loss

    def training_step(self, 
                      batch: Dict[str, Any],
                      batch_idx: int) -> torch.Tensor:
        tokens: Dict[str, Float[torch.Tensor, 'B L']] = batch['tokens']
        B: int = tokens['input_ids'].shape[0]

        # Forward pass
        loss: torch.Tensor = self.run_eval(tokens)
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
                        batch_idx: int) -> torch.Tensor:
        tokens: Dict[str, Float[torch.Tensor, 'B L']] = batch['tokens']

        loss: torch.Tensor = self.run_eval(tokens)
        ppl: torch.Tensor = torch.exp(loss).detach()

        # Logging
        self.log('val/loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('val/ppl', torch.clamp(ppl, max=100).to(torch.float32), on_epoch=True, sync_dist=True) # artificially cap to 100 so that charts look prettier

        return loss