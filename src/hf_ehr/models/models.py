import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from transformers import AutoModel, AutoConfig
from jaxtyping import Float
import pytorch_lightning as pl
from typing import Dict, List, Any, Optional, Union, Tuple
from omegaconf import DictConfig
from torchmetrics.aggregation import SumMetric
from hf_ehr.utils import lr_warmup_with_constant_plateau

class GPTLanguageModel(pl.LightningModule):
    """
    Sample model to show how to train GPT2 with a Language Model head.
    """

    def __init__(self, config: DictConfig, tokenizer) -> None:
        super(GPTLanguageModel, self).__init__()
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
        self.train_total_examples = SumMetric()
        self.train_total_tokens_PAD = SumMetric()
        self.train_total_tokens_nonPAD = SumMetric()

    def forward(self, tokens: Dict[str, Float[torch.Tensor, 'B L']]) -> Float[torch.Tensor, 'B L V']:
        B: int = tokens['input_ids'].shape[0]
        L: int = tokens['input_ids'].shape[1]
        H: int = self.hidden_size
        V: int = self.tokenizer.vocab_size
        
        hidden_states: Float[torch.Tensor, 'B L H'] = self.model(tokens['input_ids']).last_hidden_state
        logits: Float[torch.Tensor, 'B L V'] = self.lm_head(hidden_states)
        assert hidden_states.shape == (B, L, H)
        assert logits.shape == (B, L, V)

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

    def training_step(self, 
                      batch: Dict[str, Float[torch.Tensor, 'B L']],
                      batch_idx: int) -> torch.Tensor:
        B: int = batch['input_ids'].shape[0]
        L: int = batch['input_ids'].shape[1]
        V: int = self.tokenizer.vocab_size
        
        # Forward pass
        pred_logits: Float[torch.Tensor, 'B L V'] = self.forward(batch)
        loss: torch.Tensor = self.loss(pred_logits, batch['input_ids'])
        ppl: torch.Tensor = torch.exp(loss).detach()
        
        # Learning rate scheduler
        lr: float = self.trainer.lr_scheduler_configs[0].scheduler.optimizer.param_groups[0]["lr"]
        sch = self.lr_schedulers()
        sch.step()

        # Sanity checks
        assert pred_logits.shape == (B, L, V)
        
        # Metrics
        train_batch_examples: int = B
        train_batch_tokens_PAD: torch.Tensor = batch['attention_mask'].sum()
        train_batch_tokens_nonPAD: torch.Tensor = (1 - batch['attention_mask']).sum()
        self.train_total_examples.update(train_batch_examples)
        self.train_total_tokens_PAD.update(train_batch_tokens_PAD)
        self.train_total_tokens_nonPAD.update(train_batch_tokens_nonPAD)

        # Logging
        self.log('optim/lr', lr)
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/ppl', torch.tensor(min(ppl, 100), dtype=torch.float32)) # artificially cap to 100 so that charts look prettier
        self.log('train/examples/batch', torch.tensor(B, dtype=torch.float32))
        self.log('train/examples/total', self.train_total_examples.compute().to(torch.float32))
        self.log('train/tokens/batch_all', (train_batch_tokens_PAD + train_batch_tokens_nonPAD).to(torch.float32))
        self.log('train/tokens/batch_PAD', train_batch_tokens_PAD.to(torch.float32))
        self.log('train/tokens/batch_nonPAD', train_batch_tokens_nonPAD.to(torch.float32))
        self.log('train/tokens/total_all', (self.train_total_tokens_PAD.compute() + self.train_total_tokens_nonPAD.compute()).to(torch.float32))
        self.log('train/tokens/total_PAD', self.train_total_tokens_PAD.compute().to(torch.float32))
        self.log('train/tokens/total_nonPAD', self.train_total_tokens_nonPAD.compute().to(torch.float32))

        return loss

    def validation_step(self, 
                        batch: Dict[str, Float[torch.Tensor, 'B L']], 
                        batch_idx: int) -> torch.Tensor:
        pred_logits: Float[torch.Tensor, 'B L V'] = self.forward(batch)
        loss: torch.Tensor = self.loss(pred_logits, batch['input_ids'])
        ppl: torch.Tensor = torch.exp(loss)

        # Logging
        self.log('val/loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('val/ppl', torch.tensor(min(ppl, 100), dtype=torch.float32), on_epoch=True, sync_dist=True) # artificially cap to 100 so that charts look prettier

        return loss
    
    def parameters(self):
        return list(self.model.parameters()) + list(self.lm_head.parameters())

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
    
    def on_save_checkpoint(self, checkpoint):
        # Save metric's state in the checkpoint
        checkpoint['train_total_examples'] = self.train_total_examples.compute()
        checkpoint['train_total_tokens_PAD'] = self.train_total_tokens_PAD.compute()
        checkpoint['train_total_tokens_nonPAD'] = self.train_total_tokens_nonPAD.compute()

    def on_load_checkpoint(self, checkpoint):
        # Load metric's state from the checkpoint
        if 'train_total_examples' in checkpoint:
            self.train_total_examples = SumMetric()
            self.train_total_examples.update(checkpoint['train_total_examples'])
        if 'train_total_tokens_PAD' in checkpoint:
            self.train_total_tokens_PAD = SumMetric()
            self.train_total_tokens_PAD.update(checkpoint['train_total_tokens_PAD'])
        if 'train_total_tokens_nonPAD' in checkpoint:
            self.train_total_tokens_nonPAD = SumMetric()
            self.train_total_tokens_nonPAD.update(checkpoint['train_total_tokens_nonPAD'])