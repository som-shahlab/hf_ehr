import torch
from torch import optim
import lightning as L
import torch.nn.functional as F
from omegaconf import DictConfig
from torchmetrics.aggregation import SumMetric
from hf_ehr.utils import lr_warmup_with_constant_plateau
from hf_ehr.data.datasets import FEMRTokenizer
from jaxtyping import Float
from typing import Dict, List, Any, Optional, Union, Tuple

class BaseModel(L.LightningModule):
    """
    Base PyTorchLightning model with some common methods.
    """
    
    model: Any
    hidden_size: int
    model_name: str
    config: DictConfig
    vocab_size: int
    pad_token_id: int

    def __init__(self, config: DictConfig, tokenizer: FEMRTokenizer) -> None:
        super().__init__()
        self.save_hyperparameters('config') #NOTE: Need to exclude `tokenizer` otherwise internal PTL .hparam call later will hang
        self.model_name: str = config.model.name
        self.config = config
        self.vocab_size = tokenizer.vocab_size
        self.pad_token_id = tokenizer.pad_token_id
        
        # Metrics
        self.sum_metrics: Dict[str, SumMetric] = torch.nn.ModuleDict({
            'train_total_examples': SumMetric(),
            'train_total_tokens_PAD': SumMetric(),
            'train_total_tokens_nonPAD': SumMetric(),
        })

    def parameters(self) -> List:
        params = []
        if hasattr(self, 'model'):
            params += list(self.model.parameters())
        if hasattr(self, 'lm_head'):
            params += list(self.lm_head.parameters())
        return params

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

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
        """Save each metric's state in the checkpoint."""
        for key, metric in self.sum_metrics.items():
            checkpoint[key] = metric.compute()

    def on_load_checkpoint(self, checkpoint):
        """Load each metric's state in the checkpoint."""
        for key in self.sum_metrics.keys():
            if key in checkpoint:
                self.sum_metrics[key] = SumMetric()
                # Need to rescale SumMetric loaded from checkpoint since its saved value is summed across all GPUs,
                # but this `on_load_checkpoint()` gets called per GPU. Need to do this quotient/remainder thing in
                # case the SumMetric's valuvalue is not divisible by the # of GPUs
                remainder: int = checkpoint[key] % (self.trainer.num_devices * self.trainer.num_nodes)
                quotient: int = checkpoint[key] // (self.trainer.num_devices * self.trainer.num_nodes)
                self.sum_metrics[key].update(quotient + (remainder if self.trainer.global_rank == 0 else 0))


class CausalModel(BaseModel):
    def __init__(self, config: DictConfig) -> None:
        super(CausalModel, self).__init__(config)

    def forward(self, tokens: Dict[str, Float[torch.Tensor, 'B L']], is_return_hidden_states: bool = True) -> Union[Tuple[Float[torch.Tensor, 'B L V'], Float[torch.Tensor, 'B L H']], Float[torch.Tensor, 'B L V']]:
        B: int = tokens['input_ids'].shape[0]
        L: int = tokens['input_ids'].shape[1]
        H: int = self.hidden_size
        V: int = self.vocab_size
        
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
                               ignore_index=self.pad_token_id, 
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
        V: int = self.vocab_size
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

