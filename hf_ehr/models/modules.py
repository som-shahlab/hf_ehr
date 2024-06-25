import torch
from torch import optim
import lightning as L
import torch.distributed as dist

from omegaconf import DictConfig
from torchmetrics.aggregation import SumMetric
from jaxtyping import Float
from typing import Dict, List, Any, Optional, Union
from calflops import calculate_flops

from hf_ehr.utils import lr_warmup_with_constant_plateau
from hf_ehr.data.datasets import FEMRTokenizer, DescTokenizer

def calculate_flops_per_token(model, vocab_size: int) -> float:
    """Returns GFLOPs per token for model."""
    was_training: bool = model.training
    model.eval()  # Ensure model is in evaluation mode
    # inputs that match the shape and type of expected inputs
    dummy_inputs = {
        "input_ids": torch.randint(0, vocab_size, (1,1)).to(model.device),
        "labels": torch.randint(0, vocab_size, (1,1)).to(model.device)
    }
    flops, macs, params = calculate_flops(model=model, kwargs=dummy_inputs, output_as_string=False, output_precision=4)

    if was_training:
        model.train()
    return flops

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
    flops_per_token: Optional[int] = None

    def __init__(self, config: DictConfig, tokenizer: Union[FEMRTokenizer, DescTokenizer]) -> None:
        super().__init__()
        self.save_hyperparameters('config') #NOTE: Need to exclude `tokenizer` otherwise internal PTL .hparam call later will hang
        self.model_name: str = config.model.name
        self.config = config
        self.vocab_size: int = tokenizer.vocab_size
        self.pad_token_id: int = tokenizer.pad_token_id
        
        # Metrics
        self.sum_metrics: Dict[str, SumMetric] = torch.nn.ModuleDict({
            'train_total_examples': SumMetric(),
            'train_total_tokens_PAD': SumMetric(),
            'train_total_tokens_nonPAD': SumMetric(),
        })

    def calculate_flops_per_token(self) -> float:
        return calculate_flops_per_token(self.model, self.vocab_size)
    
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
                # case the SumMetric's value is not divisible by the # of GPUs
                remainder: int = checkpoint[key] % (self.trainer.num_devices * self.trainer.num_nodes)
                quotient: int = checkpoint[key] // (self.trainer.num_devices * self.trainer.num_nodes)
                self.sum_metrics[key].update(quotient + (remainder if self.trainer.global_rank == 0 else 0))

    def validation_step(self, 
                        batch: Dict[str, Any],
                        batch_idx: int) -> Optional[torch.Tensor]:
        tokens: Dict[str, Float[torch.Tensor, 'B L']] = batch['tokens']
        B: int = tokens['input_ids'].shape[0]
        
        # Forward pass
        outputs = self.model(**tokens)
        loss: torch.Tensor = outputs.loss
        
        if torch.isnan(loss).any():
            nan_detected = torch.tensor([1.0], device=self.device)
        else:
            nan_detected = torch.tensor([0.0], device=self.device)

        dist.all_reduce(nan_detected, op=dist.ReduceOp.MAX)
        if nan_detected.item() == 1:
            print("NaN detected in loss, skipping this batch across all processes.")
            return  # Skip this batch on all processes

        # Logging
        self.log_validation_step(loss)

        return loss

    def on_train_epoch_end(self):
        # Needed for ApproxBatchSampler to reset random seed after every epoch
        self.trainer.train_dataloader.batch_sampler.sampler.set_epoch(self.current_epoch + 1)
    
    def on_train_start(self):
        if self.flops_per_token is None:
            print(f"Start | Calculating FLOPs per token for model {self.model_name}...")
            self.flops_per_token = self.calculate_flops_per_token()
            print(f"End | Calculating FLOPs per token for model {self.model_name} | FLOPs/token={self.flops_per_token}")
        self.log("flops_per_token", self.flops_per_token)
    
    def log_validation_step(self, loss: torch.Tensor):
        ppl: torch.Tensor = torch.exp(loss)

        self.log('val/loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('val/ppl', torch.clamp(ppl, max=100).to(torch.float32), on_epoch=True, sync_dist=True) # artificially cap to 100 so that charts look prettier

    def log_training_step(self, loss: torch.Tensor, B: int, tokens: Dict[str, Any], lr: float):
        """
            B: batch size
        """
        loss = loss.detach()
        ppl: torch.Tensor = torch.exp(loss)

        # Metrics
        train_batch_examples: int = B
        if 'hyena' in self.model_name:
            self.sum_metrics['train_total_examples'].update(train_batch_examples)
            self.log('optim/lr', lr)
            self.log('train/loss', loss, prog_bar=True)
            self.log('train/ppl', torch.clamp(ppl, max=100).to(torch.float32)) # artificially cap to 100 so that charts look prettier
            self.log('train/examples/batch', torch.tensor(B, dtype=torch.float32))
            self.log('train/examples/total', self.sum_metrics['train_total_examples'].compute().to(torch.float32))
        else:
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
            if self.flops_per_token is not None:
                self.log('train/total_flops', self.sum_metrics['train_total_tokens_nonPAD'].compute().to(torch.float32) * self.flops_per_token)
                    
            
            
