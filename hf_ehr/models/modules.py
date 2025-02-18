import torch
from torch import optim
import lightning as L
import torch.distributed as dist
from tqdm import tqdm
from omegaconf import DictConfig
from torchmetrics.aggregation import SumMetric, CatMetric
from jaxtyping import Float
from typing import Dict, List, Any, Optional, Union
import wandb
from lightning.pytorch.utilities import rank_zero_only
from hf_ehr.utils import lr_warmup_with_constant_plateau
from loguru import logger

def calculate_flops_per_token(model, vocab_size: int) -> int:
    """Returns FLOPs per token for model."""
    # TODO - remove; needs to be done on input size level
    return 0

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

    def __init__(self, config: DictConfig, vocab_size, pad_token_id) -> None:
        super().__init__()
        self.save_hyperparameters('config') #NOTE: Need to exclude `tokenizer` otherwise internal PTL .hparam call later will hang
        self.model_name: str = config.model.name
        self.config = config
        self.vocab_size: int = vocab_size
        self.pad_token_id: int = pad_token_id
        self.flops_per_token = None
        
        # Metrics
        self.sum_metrics: Dict[str, SumMetric] = torch.nn.ModuleDict({
            'train_total_examples': SumMetric(),
            'train_total_tokens_PAD': SumMetric(),
            'train_total_tokens_nonPAD': SumMetric(),
        })
        self.cat_metrics: Dict[str, CatMetric] = torch.nn.ModuleDict({
            'val_batch_loss': CatMetric(),
            'val_batch_tokens_nonPAD': CatMetric(),
        })
    
    def post_init(self):
        """Post-initialization method to be called by subclass."""
        self.flops_per_token = 0
        # Track batch_idx
        self.batch_idx: int = 0

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
        for key, metric in self.cat_metrics.items():
            checkpoint[key] = metric.compute()
        checkpoint['batch_idx'] = self.batch_idx if hasattr(self, 'batch_idx') else 0
        checkpoint['wandb_run_id'] = wandb.run.id if wandb and wandb.run else None

    def on_load_checkpoint(self, checkpoint):
        """Restore each metric's state from the checkpoint."""
        super().on_load_checkpoint(checkpoint)
        # Sum Metrics
        for key, metric in self.sum_metrics.items():
            self.sum_metrics[key].update(checkpoint[key])
            logger.info(f"Loaded metric `{key}` from checkpoint with value: `{self.sum_metrics[key].cuda().compute()}`")
        # Cat Metrics
        for key, metric in self.cat_metrics.items():
            self.cat_metrics[key].update(checkpoint[key])
            logger.info(f"Loaded metric `{key}` from checkpoint with value: `{self.cat_metrics[key].cuda().compute()}`")
        self.batch_idx: int = checkpoint.get('batch_idx', 0)
        logger.info(f"Loaded `batch_idx` from checkpoint with value: `{self.batch_idx}`")

    def validation_step(self, 
                        batch: Dict[str, Any],
                        batch_idx: int) -> Optional[torch.Tensor]:
        tokens: Dict[str, Float[torch.Tensor, 'B L']] = batch['tokens']
        B: int = tokens['input_ids'].shape[0]
        
        if 'llama' in self.model_name:
            tokens.pop("token_type_ids", None)
        
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
        self.log_validation_step(loss, tokens) # ! NOTE: I'm assuming this loss is averaged over all non-PAD tokens for this function call

        return loss

    def on_train_epoch_end(self):
        # Needed for ApproxBatchSampler to reset random seed after every epoch
        if self.config.data.dataloader.mode == 'approx':
            self.trainer.train_dataloader.batch_sampler.set_epoch(self.current_epoch + 1)

    def on_train_start(self):
        if rank_zero_only.rank == 0 and wandb and wandb.run:
            wandb.run.summary["tokenizer_vocab_size"] = self.vocab_size
            wandb.run.summary["tokenizer_pad_token_id"] = self.pad_token_id
            wandb.run.summary["model_parameter_count"] = self.get_param_count()

        ############################
        # Start of OOM detection
        # Create a fake full batch and pass through model for early detection of OOM
        if self.config.data.dataloader.mode == 'approx':
            fake_batch = {
                'input_ids' : torch.ones((self.config.data.dataloader.approx_batch_sampler.max_tokens // self.config.data.dataloader.max_length, self.config.data.dataloader.max_length)).long().to(self.device),
                'attention_mask' : torch.ones((self.config.data.dataloader.approx_batch_sampler.max_tokens // self.config.data.dataloader.max_length, self.config.data.dataloader.max_length)).to(self.device),
                'labels' : torch.ones((self.config.data.dataloader.approx_batch_sampler.max_tokens // self.config.data.dataloader.max_length, self.config.data.dataloader.max_length)).long().to(self.device),
            }
        elif self.config.data.dataloader.mode == 'batch':
            fake_batch = {
                'input_ids' : torch.ones((self.config.data.dataloader.batch_size, self.config.data.dataloader.max_length)).long().to(self.device),
                'attention_mask' : torch.ones((self.config.data.dataloader.batch_size,self.config.data.dataloader.max_length)).to(self.device),
                'labels' : torch.ones((self.config.data.dataloader.batch_size,self.config.data.dataloader.max_length)).long().to(self.device),
            }
        else:
            raise ValueError(f"Unsupported config.data.dataloader.mode: `{self.config.data.dataloader.mode}`")
        assert fake_batch['input_ids'].numel() <= self.config.data.dataloader.approx_batch_sampler.max_tokens, f"Fake batch size is larger than max_tokens: {fake_batch['input_ids'].numel()} > {self.config.data.dataloader.approx_batch_sampler.max_tokens}"
        if 'hyena' in self.model_name:
            # Hyena doesn't support `attention_mask` in the input, so remove it
            fake_batch.pop('attention_mask')
        outputs = self.model(**fake_batch)
        del outputs
        del fake_batch
        # End of OOM detection
        ############################

        if self.trainer.global_step > 0:
            # If we're loading from a checkpoint, then we need to adjust ApproxBatchSampler
            # so that the next batch we sample isa ctually the next batch that this checkpoint
            # would have sampled (and not restart at batch_idx=0 each time)
            if self.config.data.dataloader.mode == 'approx':
                self.trainer.train_dataloader.batch_sampler.sampler.set_epoch(self.trainer.current_epoch)
                self.trainer.train_dataloader.batch_sampler.start_batch_idx = self.batch_idx if self.batch_idx > 0 else self.trainer.global_step
                logger.success(f"We are resuming from a checkpoint that used `ApproxBatchSampler`, so set: `epoch={self.trainer.current_epoch}` and `start_batch_idx={self.trainer.train_dataloader.batch_sampler.start_batch_idx}`")
        torch.distributed.barrier()

    def on_validation_start(self):
        # When we restart validation, reset # of tokens that have gone into the val loss calculation to 0
        self.cat_metrics['val_batch_loss'].reset()
        self.cat_metrics['val_batch_tokens_nonPAD'].reset()

    def on_validation_epoch_end(self):
        # Calculate per-token val loss and perplexity
        val_batch_loss: torch.Tensor = self.cat_metrics['val_batch_loss'].compute()  # Loss/token across all validation batches
        val_batch_tokens_nonPAD: torch.Tensor = self.cat_metrics['val_batch_tokens_nonPAD'].compute()  # Tokens/batch across all validation batches

        # Calculate the weighted average loss per token
        total_loss = torch.sum(val_batch_loss * val_batch_tokens_nonPAD)  # Sum of weighted losses
        total_tokens = torch.sum(val_batch_tokens_nonPAD)  # Sum of all non-PAD tokens
        loss = total_loss / total_tokens  # Average loss per token

        # Calculate perplexity from the average loss
        ppl: torch.Tensor = torch.exp(loss)

        # Log the metrics
        self.log('val/loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/ppl', torch.clamp(ppl, max=100).to(torch.float32), on_step=False, on_epoch=True, sync_dist=True)

        # Log training metrics to sync val/loss to training metrics
        self.log('val/tokens/total_all', (self.sum_metrics['train_total_tokens_PAD'].compute() + self.sum_metrics['train_total_tokens_nonPAD'].compute()).to(torch.float32), on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/tokens/total_PAD', self.sum_metrics['train_total_tokens_PAD'].compute().to(torch.float32), on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/tokens/total_nonPAD', self.sum_metrics['train_total_tokens_nonPAD'].compute().to(torch.float32), on_step=False, on_epoch=True, sync_dist=True)

    def log_validation_step(self, loss: torch.Tensor, tokens: Dict[str, Any]):
        # NOTE: Assumes `loss` has been scaled per-token already
        val_batch_tokens_nonPAD: int = (tokens['input_ids'] != self.pad_token_id).sum().detach().cpu().item()
        self.cat_metrics['val_batch_loss'].update(loss)  # Corrected to use cat_metrics
        self.cat_metrics['val_batch_tokens_nonPAD'].update(val_batch_tokens_nonPAD)

    def log_training_step(self, loss: torch.Tensor, B: int, tokens: Dict[str, Any], lr: float):
        """
            B: batch size
        """
        loss = loss.detach()
        ppl: torch.Tensor = torch.exp(loss)

        # Metrics
        train_batch_examples: int = B
        self.sum_metrics['train_total_examples'].update(train_batch_examples)
        self.log('optim/lr', lr)
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/ppl', torch.clamp(ppl, max=100).to(torch.float32))  # artificially cap to 100 so that charts look prettier
        self.log('train/examples/batch', torch.tensor(B, dtype=torch.float32))
        self.log('train/examples/total', self.sum_metrics['train_total_examples'].compute().to(torch.float32), sync_dist=True)

        if 'hyena' in self.model_name:
            # Hyena doesn't have `attention_mask` in the input, so manually calculate number of PAD tokens
            pad_token_id = self.pad_token_id
            input_ids = tokens['input_ids']
            train_batch_tokens_PAD = (input_ids == pad_token_id).sum()
            train_batch_tokens_nonPAD = (input_ids != pad_token_id).sum()
        else:
            train_batch_tokens_PAD: torch.Tensor = (1 - tokens['attention_mask']).sum()
            train_batch_tokens_nonPAD: torch.Tensor = tokens['attention_mask'].sum()

        # Update cumulative metrics
        self.sum_metrics['train_total_tokens_PAD'].update(train_batch_tokens_PAD)
        self.sum_metrics['train_total_tokens_nonPAD'].update(train_batch_tokens_nonPAD)
        self.log('train/tokens/batch_all', (train_batch_tokens_PAD + train_batch_tokens_nonPAD).to(torch.float32))
        self.log('train/tokens/batch_PAD', train_batch_tokens_PAD.to(torch.float32))
        self.log('train/tokens/batch_nonPAD', train_batch_tokens_nonPAD.to(torch.float32))
        self.log('train/tokens/total_all', (self.sum_metrics['train_total_tokens_PAD'].compute() + self.sum_metrics['train_total_tokens_nonPAD'].compute()).to(torch.float32))
        self.log('train/tokens/total_PAD', self.sum_metrics['train_total_tokens_PAD'].compute().to(torch.float32))
        self.log('train/tokens/total_nonPAD', self.sum_metrics['train_total_tokens_nonPAD'].compute().to(torch.float32))