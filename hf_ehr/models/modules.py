import torch
from torch import optim
import lightning as L
import torch.distributed as dist
from tqdm import tqdm
from omegaconf import DictConfig
from torchmetrics.aggregation import SumMetric
from jaxtyping import Float
from typing import Dict, List, Any, Optional, Union
from calflops import calculate_flops
import wandb
from lightning.pytorch.utilities import rank_zero_only
from hf_ehr.utils import lr_warmup_with_constant_plateau

def calculate_flops_per_token(model, vocab_size: int) -> int:
    """Returns FLOPs per token for model."""
    was_training: bool = model.training
    model.eval()  # Ensure model is in evaluation mode
    # inputs that match the shape and type of expected inputs
    dummy_inputs = {
        "input_ids": torch.randint(0, vocab_size, (1,1)).to(model.device),
        "labels": torch.randint(0, vocab_size, (1,1)).to(model.device)
    }
    flops, macs, params = calculate_flops(model=model, kwargs=dummy_inputs, output_as_string=False, print_results=False, output_precision=4)

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
    
    def post_init(self):
        """Post-initialization method to be called by subclass."""
        # Calculate FLOPs per token
        print("Start | Calculating FLOPs per token...")
        self.flops_per_token = calculate_flops_per_token(self.model, self.vocab_size)
        print("End | Calculating FLOPs per token...")

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
        """Restore each metric's state from the checkpoint."""
        super().on_load_checkpoint(checkpoint)
        for key, metric in self.sum_metrics.items():
            self.sum_metrics[key].update(checkpoint[key])
            print(f"Loaded metric `{key}` from chekpoint with value: `{self.sum_metrics[key].cuda().compute()}`")

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
        if rank_zero_only.rank == 0 and wandb and wandb.run:
            wandb.run.summary["flops_per_token"] = self.flops_per_token
            wandb.run.summary["tokenizer_vocab_size"] = self.vocab_size
            wandb.run.summary["tokenizer_pad_token_id"] = self.pad_token_id
            wandb.run.summary["model_parameter_count"] = self.get_param_count()
        
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
        outputs = self.model(**fake_batch)
        del outputs
        del fake_batch
        
        if self.trainer.global_step > 0:
            # Make ApproxBatchSampler deterministic by looping through dataset until we hit
            # the current step; otherwise, when Lightning restarts from a checkpoint it will
            # reset np.random.seed(0) in ApproxBatchSampler. We need to "turn the crank" on this PRNG
            # by repeatedly calling __iter__() until we hit our current step in order to recreate the
            # last actual state of the PRNG corresponding to this checkpoint
            if self.config.data.dataloader.mode == 'approx':
                self.trainer.train_dataloader.batch_sampler.sampler.set_epoch(self.trainer.current_epoch)
                # TODO - just change `current_iter`
                # self.trainer.train_dataloader.batch_sampler.sampler.current_iter = self.trainer.global_step
                print(f"We are resuming from a checkpoint that used `ApproxBatchSampler`, so iterate through training dataloader until it matches the checkpoint's current step")
                self.trainer.train_dataloader.batch_sampler.sampler.set_epoch(self.trainer.current_epoch)
                for idx, __ in tqdm(enumerate(self.trainer.train_dataloader), total=self.trainer.global_step, desc='Iterating thru train DataLoader to align `ApproxBatchSampler` with ckpt\'s current step...'):
                    if idx >= self.trainer.global_step - 1:
                        break

    def log_validation_step(self, loss: torch.Tensor):
        ppl: torch.Tensor = torch.exp(loss)
        self.log('val/loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('val/ppl', torch.clamp(ppl, max=100).to(torch.float32), on_epoch=True, sync_dist=True) # artificially cap to 100 so that charts look prettier
        self.log('val/tokens/total_all', (self.sum_metrics['train_total_tokens_PAD'].compute() + self.sum_metrics['train_total_tokens_nonPAD'].compute()).to(torch.float32))
        self.log('val/tokens/total_PAD', self.sum_metrics['train_total_tokens_PAD'].compute().to(torch.float32))
        self.log('val/tokens/total_nonPAD', self.sum_metrics['train_total_tokens_nonPAD'].compute().to(torch.float32))
        if self.flops_per_token is not None:
            self.log('val/total_flops', self.sum_metrics['train_total_tokens_nonPAD'].compute().to(torch.float32) * self.flops_per_token)

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
        self.log('train/examples/total', self.sum_metrics['train_total_examples'].compute().to(torch.float32))

        if 'hyena' in self.model_name:
            pad_token_id = self.pad_token_id
            input_ids = tokens['input_ids']
            train_batch_tokens_PAD = (input_ids == pad_token_id).sum()
            train_batch_tokens_nonPAD = (input_ids != pad_token_id).sum()
        else:
            train_batch_tokens_PAD: torch.Tensor = (1 - tokens['attention_mask']).sum()
            train_batch_tokens_nonPAD: torch.Tensor = tokens['attention_mask'].sum()

        # Update cumulative metrics for both models
        self.sum_metrics['train_total_tokens_PAD'].update(train_batch_tokens_PAD)
        self.sum_metrics['train_total_tokens_nonPAD'].update(train_batch_tokens_nonPAD)

        self.log('train/tokens/batch_all', (train_batch_tokens_PAD + train_batch_tokens_nonPAD).to(torch.float32))
        self.log('train/tokens/batch_PAD', train_batch_tokens_PAD.to(torch.float32))
        self.log('train/tokens/batch_nonPAD', train_batch_tokens_nonPAD.to(torch.float32))
        self.log('train/tokens/total_all', (self.sum_metrics['train_total_tokens_PAD'].compute() + self.sum_metrics['train_total_tokens_nonPAD'].compute()).to(torch.float32))
        self.log('train/tokens/total_PAD', self.sum_metrics['train_total_tokens_PAD'].compute().to(torch.float32))
        self.log('train/tokens/total_nonPAD', self.sum_metrics['train_total_tokens_nonPAD'].compute().to(torch.float32))
        
        if self.flops_per_token is not None:
            self.log('train/total_flops', self.sum_metrics['train_total_tokens_nonPAD'].compute().to(torch.float32) * self.flops_per_token)
