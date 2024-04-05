import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
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
        super(BERTLanguageModel, self).__init__(config)

        # Tokenizer
        self.tokenizer = tokenizer
        mask_token: str = "[MASK]"
        if mask_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_token('mask', mask_token)
            assert hasattr(self.tokenizer, 'mask_token_id'), f"Error - couldn't add [MASK] token to tokenizer"

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
        self.model = AutoModel.from_config(model_config, add_pooling_layer=False) # no pooling on [CLS] b/c doing MLM on each token, and [CLS] will just get ignored
        self.lm_head = nn.Linear(self.hidden_size, tokenizer.vocab_size, bias=False)

    def forward(self, tokens: Dict[str, Float[torch.Tensor, 'B L']], is_return_hidden_states: bool = True) -> Union[Tuple[Float[torch.Tensor, 'B L V'], Float[torch.Tensor, 'B L H']], Float[torch.Tensor, 'B L V']]:
        B: int = tokens['input_ids'].shape[0]
        L: int = tokens['input_ids'].shape[1]
        H: int = self.hidden_size
        V: int = self.tokenizer.vocab_size
        
        hidden_states: Float[torch.Tensor, 'B L H'] = self.model(**tokens).last_hidden_state
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
        
        # Calculate loss
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), 
                               targets.view(-1), 
                               ignore_index=self.tokenizer.pad_token_id, 
                               reduction='mean')
    
        # Sanity checks
        assert logits.shape == (B, L, V)
        assert targets.shape == (B, L)
        """
        OPTIMIZATION opportunity  - remove the addition shape checks for logits and targets
        """
        return loss
    
    def run_eval(self, tokens: Dict[str, Float[torch.Tensor, 'B L']]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B: int = tokens['input_ids'].shape[0]
        L: int = tokens['input_ids'].shape[1]
        V: int = self.tokenizer.vocab_size

        # Save targets before modifying tokens with masking
        pred_targets: Float[torch.Tensor, 'B L'] = tokens['input_ids'].clone()
        device = tokens['input_ids'].device
        # Dynamic mask generation for MLM using in-place operations where possible
        mask: Float[torch.Tensor, 'B L'] = (torch.rand((B, L), device=device, dtype=torch.float32) < self.config.trainer.mlm_mask_pct)
        mask = mask.to(torch.bool)  # Use torch.bool for masks to save memory and computation
        # Efficiently compute indices for masking without moving tensors between devices
        ones_indices = torch.nonzero(mask, as_tuple=True)
        # Directly apply [MASK] and random tokens without shuffling to reduce operations
        n_RANDOM = int(0.1 * mask.sum().item())
        # Direct application of masks and random tokens without unnecessary tensor manipulation
        tokens['input_ids'].masked_fill_(mask, self.tokenizer.mask_token_id)
        # For random replacement, use a more direct method to avoid overhead
        random_indices = torch.randperm(mask.sum().item(), device=device)[:n_RANDOM]
        non_special_tokens = self.tokenizer.get_vocab_tokens(is_include_special_tokens=False).to(device)
        random_tokens = non_special_tokens[torch.randint(0, len(non_special_tokens), (n_RANDOM,), device=device)]
        tokens['input_ids'][ones_indices[0][random_indices], ones_indices[1][random_indices]] = random_tokens
        #Forward pass
        pred_logits = self.forward(tokens, is_return_hidden_states=False)
        pred_targets[~mask] = self.tokenizer.pad_token_id
        loss: torch.Tensor = self.loss(pred_logits, pred_targets)
        # Sanity checks
        assert pred_logits.shape == (B, L, V)
        
        return loss, mask, pred_targets
        
    
    def return_safe_training_step_none_for_ddp(self, loss: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """This allows us to return `None` in training_step() - useful to avoid nan losses"""
        # See `https://github.com/a-r-j/ProteinWorkshop/pull/81/files` for code
        # See discussion `https://github.com/Lightning-AI/pytorch-lightning/issues/5243` for why this is necessary for returning None with DDP
        if loss is None:
            flag_skip = torch.ones((), device=self.device, dtype=torch.bool)
        else:
            flag_skip = torch.zeros((), device=loss.device, dtype=torch.bool)
        print("Flagging skip", flag_skip)

        if torch_dist.is_initialized():
            # if any rank skips a batch, then all other ranks need to skip
            # their batches as well so DDP can properly keep all ranks synced
            world_size = torch_dist.get_world_size()
            print("world_size", world_size)
            torch_dist.barrier()
            print("barrier passed")
            result = [torch.zeros_like(flag_skip) for _ in range(world_size)]
            print("result", result)
            torch_dist.all_gather(result, flag_skip)
            print("post-gather result", result)
            any_skipped = torch.sum(torch.stack(result)).bool().item()
            print("any_skipped", any_skipped)
            if any_skipped:
                for p in self.trainer.model.parameters():
                    if p.grad is not None:
                        del p.grad
                return None
        return loss

    def training_step(self, 
                      batch: Dict[str, Any],
                      batch_idx: int) -> Optional[torch.Tensor]:
        tokens: Dict[str, Float[torch.Tensor, 'B L']] = batch['tokens']
        B: int = tokens['input_ids'].shape[0]

        # Forward pass
        loss, mask, pred_targets = self.run_eval(tokens)
        ppl: torch.Tensor = torch.exp(loss).detach()

        # None of the tokens were randomly chosen to be MASK'd, so loss will be `nan`, so skip
        if pred_targets.sum() == 0:
            print("NONE detected from pred_targets.sum() == 0 in training_step()")
            return self.return_safe_training_step_none_for_ddp(None)
        
        # Throw out bad batches
        if ppl > 100 and self.trainer.global_step > self.config.trainer.scheduler.num_warmup_steps / len(self.config.trainer.devices):
            print("NONE detected from ppl > 100 in training_step()")
            return self.return_safe_training_step_none_for_ddp(None)

        # Learning rate scheduler
        lr: float = self.trainer.lr_scheduler_configs[0].scheduler.optimizer.param_groups[0]["lr"]
        sch = self.lr_schedulers()
        sch.step()
        
        # Metrics
        train_batch_examples: int = B
        train_batch_tokens_MASK: torch.Tensor = mask.sum()
        train_batch_tokens_PAD: torch.Tensor = (1 - tokens['attention_mask']).sum()
        train_batch_tokens_nonPAD: torch.Tensor = tokens['attention_mask'].sum()
        self.sum_metrics['train_total_examples'].update(train_batch_examples)
        self.sum_metrics['train_total_tokens_MASK'].update(train_batch_tokens_MASK)
        self.sum_metrics['train_total_tokens_PAD'].update(train_batch_tokens_PAD)
        self.sum_metrics['train_total_tokens_nonPAD'].update(train_batch_tokens_nonPAD)

        # Logging
        #self.log('optim/lr', lr)
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/ppl', torch.clamp(ppl, max=100).to(torch.float32)) # artificially cap to 100 so that charts look prettier
        self.log('train/examples/batch', torch.tensor(B, dtype=torch.float32))
        self.log('train/examples/total', self.sum_metrics['train_total_examples'].compute().to(torch.float32))
        self.log('train/tokens/batch_all', (train_batch_tokens_PAD + train_batch_tokens_nonPAD).to(torch.float32))
        #self.log('train/tokens/batch_MASK', train_batch_tokens_MASK.to(torch.float32))
        #self.log('train/tokens/batch_PAD', train_batch_tokens_PAD.to(torch.float32))
        #self.log('train/tokens/batch_nonPAD', train_batch_tokens_nonPAD.to(torch.float32))
        self.log('train/tokens/total_all', (self.sum_metrics['train_total_tokens_PAD'].compute() + self.sum_metrics['train_total_tokens_nonPAD'].compute()).to(torch.float32))
        #self.log('train/tokens/total_PAD', self.sum_metrics['train_total_tokens_PAD'].compute().to(torch.float32))
        self.log('train/tokens/total_MASK', self.sum_metrics['train_total_tokens_MASK'].compute().to(torch.float32))
        #self.log('train/tokens/total_nonPAD', self.sum_metrics['train_total_tokens_nonPAD'].compute().to(torch.float32))

        return self.return_safe_training_step_none_for_ddp(loss)

    def validation_step(self, 
                        batch: Dict[str, Any],
                        batch_idx: int) -> Optional[torch.Tensor]:
        tokens: Dict[str, Float[torch.Tensor, 'B L']] = batch['tokens']
        
        # Forward pass
        loss, mask, pred_targets = self.run_eval(tokens)
        ppl: torch.Tensor = torch.exp(loss).detach()
        if pred_targets.sum() == 0:
            # None of the tokens were randomly chosen to be MASK'd, so loss will be `nan`, so skip
            return self.return_safe_training_step_none_for_ddp(None)

        # Logging
        self.log('val/loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('val/ppl', torch.clamp(ppl, max=100).to(torch.float32), on_epoch=True, sync_dist=True) # artificially cap to 100 so that charts look prettier
        return self.return_safe_training_step_none_for_ddp(loss)