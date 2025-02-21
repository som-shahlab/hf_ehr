import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from typing import Dict, Any, Optional, Union
from omegaconf import DictConfig
from typing import Dict, Any, Optional
from jaxtyping import Float
from lightning.pytorch.utilities import rank_zero_only
from hf_ehr.models.base import BaseModel

#######################################
## Rotary Positional Embedding (ROPE) ##
#######################################
def rotate_every_two_v2(x):
    """Performs rotary transformation on input tensor"""
    flat_x = x.reshape(-1, x.shape[-1])
    x1 = flat_x[:, ::2]
    x2 = flat_x[:, 1::2]
    result = torch.stack((-x2, x1), axis=-1).reshape(x.shape)
    return result

def fixed_pos_embedding(ages, dim, dtype):
    """Generates sinusoidal positional embeddings based on inv freq of positions"""
    inv_freq = 1.0 / (10000 ** (torch.linspace(0, dim - 1, steps=dim // 2, device=ages.device)))
    inv_freq = inv_freq.reshape(1, 1, dim // 2)
    ages = ages.reshape(ages.shape[0], 1)
    t = inv_freq * ages
    sin, cos = torch.sin(t), torch.cos(t)
    final_shape = (ages.shape[0], 1, dim)
    sin = torch.stack((sin, sin), axis=-1).reshape(final_shape).type(dtype)
    cos = torch.stack((cos, cos), axis=-1).reshape(final_shape).type(dtype)
    return sin, cos

def apply_rotary_pos_emb(x, sincos):
    """Applies rotary positional embedding"""
    sin, cos = sincos
    sin = sin.to(dtype=x.dtype)
    cos = cos.to(dtype=x.dtype)

    if sin.dim() == 3 and x.dim() == 4:
        sin = sin.unsqueeze(1)  # Add extra dimension to sin and cos if necessary
        cos = cos.unsqueeze(1)
    
    if sin.shape[0] != x.shape[0]:
        sin = sin.transpose(0, 1)
        cos = cos.transpose(0, 1)

    sin = sin.expand_as(x)
    cos = cos.expand_as(x)

    result = (x * cos) + (rotate_every_two_v2(x) * sin)
    assert result.shape == x.shape, f"Result shape mismatch: {result.shape} vs {x.shape}"

    return result

# Custom GPT-2 Attention Layer with RoPE
class RoPEGPT2Attention(GPT2Attention):
    """"Extends the default GPT-2 attention mechanism to support RoPE"""
    def __init__(self, config):
        super().__init__(config)
        self.split_size = config.hidden_size

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False):
        # linear transformation to create Q, K, V tensors 
        mixed_x_layer = self.c_attn(hidden_states)

        # Split into Q, K, V tensors
        q, k, v = mixed_x_layer.split(self.split_size, dim=-1)        

        # Generating Rotary Positional Embeddings 
        seq_length = q.shape[-2]
        sincos = fixed_pos_embedding(torch.arange(seq_length, device=q.device).float(), q.shape[-1], q.dtype)

        # Applying Rotary Positionam Embeddings
        q = apply_rotary_pos_emb(q, sincos)
        k = apply_rotary_pos_emb(k, sincos)

        # Attention step
        attn_output, present = self._attn(q, k, v, attention_mask, head_mask)

        # Reshape attn_output to match hidden_states' batch and sequence dimensions
        if attn_output.shape[1] != hidden_states.shape[1]:
            attn_output = attn_output.mean(dim=1)

        # Final linear projection
        attn_output = self.c_proj(attn_output)
        return attn_output, present

class GPTLanguageModel(BaseModel):
    """
    GPT with a Language Model head.
    """

    def __init__(self, config: DictConfig, vocab_size, pad_token_id) -> None:
        super(GPTLanguageModel, self).__init__(config, vocab_size, pad_token_id)

        # Enable flash attention
        if torch.cuda.get_device_capability('cuda')[0] >= 8:
            kwargs = {
                # 'attn_implementation': 'flash_attention_2',
                # 'torch_dtype': torch.bfloat16,
            }
        else:
            kwargs = {}

        # Model specs
        model_config = AutoConfig.from_pretrained(config.model.hf_name if hasattr(config.model, 'hf_name') else 'gpt2', **kwargs)
        model_config.vocab_size = vocab_size
        model_config.n_positions = config.data.dataloader.max_length
        for key, val in config.model.config_kwargs.items():
            assert hasattr(
                model_config, key
            ), f"Config for HF model {model_name} does not have attribute {key}"
            setattr(model_config, key, val)
        self.hidden_size = model_config.n_embd

        # Model
        if getattr(config.model, 'is_keep_pretrained_weights', False):
            # TODO: Implement loading of pretrained weights
            raise NotImplementedError("Loading of pretrained weights is not yet implemented.")
            self.model = AutoModelForCausalLM.from_pretrained(model_config, **kwargs)
        else:
            self.model = AutoModelForCausalLM.from_config(model_config, **kwargs)
            
        # ROPE
        self.is_use_rope = getattr(config.data.dataloader, 'is_use_rope', False) # added to replace default attention layers with custom RoPEGPT2Attention layers
        if self.is_use_rope:
            self._replace_attention_with_rope()

        # Run any post-init handlers from super()
        self.post_init()

    def _replace_attention_with_rope(self):
        """Adds RoPE for all layers"""
        for block in self.model.transformer.h:
            block.attn = RoPEGPT2Attention(self.model.config)
    
    def training_step(self, 
                      batch: Dict[str, Any],
                      batch_idx: int) -> Optional[torch.Tensor]:
        self.batch_idx = batch_idx
        tokens: Dict[str, Float[torch.Tensor, 'B L']] = batch['tokens']
        B: int = tokens['input_ids'].shape[0]

        outputs = self.model(**tokens)
        loss: torch.Tensor = outputs.loss

        # Check if loss is NaN and handle accordingly
        if torch.isnan(loss).any():
            print("NaN detected in loss, skipping this batch.")
            return  # Skip this batch

        # Learning rate scheduler
        if self.trainer:
            lr: float = self.trainer.lr_scheduler_configs[0].scheduler.optimizer.param_groups[0]["lr"]
            sch = self.lr_schedulers()
            sch.step()

        # Logging + Metrics
        self.log_training_step(loss.detach(), B, tokens, lr)

        return loss