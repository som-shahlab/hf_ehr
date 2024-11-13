import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from typing import Dict, Any, Optional, Union
from omegaconf import DictConfig
from typing import Dict, Any, Optional
from jaxtyping import Float
from lightning.pytorch.utilities import rank_zero_only

from hf_ehr.models.modules import BaseModel

# Updated fixed_pos_embedding function
def fixed_pos_embedding(seq_length, head_dim, device, dtype):
    """Generates sinusoidal positional embeddings based on inv freq of positions"""
    position_ids = torch.arange(0, seq_length, dtype=torch.float, device=device)  # [seq_length]
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, dtype=torch.float, device=device) / head_dim))  # [head_dim // 2]
    sinusoid_inp = torch.outer(position_ids, inv_freq)  # [seq_length, head_dim // 2]
    sin = torch.sin(sinusoid_inp).unsqueeze(0).unsqueeze(0).to(dtype)  # [1, 1, seq_length, head_dim // 2]
    cos = torch.cos(sinusoid_inp).unsqueeze(0).unsqueeze(0).to(dtype)  # [1, 1, seq_length, head_dim // 2]
    return cos, sin

# Updated apply_rotary_pos_emb function
def apply_rotary_pos_emb(x, cos, sin):
    """Applies rotary positional embedding to tensor x"""
    # x shape: [batch_size, num_heads, seq_length, head_dim]
    x1 = x[..., ::2]  # [batch_size, num_heads, seq_length, head_dim // 2]
    x2 = x[..., 1::2]  # [batch_size, num_heads, seq_length, head_dim // 2]
    x_rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)  # [batch_size, num_heads, seq_length, head_dim]
    return x_rotated

# Custom GPT-2 Attention Layer with RoPE
class RoPEGPT2Attention(GPT2Attention):
    """Extends the default GPT-2 attention mechanism to support RoPE"""
    def __init__(self, config):
        super().__init__(config)
        self.split_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.split_size // self.num_heads

    def split_heads(self, x):
        """Split the last dimension into (num_heads, head_dim)."""
        # x: [batch_size, seq_length, hidden_size]
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)  # [batch_size, seq_length, num_heads, head_dim]
        x = x.view(*new_shape)  # [batch_size, seq_length, num_heads, head_dim]
        return x.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_length, head_dim]

    def merge_heads(self, x):
        """Merge the heads and last two dimensions."""
        # x: [batch_size, num_heads, seq_length, head_dim]
        x = x.permute(0, 2, 1, 3).contiguous()  # [batch_size, seq_length, num_heads, head_dim]
        new_shape = x.size()[:-2] + (self.num_heads * self.head_dim,)  # [batch_size, seq_length, hidden_size]
        return x.view(*new_shape)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        # Linear transformation to create QKV tensors
        mixed_x_layer = self.c_attn(hidden_states)  # [batch_size, seq_length, 3 * hidden_size]

        # Split into Q, K, V tensors
        query, key, value = mixed_x_layer.split(self.split_size, dim=2)  # Each is [batch_size, seq_length, hidden_size]

        # Split heads
        query = self.split_heads(query)  # [batch_size, num_heads, seq_length, head_dim]
        key = self.split_heads(key)
        value = self.split_heads(value)

        # Generate Rotary Positional Embeddings
        seq_length = query.size(2)
        cos, sin = fixed_pos_embedding(seq_length, self.head_dim, query.device, query.dtype)

        # Apply RoPE to query and key
        query = apply_rotary_pos_emb(query, cos, sin)
        key = apply_rotary_pos_emb(key, cos, sin)

        # Adjust attention mask shape if necessary
        if attention_mask is not None and attention_mask.dim() == 2:
            # attention_mask: [batch_size, seq_length] -> [batch_size, 1, 1, seq_length]
            attention_mask = attention_mask[:, None, None, :]

        # Compute attention
        attn_outputs = self._attn(query, key, value, attention_mask, head_mask)
        attn_output = attn_outputs[0]  # [batch_size, num_heads, seq_length, head_dim]
        present = attn_outputs[1]

        # Merge heads
        attn_output = self.merge_heads(attn_output)  # [batch_size, seq_length, hidden_size]

        # Final linear projection
        attn_output = self.c_proj(attn_output)
        return attn_output, present

class GPTLanguageModel(BaseModel):
    """
    GPT with a Language Model head.
    """

    def __init__(self, config: DictConfig, vocab_size, pad_token_id) -> None:
        super(GPTLanguageModel, self).__init__(config, vocab_size, pad_token_id)

        # Model specs
        model_name = config.model.hf_name if hasattr(config.model, 'hf_name') else 'gpt2'
        model_config = AutoConfig.from_pretrained(model_name)
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
            self.model = AutoModelForCausalLM.from_pretrained(model_name, config=model_config)
        else:
            self.model = AutoModelForCausalLM.from_config(model_config)

        self.is_use_rope = getattr(
            config.data.dataloader, 'is_use_rope', False
        )  # Replace attention layers with custom RoPEGPT2Attention layers

        if self.is_use_rope:
            self._replace_attention_with_rope()

        # Run any post-init handlers from super()
        self.post_init()

    def _replace_attention_with_rope(self):
        """Replaces attention layers with RoPEGPT2Attention"""
        for block in self.model.transformer.h:
            block.attn = RoPEGPT2Attention(self.model.config)

    def training_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Optional[torch.Tensor]:
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
