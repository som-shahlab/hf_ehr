import torch
from transformers import AutoModelForMaskedLM, AutoConfig
from jaxtyping import Float
from typing import Dict, Any, Optional, Union
from torch import nn
from omegaconf import DictConfig
from transformers.models.bert.modeling_bert import BertSelfAttention
from hf_ehr.models.base import BaseModel

# Custom Bert Self Attention Layer with RoPE
class RoPEBertSelfAttention(BertSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        print("Initialized RoPEBertSelfAttention with RoPE enabled.")

    def apply_rope(self, q, k):
        seq_len = q.shape[2]
        dim = q.shape[-1]
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))

        sinusoid_inp = torch.einsum("i,d->id", torch.arange(seq_len).float(), inv_freq)
        sin = torch.sin(sinusoid_inp)
        cos = torch.cos(sinusoid_inp)
        sin, cos = sin.to(q.device), cos.to(q.device)

        print(f"RoPE is being applied: Sequence Length = {seq_len}, Dim = {dim}")
        q_sin_cos = torch.cat([q[..., ::2] * cos - q[..., 1::2] * sin,
                               q[..., ::2] * sin + q[..., 1::2] * cos], dim=-1)
        k_sin_cos = torch.cat([k[..., ::2] * cos - k[..., 1::2] * sin,
                               k[..., ::2] * sin + k[..., 1::2] * cos], dim=-1)

        print(f"q shape after RoPE: {q_sin_cos.shape}, k shape after RoPE: {k_sin_cos.shape}")
        return q_sin_cos, k_sin_cos

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False):
        mixed_query_layer = self.query(hidden_states)
        if self.is_decoder:
            mixed_key_layer = self.key(hidden_states) if encoder_hidden_states is None else self.key(encoder_hidden_states)
            mixed_value_layer = self.value(hidden_states) if encoder_hidden_states is None else self.value(encoder_hidden_states)
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        query_layer, key_layer = self.apply_rope(query_layer, key_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.attention_head_size, dtype=torch.float32))

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

class BERTLanguageModel(BaseModel):
    """
    BERT with a Language Model head.
    """

    def __init__(self, config: DictConfig, vocab_size, pad_token_id) -> None:
        super(BERTLanguageModel, self).__init__(config, vocab_size, pad_token_id)

        # Model specs
        model_config = AutoConfig.from_pretrained(config.model.hf_name if hasattr(config.model, 'hf_name') else 'bert-base-uncased')
        model_config.vocab_size = vocab_size
        model_config.n_positions = config.data.dataloader.max_length
        for key, val in config.model.config_kwargs.items():
            assert hasattr(model_config, key), f"Config for HF model {config.model.hf_name if hasattr(config.model, 'hf_name') else ''} does not have attribute {key}"
            setattr(model_config, key, val)
        self.model_config = model_config
        self.hidden_size = model_config.n_embd if hasattr(model_config, 'n_embd') else model_config.hidden_size

        # Model
        self.model = AutoModelForMaskedLM.from_config(model_config)
        self.is_use_rope = getattr(config.data.dataloader, 'is_use_rope', False) # added to replace default attention layers with custom RoPEGPT2Attention layers
        
        if self.is_use_rope:
            self._replace_attention_with_rope()

        # Run any post-init handlers from super()
        self.post_init()
    
    def _replace_attention_with_rope(self):
        # Iterate over each encoder layer and replace its self-attention layer with RoPE-enhanced version
        for layer in self.model.bert.encoder.layer:
            layer.attention.self = RoPEBertSelfAttention(self.model.config)

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
        
        # Logging + Metrics
        self.log_training_step(loss.detach(), B, tokens, lr)

        return loss
