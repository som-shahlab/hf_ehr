import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutput
from omegaconf import DictConfig
from typing import Dict, Any, Optional, Tuple, Union
from jaxtyping import Float

from hf_ehr.models.base import BaseModel

def hyena_forward(
    self: AutoModelForCausalLM,
    input_ids: torch.LongTensor = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pad_token_id: Optional[int] = None,
) -> Union[Tuple, CausalLMOutput]:

    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.hyena(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)
    logits = logits.float()

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Add the ignore_index flag to the CrossEntropyLoss initialization
        loss_fct = nn.CrossEntropyLoss()  # Use pad_token_id for ignore_index - ignore_index=pad_token_id
        shift_logits = shift_logits.view(-1, self.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutput(
        loss=loss,
        logits=logits,
        hidden_states=outputs.hidden_states,
    )
    
class HyenaLanguageModel(BaseModel):
    """
    Hyena with a Language Model head.
    """

    def __init__(self, config: DictConfig, vocab_size, pad_token_id) -> None:
        super(HyenaLanguageModel, self).__init__(config, vocab_size, pad_token_id)

        # Model specs
        model_config = AutoConfig.from_pretrained(config.model.hf_name, trust_remote_code=True)
        model_config.vocab_size = vocab_size
        model_config.n_positions = config.data.dataloader.max_length
        for key, val in config.model.config_kwargs.items():
            assert hasattr(model_config, key), f"Config for HF model {self.model_name} does not have attribute {key}"
            setattr(model_config, key, val)
        self.hidden_size = model_config.d_model
        self.model_config = model_config

        # Model
        self.model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True)
        
        # Run any post-init handlers from super()
        self.post_init()
    
    def forward(self, input_ids=None, inputs_embeds=None, labels=None, output_hidden_states=None, return_dict=None):
        return hyena_forward(
            self.model,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pad_token_id=self.pad_token_id
        )
    
    def training_step(self, 
                      batch: Dict[str, Any],
                      batch_idx: int) -> Optional[torch.Tensor]:
        tokens: Dict[str, Float[torch.Tensor, 'B L']] = batch['tokens'].copy()
        B: int = tokens['input_ids'].shape[0]
        
        # Need to adjust for Hyena
        tokens.pop("attention_mask", None)
        tokens.pop("token_type_ids", None)
        
        outputs = hyena_forward(self.model, **tokens, pad_token_id=self.pad_token_id)
        loss: torch.Tensor = outputs.loss
        
        # Learning rate scheduler
        lr: float = self.trainer.lr_scheduler_configs[0].scheduler.optimizer.param_groups[0]["lr"]
        sch = self.lr_schedulers()
        sch.step()
        
        # Logging + Metrics
        self.log_training_step(loss.detach(), B, tokens, lr)

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Optional[torch.Tensor]:
        tokens: Dict[str, Float[torch.Tensor, 'B L']] = batch['tokens']
        B: int = tokens['input_ids'].shape[0]
        
        tokens.pop("attention_mask", None)
        tokens.pop("token_type_ids", None)
        
        # Forward pass
        outputs = hyena_forward(self.model, **tokens, pad_token_id=self.pad_token_id)
        loss: torch.Tensor = outputs.loss

        # Logging
        self.log_validation_step(loss.detach(), tokens)  # Pass both loss and tokens

        return loss
