import torch.nn as nn
from transformers import AutoModel, AutoConfig
from omegaconf import DictConfig
from hf_ehr.models.modules import CausalModel
from typing import Union, Tuple, Dict, List
from jaxtyping import Float
import torch

class T5LanguageModel(CausalModel):
    """
    T5 with a Language Model head.
    """

    def __init__(self, config: DictConfig, tokenizer) -> None:
        super(T5LanguageModel, self).__init__(config)
        self.save_hyperparameters()

        # Tokenizer
        self.tokenizer = tokenizer

        # Model specs
        model_config = AutoConfig.from_pretrained(config.model.hf_name)
        model_config.vocab_size = tokenizer.vocab_size
        for key, val in config.model.config_kwargs.items():
            assert hasattr(model_config, key), f"Config for HF model {self.model_name} does not have attribute {key}"
            setattr(model_config, key, val)
        self.hidden_size = model_config.d_model

        # Model
        self.model = AutoModel.from_config(model_config)
        self.lm_head = nn.Linear(self.hidden_size, tokenizer.vocab_size, bias=False)

    
    def forward(self, tokens: Dict[str, Float[torch.Tensor, 'B L']], is_return_hidden_states: bool = True) -> Union[Tuple[Float[torch.Tensor, 'B L V'], Float[torch.Tensor, 'B L H']], Float[torch.Tensor, 'B L V']]:
        B: int = tokens['input_ids'].shape[0]
        L: int = tokens['input_ids'].shape[1]
        H: int = self.hidden_size
        V: int = self.tokenizer.vocab_size
        # TODO - ValueError: You have to specify either decoder_input_ids or decoder_inputs_embeds
        hidden_states: Float[torch.Tensor, 'B L H'] = self.model(tokens['input_ids']).last_hidden_state
        logits: Float[torch.Tensor, 'B L V'] = self.lm_head(hidden_states)
        assert hidden_states.shape == (B, L, H)
        assert logits.shape == (B, L, V)

        if is_return_hidden_states:
            return logits, hidden_states
        else:
            return logits