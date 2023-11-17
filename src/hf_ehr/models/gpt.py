import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Dict, List, Any, Optional, Union, Tuple
from omegaconf import DictConfig
from hf_ehr.models.modules import CausalModel

class GPTLanguageModel(CausalModel):
    """
    GPT2 with a Language Model head.
    """

    def __init__(self, config: DictConfig, tokenizer) -> None:
        super(GPTLanguageModel, self).__init__(config)
        self.save_hyperparameters()
        self.model_name: str = config.model.name
        self.config = config

        # Model specs
        model_config = AutoConfig.from_pretrained(self.model_name)
        model_config.vocab_size = tokenizer.vocab_size
        model_config.n_positions = config.data.dataloader.max_length
        for key, val in config.model.config_kwargs.items():
            assert hasattr(model_config, key), f"Config for HF model {self.model_name} does not have attribute {key}"
            setattr(model_config, key, val)
        self.hidden_size = model_config.n_embd

        # Model
        self.model = AutoModel.from_config(model_config)
        self.lm_head = nn.Linear(self.hidden_size, tokenizer.vocab_size, bias=False)

        # Tokenizer
        self.tokenizer = tokenizer