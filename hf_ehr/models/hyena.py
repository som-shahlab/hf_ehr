import torch.nn as nn
from torch import optim
from transformers import AutoModel, AutoConfig
from omegaconf import DictConfig
from hf_ehr.models.modules import CausalModel
from hf_ehr.utils import lr_warmup_with_constant_plateau

class HyenaLanguageModel(CausalModel):
    """
    Hyena with a Language Model head.
    """

    def __init__(self, config: DictConfig, tokenizer) -> None:
        super(HyenaLanguageModel, self).__init__(config)
        self.save_hyperparameters()

        # Tokenizer
        self.tokenizer = tokenizer

        # Model specs
        model_config = AutoConfig.from_pretrained(config.model.hf_name, trust_remote_code=True)
        model_config.vocab_size = tokenizer.vocab_size
        for key, val in config.model.config_kwargs.items():
            assert hasattr(model_config, key), f"Config for HF model {self.model_name} does not have attribute {key}"
            setattr(model_config, key, val)
        self.hidden_size = model_config.d_model

        # Model
        self.model = AutoModel.from_config(model_config, trust_remote_code=True)
        self.lm_head = nn.Linear(self.hidden_size, tokenizer.vocab_size, bias=False)

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