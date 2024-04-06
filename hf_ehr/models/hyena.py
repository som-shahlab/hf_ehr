import torch
import torch.nn as nn
from torch import optim
from transformers import AutoModel, AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutput
from omegaconf import DictConfig
from typing import Dict, Any, Optional, Tuple, Union
from jaxtyping import Float
from hf_ehr.models.modules import BaseModel
from hf_ehr.utils import lr_warmup_with_constant_plateau

class HyenaLanguageModel(BaseModel):
    """
    Hyena with a Language Model head.
    """

    def __init__(self, config: DictConfig, tokenizer) -> None:
        super(HyenaLanguageModel, self).__init__(config, tokenizer)

        # Model specs
        model_config = AutoConfig.from_pretrained(config.model.hf_name, trust_remote_code=True)
        model_config.vocab_size = tokenizer.vocab_size
        model_config.n_positions = config.data.dataloader.max_length
        for key, val in config.model.config_kwargs.items():
            assert hasattr(model_config, key), f"Config for HF model {self.model_name} does not have attribute {key}"
            setattr(model_config, key, val)
        self.hidden_size = model_config.d_model
        self.model_config = model_config

        # Model
        self.model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True)
        self.lm_head = nn.Linear(self.hidden_size, tokenizer.vocab_size, bias=False)
    
    """
    def configure_optimizers(self):
        #Sets Learning rate for different parameter groups.
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
    """
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutput]:

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model_config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.model_config.use_return_dict

        outputs = self.model(
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
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )

    def training_step(self, 
                      batch: Dict[str, Any],
                      batch_idx: int) -> Optional[torch.Tensor]:
        # TODO (@Suhana) -- adapt for Hyena
        tokens = {key: value for key, value in batch['tokens'].items() if key != 'attention_mask'}
        #tokens: Dict[str, Float[torch.Tensor, 'B L']] = batch['tokens']
        B: int = tokens['input_ids'].shape[0]

        outputs = self.model(input_ids=tokens['input_ids'], labels=batch.get('labels', None))
        loss: torch.Tensor = outputs.loss
        
        # Learning rate scheduler
        lr: float = self.trainer.lr_scheduler_configs[0].scheduler.optimizer.param_groups[0]["lr"]
        sch = self.lr_schedulers()
        sch.step()
        
        # Logging + Metrics
        self.log_training_step(loss.detach(), B, tokens, lr)

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Optional[torch.Tensor]:
        """
        Perform a validation step.

        Args:
            batch (Dict[str, Any]): The batch of data from the validation DataLoader.
            batch_idx (int): The index of the current batch.

        Returns:
            Optional[torch.Tensor]: The validation loss of the current batch.
        """
        # Prepare inputs
        input_ids = batch["tokens"]["input_ids"]
        labels = batch.get('labels', None)  # Assuming 'labels' is directly in 'batch'
        attention_mask = batch["tokens"].get("attention_mask", None)

        # Pass only the necessary arguments to the model
        model_inputs = {'input_ids': input_ids}
        if attention_mask is not None:
            model_inputs['attention_mask'] = attention_mask

        # Forward pass
        with torch.no_grad():  # No gradients needed for validation
            outputs = self.model(**model_inputs, labels=labels)
            loss = outputs.loss

        # Optionally, add logging or metric tracking here.
        # Example: self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss