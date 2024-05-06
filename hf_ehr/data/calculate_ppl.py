from typing import Dict
import torch
import hydra
from omegaconf import DictConfig
from typing import List, Dict, Tuple, Union, Optional
from torch.utils.data import DataLoader
import json
from pytorch_lightning import Trainer

PATH_TO_TOKENIZER_CODE_2_COUNT = '/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9_lite/code_2_count.json'
PATH_TO_TOKENIZER_CODE_2_INT = '/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9_lite/code_2_int.json'

# Assuming FEMRDataset, GPTLanguageModel and load_datasets from your provided modules
from hf_ehr.data.datasets import FEMRDataset
from hf_ehr.data.datasets import FEMRTokenizer
from hf_ehr.models.gpt import GPTLanguageModel
from hf_ehr.trainer.loaders import load_datasets, load_dataloaders

# Perplexity calculation function
def calculate_per_sample_perplexity(model, dataloader, device='cuda'):
    model.eval()
    model.to(device)
    result = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 10:
                break
            # Access 'input_ids' from the nested 'tokens' dictionary
            inputs = batch['tokens']['input_ids'].to(device)
            attention_mask = batch['tokens'].get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            outputs = model.model(input_ids=inputs, attention_mask=attention_mask)
            logits = outputs.logits

            # Shift logits and labels for calculating the loss on the next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs[..., 1:].contiguous()

            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

            if attention_mask is not None:
                attention_mask = attention_mask[..., 1:].contiguous()  # Adjust the mask for shifted labels
                log_probs *= attention_mask

            sum_log_probs = log_probs.sum(1)
            count_tokens = attention_mask.sum(1) if attention_mask is not None else log_probs.size(1)
            avg_log_probs = sum_log_probs / count_tokens
            sample_perplexities = torch.exp(-avg_log_probs)

            #perplexities.extend(sample_perplexities.tolist())
            # Collect results along with input ids for each sequence in the batch
            for i, (inp_id, perplexity) in enumerate(zip(inputs, sample_perplexities)):
                result.append({
                    'input_id': inp_id.tolist(),  # Convert tensors to lists
                    'perplexity': perplexity.item()  # Convert tensors to standard Python numbers
                })

    return result

# Hydra main function to manage configurations
@hydra.main(version_base=None, config_path='../configs/', config_name="config")
def main(config: DictConfig) -> None:
    print("Loading the datasets...")
    datasets = load_datasets(config)

    # Load tokenizer
    atoi: Dict[str, int] = json.load(open(PATH_TO_TOKENIZER_CODE_2_INT, 'r'))
    code_to_count: Dict[str, int] = json.load(open(PATH_TO_TOKENIZER_CODE_2_COUNT, 'r'))
    min_code_count: Optional[int] = None
    tokenizer = FEMRTokenizer(code_to_count, min_code_count=min_code_count)
    
    print("Setting up dataloaders...")
    dataloaders = load_dataloaders(config, datasets, tokenizer)
    
    print("Loading the GPT model...")
    model = GPTLanguageModel.load_from_checkpoint("/share/pi/nigam/migufuen/hf_ehr/cache/runs/gpt2-base-10-epochs/ckpts/epoch=1-epoch.ckpt", tokenizer=tokenizer)
    # Setup Trainer
    #trainer = Trainer()

    # Attach model to trainer to ensure all properties are available
    #trainer.model = model
    
    print("Calculating Perplexity...")
    perplexities = calculate_per_sample_perplexity(model, dataloaders['train'])
    
    # Optionally save perplexities to JSON
    with open('perplexities_10.json', 'w') as f:
        json.dump(perplexities, f, indent=4)
    print("Perplexities calculated and saved.")

if __name__ == "__main__":
    main()
