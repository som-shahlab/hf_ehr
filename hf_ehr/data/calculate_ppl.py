from typing import Dict
import torch
import hydra
from omegaconf import DictConfig
from typing import List, Dict, Tuple, Union, Optional
from torch.utils.data import DataLoader
import json
from pytorch_lightning import Trainer
from tqdm import tqdm

PATH_TO_TOKENIZER_CODE_2_COUNT = '/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9_lite/code_2_count.json'
PATH_TO_TOKENIZER_CODE_2_INT = '/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9_lite/code_2_int.json'

# Assuming FEMRDataset, GPTLanguageModel and load_datasets from your provided modules
from hf_ehr.data.datasets import FEMRDataset
from hf_ehr.data.datasets import FEMRTokenizer
from hf_ehr.models.gpt import GPTLanguageModel
from hf_ehr.trainer.loaders import load_datasets, load_dataloaders

# Perplexity calculation function for a single batch
def calculate_perplexity_batch(model, batch, device='cuda'):
    model.eval()
    model.to(device)
    results = []

    with torch.no_grad():
        inputs = batch['tokens']['input_ids'].to(device)
        attention_mask = batch['tokens'].get('attention_mask', None)
        patient_ids = batch['patient_ids']
        #print(batch['patient_ids'])
        #print(inputs)
        # double check attention masks
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        outputs = model.model(input_ids=inputs, attention_mask=attention_mask)
        # logits from model output
        logits = outputs.logits
        
        # shift logits from model output to align with labels
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs[..., 1:].contiguous()

        # gather log probs
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

        # attention mask is used weed out irrelevant tokens - like padding tokens
        if attention_mask is not None:
            attention_mask = attention_mask[..., 1:].contiguous()
            log_probs *= attention_mask

        # sum the log probabilites for all tokens in the sequence
        sum_log_probs = log_probs.sum(1)
        count_tokens = attention_mask.sum(1) if attention_mask is not None else log_probs.size(1)
        avg_log_probs = sum_log_probs / count_tokens
        sample_perplexities = torch.exp(-avg_log_probs)

        for pid, inp_id, perplexity in zip(patient_ids, inputs, sample_perplexities):
            results.append({
                'patient_id': int(pid),
                #'input_id': inp_id.tolist(),
                'perplexity': perplexity.item()
            })

    return results

# Process and save results in batches
def process_and_save_batches(model, dataloader, output_path, batch_size=100):
    all_results = []
    batch_results = []
    patient_count = 0
    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch_results = calculate_perplexity_batch(model, batch)
        all_results.extend(batch_results)
        patient_count += len(batch_results)
        with open(output_path, 'w') as f:
                json.dump(all_results, f)

    """
    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch_results.extend(calculate_perplexity_batch(model, batch))
        if (batch_idx + 1) % batch_size == 0 or batch_idx == len(dataloader) - 1:
            all_results.extend(batch_results)
            # Save to file periodically
            with open(output_path, 'w') as f:
                json.dump(all_results, f, indent=4)
            batch_results = []

    print("Perplexities calculated and saved.")
    """
    

# Hydra main function to manage configurations
@hydra.main(version_base=None, config_path='../configs/', config_name="config")
def main(config: DictConfig) -> None:
    print("Loading the datasets...")
    datasets = load_datasets(config)

    # Load tokenizer
    #atoi: Dict[str, int] = json.load(open(PATH_TO_TOKENIZER_CODE_2_INT, 'r'))
    code_to_count: Dict[str, int] = json.load(open(PATH_TO_TOKENIZER_CODE_2_COUNT, 'r'))
    min_code_count: Optional[int] = None
    tokenizer = FEMRTokenizer(code_to_count, min_code_count=min_code_count)
    
    print("Setting up dataloaders...")
    dataloaders = load_dataloaders(config, datasets, tokenizer)
    
    # Setup Trainer
    #trainer = Trainer()
    print("Loading the GPT model...")
    model = GPTLanguageModel.load_from_checkpoint("/share/pi/nigam/migufuen/hf_ehr/cache/runs/gpt2-base-h100-1gpu/ckpts/last.ckpt", tokenizer=tokenizer)

    # Attach model to trainer to ensure all properties are available
    #trainer.model = model
    
    print("Calculating Perplexity...")
    output_path = '/share/pi/nigam/suhana/hf_ehr_repo/hf_ehr/hf_ehr/data/perplexities_new_full.json'
    process_and_save_batches(model, dataloaders['train'], output_path)
    #perplexities = calculate_per_sample_perplexity(model, dataloaders['train'])

    print("Perplexities calculated and saved.")

if __name__ == "__main__":
    main()
