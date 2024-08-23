import json
import numpy as np
import os
import time
from hf_ehr.models.modules import BaseModel
import torch

from argparse import ArgumentParser, Namespace
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Tuple
from typing import Dict
from tqdm import tqdm
from jaxtyping import Float

from hf_ehr.config import H100_BASE_DIR, A100_BASE_DIR, V100_BASE_DIR, GPU_BASE_DIR
from hf_ehr.data.datasets import BaseDataset
from hf_ehr.data.tokenization import CLMBRTokenizer, CookbookTokenizer, DescTokenizer
from hf_ehr.trainer.loaders import load_datasets, load_dataloaders
from hf_ehr.utils import load_config_from_ckpt, load_tokenizer_from_config, load_model_from_path, load_ckpt


def parse_arguments() -> Namespace:
    """"Parse command-line arguments."""
    parser = ArgumentParser(description='Calculate PPL for a model checkpoint.')
    parser.add_argument('--path_to_ckpt_dir', type=str, required=True, help='Path to model checkpoint directory')
    parser.add_argument('--device', type=str, default="cuda", help='Device to run validation on')
    parser.add_argument('--split', type=str, default="val", help='Split on which to calculate PPL')
    return parser.parse_args()

def patch_config(config: DictConfig) -> None:
    """Rewrite paths for Carina partitions."""

    base_dir = GPU_BASE_DIR
    femr_dataset_dirname = os.path.basename(config.data.dataset.path_to_femr_extract)
    # 'path_to_femr_extract': '/local-scratch/nigam/hf_ehr/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_02_08_extract_v8_no_notes'
    if os.environ.get('SLURM_JOB_PARTITION') == 'nigam-v100':
        base_dir = V100_BASE_DIR
    elif os.environ.get('SLURM_JOB_PARTITION') == 'nigam-a100':
        base_dir = A100_BASE_DIR
    elif os.environ.get('SLURM_JOB_PARTITION') == 'nigam-h100':
        base_dir = H100_BASE_DIR
    elif os.environ.get('SLURM_JOB_PARTITION') == 'gpu':
        base_dir = GPU_BASE_DIR
    config.data.dataset.path_to_femr_extract = os.path.join(base_dir, femr_dataset_dirname)

def load_config(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    """Load configuration from a checkpoint."""
    config = load_config_from_ckpt(ckpt)
    patch_config(config)
    return config

def calculate_perplexity_batch(model: BaseModel,
                               batch: Dict[str, Any],
                               config: Dict[str, Any],
                               device: str = "cuda") -> Tuple[float, int]:
    """Calculate perplexity for a single batch."""

    model.eval()
    model.to(device)

    total_log_probs = 0.0
    total_token_count = 0
    with torch.no_grad():
<<<<<<< HEAD
        inputs: Float[torch.Tensor, 'B L'] = batch['tokens']['input_ids'].to(device)
        attention_mask: Float[torch.Tensor, 'B L'] = batch['tokens'].get('attention_mask', None)
=======
        inputs = batch['tokens']['input_ids'].to(device)

        if 'hyena' in config['model']['name']:
            batch["tokens"].pop('attention_mask') 
            
        attention_mask = batch['tokens'].get('attention_mask', None)
>>>>>>> f00fbf8ae17df227404df819aa0f5b86fe8f6d8e



        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        outputs = model.model(input_ids=inputs, attention_mask=attention_mask)

        logits: Float[torch.Tensor, 'B L V'] = outputs.logits

        shift_logits: Float[torch.Tensor, 'B L-1 V'] = logits[:, :-1, :].contiguous()
        shift_labels: Float[torch.Tensor, 'B L-1'] = inputs[:, 1:].contiguous()

        log_probs: Float[torch.Tensor, 'B L-1 V'] = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        log_probs: Float[torch.Tensor, 'B L-1'] = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

        if attention_mask is not None:
            attention_mask: Float[torch.Tensor, 'B L-1'] = attention_mask[:, 1:].contiguous()
            log_probs *= attention_mask

        sum_log_probs: float = log_probs.sum().item()
        count_tokens: int = attention_mask.sum().item() if attention_mask is not None else log_probs.size(1).item()

        total_log_probs += sum_log_probs
        total_token_count += count_tokens

    return total_log_probs, total_token_count

def calculate_avg_ppl(model: BaseModel,
                      dataloader: DataLoader,
                      config: Dict[str, Any],
                      device: str = "cuda") -> Dict[str, Any]:
    """Calculate average perplexity for a dataset split."""

    total_log_probs = 0.0
    total_token_count = 0

    for batch in tqdm(dataloader, total=len(dataloader)):
        batch_log_probs, batch_token_count = calculate_perplexity_batch(model, batch, config, device)
        total_log_probs += batch_log_probs
        total_token_count += batch_token_count

    avg_log_probs = total_log_probs / total_token_count
    avg_ppl = np.exp(-avg_log_probs)

    results = {
        "avg_loss" : avg_log_probs,
        "avg_ppl": avg_ppl,
        "total_nonPAD_token_count": total_token_count
    }

    return results
    
def save_results(results: Dict[str, Any],
                 path_to_output: str) -> None:
    """Save perplexity results to a file."""
    with open(path_to_output, 'w') as f:
        json.dump(results, f)

def get_output_dir(ckpt_path: str, split: str) -> None:
    """Create output directory if it does not exist."""

    ckpt_dir = os.path.dirname(ckpt_path)
    output_dir = os.path.join(ckpt_dir, ".." ,"ppl", split)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    return output_dir

def process_ckpt(path_to_ckpt: str, output_path: str, split: str, device: str) -> None:
    ckpt: str = load_ckpt(path_to_ckpt)
    model: BaseModel = load_model_from_path(path_to_ckpt)
    config: Dict[str, Any] = load_config(ckpt)
    print("Model config: ", config)
    tokenizer: CLMBRTokenizer | DescTokenizer | CookbookTokenizer = load_tokenizer_from_config(config)
    
    
    dataset_name: str = config.data.dataset.name
    path_to_femr_extract: str = config.data.dataset.path_to_femr_extract
    is_debug: bool = getattr(config.data.dataset, 'is_debug', False)
    seed: int = config.main.seed

    # Load datasets
    if dataset_name == 'FEMRDataset':
        val_dataset = FEMRDataset(path_to_femr_extract, split='val', is_debug=is_debug, seed=seed)
    elif dataset_name == 'AllTokensFEMRDataset':
        val_dataset = AllTokensFEMRDataset(tokenizer, max_length, path_to_femr_extract, split='val', is_debug=is_debug, seed=seed)
        
    
    batch_max_tokens = 16384
    if dataset_name == 'FEMRDataset':
        val_idx_to_seq_length: List[int] = tokenizer.get_seq_length_per_patient(datasets['val'])
    elif dataset_name == 'AllTokensFEMRDataset':
        val_idx_to_seq_length: List[int] = datasets['val'].idx_to_seq_length
    
    val_sort_sampler = SortishSampler( val_idx_to_seq_length, 1, is_random_shuffle_across_buckets=False, is_random_shuffle_within_buckets=False, n_replicas=n_replicas)
    val_batch_sampler = ApproxBatchSampler( val_idx_to_seq_length, val_sort_sampler, max_length, batch_max_tokens, )
    val_batch_sampler_kwargs = { 'batch_sampler' : val_batch_sampler, }

    val_loader = DataLoader(
        dataset=datasets['val'],
        collate_fn=lambda x: collate_femr_timelines(x, tokenizer, dataset_name, max_length, is_truncation_random, is_mlm, mlm_prob, seed),
        num_workers=n_workers,
        pin_memory=True,
        **val_batch_sampler_kwargs,
    )
    
    datasets: Dict[str, BaseDataset] = load_datasets(config, tokenizer)
    print("Loading dataloaders...")
    dataloaders: Dict[str, DataLoader] = load_dataloaders(config, datasets, tokenizer)
    dataloader: DataLoader = dataloaders[split]
    
    print("Calculating average perplexity...")
    start_time = time.time()
    results = calculate_avg_ppl(model, dataloader, config, device)
    end_time = time.time()
    elapsed_time = end_time - start_time

    results["split"] = split
    results["stats"] = {
        "time_taken_seconds": elapsed_time,
        "num_batches": len(dataloader)
    }
    results["model_ckpt"] = path_to_ckpt
    results["model_conifg"] = OmegaConf.to_container(config, resolve=True)

    save_results(results, args.output_path)
    print("Average perplexity: ", results["avg_ppl"])
    print("Stats: ", results["stats"])
    print(f"Results for {split} saved under {output_path}.")


def main() -> None:
    args: Namespace = parse_arguments()
    
    path_to_ckpt_dir: str = args.path_to_ckpt_dir
    device: str = args.device
    split: str = args.split
    output_dir: str = get_output_dir(args.path_to_ckpt_dir, split)
    unevaluated_ckpts: List[str] = []
    for file in os.listdir(path_to_ckpt_dir):
        if not file.endswith(".ckpt"):
            continue
        path_to_ckpt: str = os.path.join(path_to_ckpt_dir, file)
        output_path: str = os.path.join(output_dir, f"{file}.json")
        print("#"* 50)
        print(f"Processing model checkpoint : {path_to_ckpt}")
        print("-"* 30)
        try:
            process_ckpt(path_to_ckpt, output_path, args.split, device)
        except Exception as e:
            print(f"Error processing checkpoint {path_to_ckpt}: {e}")
            unevaluated_ckpts.append(path_to_ckpt)
    
    if len(unevaluated_ckpts) > 0:
        print("Unevaluated checkpoints: ")
        print(unevaluated_ckpts)

if __name__ == "__main__":
    main()
