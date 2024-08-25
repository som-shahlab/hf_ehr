import json
from hf_ehr.trainer.loaders import load_dataloaders, load_datasets
import numpy as np
import os
import time
from hf_ehr.models.modules import BaseModel
import torch
from argparse import ArgumentParser, Namespace
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any, List, Tuple
from typing import Dict
from tqdm import tqdm
from jaxtyping import Float
import datetime
import traceback
from hf_ehr.config import H100_BASE_DIR, A100_BASE_DIR, V100_BASE_DIR, GPU_BASE_DIR
from hf_ehr.data.datasets import AllTokensFEMRDataset, FEMRDataset
from hf_ehr.data.tokenization import BaseTokenizer, collate_femr_timelines
from hf_ehr.trainer.samplers import SortishSampler, ApproxBatchSampler
from hf_ehr.utils import load_config_from_ckpt, load_tokenizer_from_config, load_model_from_path, load_ckpt

def parse_args() -> Namespace:
    """"Parse command-line arguments."""
    parser = ArgumentParser(description='Calculate PPL for a model checkpoint.')
    parser.add_argument('--path_to_ckpt_dir', type=str, required=True, help='Path to model checkpoint directory')
    parser.add_argument('--device', type=str, default="cuda", help='Device to run validation on')
    parser.add_argument('--split', type=str, default="val", help='Split on which to calculate PPL')
    parser.add_argument('--dataset', type=str, default="FEMRDataset", help='Type of dataset -- AllTokensFEMRDataset or AllTokensDataset')
    parser.add_argument('--stride', type=int, default=512, help='Stride')
    parser.add_argument('--n_patients', type=int, default=10, help='# of val patients')
    parser.add_argument('--is_debug', action='store_true', default=False, help='Debug setting')
    parser.add_argument('--is_load_from_config', action='store_true', default=False,  help='If TRUE, load dataset based on config')
    return parser.parse_args()

def patch_config(config: DictConfig) -> None:
    """Rewrite paths for Carina partitions."""
    base_dir = GPU_BASE_DIR
    femr_dataset_dirname = os.path.basename(config.data.dataset.path_to_femr_extract)
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

def calc_weighted_average(values: List[float], weights: List[float]) -> float:
    """Stable weighted average of `values` (weighted by `weights`)"""
    assert len(values) == len(weights), "Error -- len(values) must equal len(weights)"

    # Convert to numpy arrays
    values = np.array(values, dtype=np.float64)
    weights = np.array(weights, dtype=np.float64)
    
    # Normalize weights by their sum to avoid very large numbers
    sum_weights = np.sum(weights)
    normalized_weights = weights / sum_weights

    # Calculate the weighted average
    weighted_avg: float = float(np.sum(normalized_weights * values))
    
    return weighted_avg


def eval(model: BaseModel,
            dataset,
            tokenizer,
            max_length: int,
            p_idxs: List[int],
            stride: int,
            config: Dict[str, Any],
            device: str = "cuda",
            is_debug: bool = False) -> Dict[str, Any]:
    """Calculate average perplexity for a dataset split."""
    log_prob_per_tokens: List[float] = []
    n_tokens: List[float] = []

    model.eval()
    model.to(device)
    with torch.no_grad():
        for p_idx in tqdm(p_idxs, total=len(p_idxs), desc='eval() | Iterating over patients...'):
            # Tokenize this patients timeline
            pid, events = dataset[p_idx]
            tokens: Dict[str, Float[torch.Tensor, 'B max_length']] = tokenizer([ events ], 
                                                                                truncation=False, 
                                                                                padding=False, 
                                                                                max_length=max_length,
                                                                                is_truncation_random=False,
                                                                                add_special_tokens=False,
                                                                                return_tensors='pt')
            tokens['labels'] = tokens['input_ids']
            
            # Split timeline into batches of length `max_length` for model to ingest
            for start_idx in range(0, tokens['input_ids'].shape[1] - max_length, stride):
                input_ids: Float[torch.Tensor, 'B L'] = tokens['input_ids'][:,start_idx:start_idx + max_length].to(device)

                # Attention mask
                if 'hyena' in config['model']['name']:
                    attention_mask = None
                else:
                    attention_mask: Float[torch.Tensor, 'B L'] = tokens['attention_mask'][:,start_idx:start_idx + max_length].to(device)

                # Run model
                outputs = model.model(input_ids=input_ids, attention_mask=attention_mask)
                logits: Float[torch.Tensor, 'B L V'] = outputs.logits

                # Calculate log probs
                shift_logits: Float[torch.Tensor, 'B L-1 V'] = logits[:, :-1, :].contiguous()
                shift_labels: Float[torch.Tensor, 'B L-1'] = input_ids[:, 1:].contiguous()
                log_probs: Float[torch.Tensor, 'B L-1 V'] = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                log_probs: Float[torch.Tensor, 'B L-1'] = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
                if attention_mask is not None:
                    attention_mask: Float[torch.Tensor, 'B L-1'] = attention_mask[:, :-1].contiguous()
                    log_probs *= attention_mask
                
                # Save results depending on stride
                if start_idx == 0:
                    # Keep all tokens
                    n_token: int = attention_mask.sum().item() if attention_mask is not None else log_probs.numel().item()
                    log_prob_per_token: float = log_probs.sum().item() / n_token
                else:
                    # Keep only last `stride` tokens
                    n_token: int = attention_mask[:,-stride:].sum().item() if attention_mask is not None else log_probs[:,-stride:].numel().item()
                    log_prob_per_token: float = log_probs[:,-stride:].sum().item() / n_token
                    assert n_token <= stride, f"Error -- n_token={n_token} must be <= stride={stride}"
                assert n_token <= max_length, f"Error -- n_token={n_token} must be <= max_length={max_length}"

                n_tokens.append(n_token)
                log_prob_per_tokens.append(log_prob_per_token)
                print(f"pid={pid} | n_events={len(events)} | n_tokens={tokens['input_ids'].shape[1]} | start={start_idx} | end={min(tokens['input_ids'].shape[1], start_idx + max_length)} | toks={n_token} | ppl={np.exp(-log_prob_per_token)}")

            if is_debug and sum(n_tokens) > 16384 * 500:
                break

    loss_per_token: float = calc_weighted_average(log_prob_per_tokens, n_tokens)
    ppl_per_token: float = float(np.exp(-loss_per_token))

    return {
        "loss_per_token" : loss_per_token,
        "ppl_per_token": ppl_per_token,
        "n_tokens": sum(n_tokens),
        "n_batches" : len(p_idxs),
    }

def get_path_to_output_dir(path_to_ckpt: str, split: str, dataset: str) -> str:
    """Create output directory if it does not exist."""
    path_to_ckpt_dir = os.path.dirname(path_to_ckpt)
    path_to_output_dir = os.path.abspath(os.path.join(path_to_ckpt_dir, ".." ,"ppl", split, dataset))
    os.makedirs(path_to_output_dir, exist_ok=True)
    return path_to_output_dir

def run_ckpt(path_to_ckpt: str, 
             path_to_output: str, 
             dataset,
             split: str, 
             n_patients: int,
             stride: int,
             device: str, 
             is_load_from_config: bool, 
             is_eval_debug: bool = False) -> None:
    """Load ckpt and run eval() on it"""
    initial_start = time.time()
    # Load model, tokenizer, config
    ckpt: Dict[str, Any] = load_ckpt(path_to_ckpt)
    model: BaseModel = load_model_from_path(path_to_ckpt)
    config: Dict[str, Any] = load_config(ckpt)
    tokenizer: BaseTokenizer = load_tokenizer_from_config(config)
    print("Model config: ", config)
    
    # Load dataset/dataloader using fixed settings
    print("Start | Loading dataset")
    start = time.time()
    path_to_femr_extract: str = config.data.dataset.path_to_femr_extract
    max_length: int = config.data.dataloader.max_length
    is_debug: bool = getattr(config.data.dataset, 'is_debug', False)
    seed: int = config.main.seed
    dataset = FEMRDataset(path_to_femr_extract, split=split, is_debug=is_debug, seed=seed)
    print("Finish | Loading dataset | t=", time.time() - start)
    
    max_length = 24 # TODO - remove

    assert stride <= max_length, f"Error -- stride={stride} must be <= max_length={max_length} of model, otherwise tokens will be skipped"

    # Run eval
    print("Start | Calculating average perplexity")
    start = time.time()
    p_idxs: List[int] = list(range(n_patients))
    results = eval(model, dataset, tokenizer, max_length, p_idxs, stride, config, device, is_debug=is_eval_debug)
    print("Mean ppl per token: ", results['ppl_per_token'])
    print("Total tokens: ", results['n_tokens'])
    print("Finish | Calculating average perplexity | t=", time.time() - start)

    # Save results
    results = {
        'timestamp' : str(datetime.datetime.now()),
        "runtime_seconds": time.time() - initial_start,
        'split' : split,
        'dataset' : 'FEMRDataset',
        'path_to_ckpt' : path_to_ckpt,
        'config' : OmegaConf.to_container(config, resolve=True),
        'is_debug' : is_eval_debug,
        'results' : {
            'loss_per_token' : results['loss_per_token'],
            'ppl_per_token' : results['ppl_per_token'],
            'n_tokens' : results['n_tokens'],
        }
    }
    with open(path_to_output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results @ `{path_to_output}`")


def main() -> None:
    args = parse_args()
    path_to_ckpt_dir: str = args.path_to_ckpt_dir
    if not 'ckpts' in path_to_ckpt_dir:
        # Go to `ckpts/` subfolder if not directly specified
        path_to_ckpt_dir = os.path.join(path_to_ckpt_dir, 'ckpts/')
    device: str = args.device
    split: str = args.split
    dataset: str = args.dataset
    stride: int = args.stride
    n_patients: int = args.n_patients
    is_load_from_config: bool = args.is_load_from_config
    is_debug: bool = args.is_debug
    path_to_output_dir: str = get_path_to_output_dir(path_to_ckpt_dir, split, f"dataset={dataset}-is_config={is_load_from_config}")

    # Find all .ckpt files in `path_to_ckpt_dir`
    skipped_ckpts: List[str] = []
    for file in os.listdir(path_to_ckpt_dir):
        if not file.endswith(".ckpt"):
            continue

        # TODO - remove
        if not (
            file.startswith('train-tokens-total_nonPAD')
            and 'ckpt_val=2100000000-persist' in file
        ):
            continue

        # Get paths
        path_to_ckpt: str = os.path.join(path_to_ckpt_dir, file)
        path_to_output: str = os.path.join(path_to_output_dir, f"{file.replace('.ckpt', '')}.json")

        # Run eval
        print("#"* 50)
        print(f"Start | Processing model ckpt @ `{path_to_ckpt}`")
        try:
            run_ckpt(path_to_ckpt, path_to_output, dataset, split, n_patients, stride, device, is_load_from_config, is_debug)
        except Exception as e:
            print(f"Error processing checkpoint @ `{path_to_ckpt}`: {e}")
            traceback.print_exc()
            skipped_ckpts.append(path_to_ckpt)
            
        print(f"Finish | Processing model ckpt @ `{path_to_ckpt}`")
        print("#"* 50)
    
    if len(skipped_ckpts) > 0:
        print("Skipped checkpoints:", skipped_ckpts)

if __name__ == "__main__":
    main()
