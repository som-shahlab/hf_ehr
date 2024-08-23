import json
from hf_ehr.trainer.loaders import load_dataloaders, load_datasets
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
    parser.add_argument('--batch_size', type=int, default=16384, help='Batch size (in tokens)')
    parser.add_argument('--is_debug', action='store_true', default=False, help='Debug setting')
    parser.add_argument('--is_use_config_dataloader', action='store_true', default=False, help='Debug setting')
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

def calculate_log_probs_batch(model: BaseModel,
                               batch: Dict[str, Any],
                               config: Dict[str, Any],
                               device: str = "cuda") -> Tuple[float, int]:
    """Calculate log probs for a single batch."""
    # Tokens
    input_ids: Float[torch.Tensor, 'B L'] = batch['tokens']['input_ids'].to(device)

    # Attention mask
    if 'hyena' in config['model']['name']:
        attention_mask = None
    else:
        attention_mask: Float[torch.Tensor, 'B L'] = batch['tokens']['attention_mask'].to(device)

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
    
    # Calculate log probs
    n_tokens: int = attention_mask.sum().item() if attention_mask is not None else log_probs.numel().item()
    log_prob_per_token: float = log_probs.sum().item() / n_tokens

    return log_prob_per_token, n_tokens

def eval(model: BaseModel,
        dataloader: DataLoader,
        config: Dict[str, Any],
        device: str = "cuda",
        is_debug: bool = False) -> Dict[str, Any]:
    """Calculate average perplexity for a dataset split."""
    log_prob_per_tokens: List[float] = []
    n_tokens: List[float] = []

    model.eval()
    model.to(device)
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, total=len(dataloader), desc='eval() | Iterating over dataloader...')):
            batch_log_prob_per_token, batch_n_tokens = calculate_log_probs_batch(model, batch, config, device)
            log_prob_per_tokens.append(batch_log_prob_per_token)
            n_tokens.append(batch_n_tokens)
            if is_debug and sum(n_tokens) > 16384 * 500:
                break

    loss_per_token: float = calc_weighted_average(log_prob_per_tokens, n_tokens)
    ppl_per_token: float = float(np.exp(-loss_per_token))

    return {
        "loss_per_token" : loss_per_token,
        "ppl_per_token": ppl_per_token,
        "n_tokens": sum(n_tokens),
        "n_batches" : batch_idx + 1,
    }

def get_path_to_output_dir(path_to_ckpt: str, split: str) -> str:
    """Create output directory if it does not exist."""
    path_to_ckpt_dir = os.path.dirname(path_to_ckpt)
    path_to_output_dir = os.path.abspath(os.path.join(path_to_ckpt_dir, ".." ,"ppl", split))
    os.makedirs(path_to_output_dir, exist_ok=True)
    return path_to_output_dir

def run_ckpt(path_to_ckpt: str, path_to_output: str, dataset_name: str, split: str, device: str, batch_size: int, is_eval_debug: bool = False) -> None:
    """Load ckpt and run eval() on it"""
    initial_start = time.time()
    # Load model, tokenizer, config
    ckpt: Dict[str, Any] = load_ckpt(path_to_ckpt)
    model: BaseModel = load_model_from_path(path_to_ckpt)
    config: Dict[str, Any] = load_config(ckpt)
    tokenizer: BaseTokenizer = load_tokenizer_from_config(config)
    print("Model config: ", config)
    
    # Load Dataset
    # print("Start | Loading dataset")
    # start = time.time()
    # path_to_femr_extract: str = config.data.dataset.path_to_femr_extract
    # max_length: int = config.data.dataloader.max_length
    # is_debug: bool = getattr(config.data.dataset, 'is_debug', False)
    # seed: int = config.main.seed
    # if dataset_name == 'FEMRDataset':
    #     dataset = FEMRDataset(path_to_femr_extract, split=split, is_debug=is_debug, seed=seed)
    #     idx_to_seq_length: List[int] = tokenizer.get_seq_length_per_patient(dataset)
    # elif dataset_name == 'AllTokensFEMRDataset':
    #     dataset = AllTokensFEMRDataset(tokenizer, max_length, path_to_femr_extract, split=split, is_debug=is_debug, seed=seed)
    #     idx_to_seq_length: List[int] = dataset.idx_to_seq_length
    # else:
    #     raise ValueError(f"Unknown dataset_name: {dataset_name}")
    # print("Finish | Loading dataset | t=", time.time() - start)
    
    # Load Dataloader
    # print("Start | Loading dataloader")
    # start = time.time()
    # sort_sampler = SortishSampler( idx_to_seq_length, 1, is_random_shuffle_across_buckets=False, is_random_shuffle_within_buckets=False, n_replicas=1)
    # batch_sampler = ApproxBatchSampler( idx_to_seq_length, sort_sampler, max_length, batch_size, )
    # batch_sampler_kwargs = { 'batch_sampler' : batch_sampler, }
    # dataloader = DataLoader(
    #     dataset=dataset,
    #     collate_fn=lambda x: collate_femr_timelines(x, tokenizer, dataset_name, max_length, is_truncation_random=False, is_mlm=False, mlm_prob=0, seed=seed),
    #     num_workers=4,
    #     pin_memory=True,
    #     **batch_sampler_kwargs,
    # )
    # print("Finish | Loading dataloader | t=", time.time() - start)

    # Miguel version
    OmegaConf.set_struct(config, False)
    config.data.dataset.name = 'FEMRDataset' # TODO - remove
    datasets: Dict[str, Any] = load_datasets(config, tokenizer)
    print("Loading dataloaders...")
    dataloaders: Dict[str, DataLoader] = load_dataloaders(config, datasets, tokenizer)
    dataloader: DataLoader = dataloaders[split]
    
    # Run eval
    print("Start | Calculating average perplexity")
    start = time.time()
    results = eval(model, dataloader, config, device, is_debug=is_eval_debug)
    print("Mean ppl per token: ", results['ppl_per_token'])
    print("Total tokens: ", results['n_tokens'])
    print("Finish | Calculating average perplexity | t=", time.time() - start)

    # Save results
    results = {
        'timestamp' : str(datetime.datetime.now()),
        "runtime_seconds": time.time() - initial_start,
        'split' : split,
        'dataset' : dataset_name,
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
    path_to_output_dir: str = get_path_to_output_dir(path_to_ckpt_dir, split)
    batch_size: int = args.batch_size
    is_debug: bool = args.is_debug

    # Find all .ckpt files in `path_to_ckpt_dir`
    skipped_ckpts: List[str] = []
    for file in os.listdir(path_to_ckpt_dir):
        if not file.endswith(".ckpt"):
            continue

        # TODO - remove
        if file != 'epoch=1-step=150000-persist.ckpt':
            continue

        # Get paths
        path_to_ckpt: str = os.path.join(path_to_ckpt_dir, file)
        path_to_output: str = os.path.join(path_to_output_dir, f"{file.replace('.ckpt', '')}.json")

        # Run eval
        print("#"* 50)
        print(f"Start | Processing model ckpt @ `{path_to_ckpt}`")
        try:
            run_ckpt(path_to_ckpt, path_to_output, dataset, split, device, batch_size, is_debug)
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
