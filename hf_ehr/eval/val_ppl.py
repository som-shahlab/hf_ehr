import json
import numpy as np
import os
import time
from hf_ehr.models.modules import BaseModel
import torch

from argparse import ArgumentParser, Namespace
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple
from typing import Dict
from tqdm import tqdm

from hf_ehr.config import H100_BASE_DIR, A100_BASE_DIR, V100_BASE_DIR, GPU_BASE_DIR
from hf_ehr.data.datasets import BaseDataset
from hf_ehr.data.tokenization import CLMBRTokenizer, CookbookTokenizer, DescTokenizer
from hf_ehr.trainer.loaders import load_datasets, load_dataloaders
from hf_ehr.utils import load_config_from_ckpt, load_tokenizer_from_config, load_model_from_path, load_ckpt


def parse_arguments() -> Namespace:
    """"Parse command-line arguments."""

    parser = ArgumentParser(description='Calculate PPL for a model checkpoint.')
    parser.add_argument('--path_to_ckpt', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default="cuda", help='Device to run validation on')
    parser.add_argument('--split', type=str, default="val", help='Split on which to calculate PPL')
    parser.add_argument('--output_json', type=str, default="./ppl.json", help='Path to output json to save results')
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
                               device: str = "cuda") -> Tuple[float, int]:
    """Calculate perplexity for a single batch."""

    model.eval()
    model.to(device)
    total_log_probs = 0.0
    total_token_count = 0

    with torch.no_grad():
        inputs = batch['tokens']['input_ids'].to(device)
        attention_mask = batch['tokens'].get('attention_mask', None)

        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        outputs = model.model(input_ids=inputs, attention_mask=attention_mask)

        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs[..., 1:].contiguous()

        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

        if attention_mask is not None:
            attention_mask = attention_mask[..., 1:].contiguous()
            log_probs *= attention_mask

        sum_log_probs = log_probs.sum(1)
        count_tokens = attention_mask.sum(1) if attention_mask is not None else log_probs.size(1)

        total_log_probs += sum_log_probs.sum().item()
        total_token_count += count_tokens.sum().item()

    return total_log_probs, total_token_count

def calculate_avg_ppl(model: BaseModel,
                      dataloader: DataLoader,
                      device: str = "cuda") -> Dict[str, Any]:
    """Calculate average perplexity for a dataset split."""

    total_log_probs = 0.0
    total_token_count = 0

    for batch in tqdm(dataloader, total=len(dataloader)):
        batch_log_probs, batch_token_count = calculate_perplexity_batch(model, batch, device)
        total_log_probs += batch_log_probs
        total_token_count += batch_token_count

    avg_log_probs = total_log_probs / total_token_count
    average_perplexity = np.exp(-avg_log_probs)

    results = {
        "average_perplexity": average_perplexity,
        "total_token_count": total_token_count
    }

    return results
    
def save_results(results: Dict[str, Any],
                 path_to_output: str) -> None:
    """Save perplexity results to a file."""

    with open(path_to_output, 'w') as f:
        json.dump(results, f)

def main() -> None:
    args: Namespace = parse_arguments()
    print("Loading model configuration from checkpoint...")
    device: str = args.device
    ckpt: str = load_ckpt(args.path_to_ckpt)
    model: BaseModel = load_model_from_path(args.path_to_ckpt)
    config: Dict[str, Any] = load_config(ckpt)
    tokenizer: CLMBRTokenizer | DescTokenizer | CookbookTokenizer = load_tokenizer_from_config(config)
    datasets: Dict[str, BaseDataset] = load_datasets(config, tokenizer)
    print("Loading dataloaders...")
    dataloaders: Dict[str, DataLoader] = load_dataloaders(config, datasets, tokenizer)
    dataloader: DataLoader = dataloaders[args.split]
    
    print("Calculating average perplexity...")
    start_time = time.time()
    results = calculate_avg_ppl(model, dataloader, device)
    end_time = time.time()
    elapsed_time = end_time - start_time

    results["split"] = args.split
    results["stats"] = {
        "time_taken_seconds": elapsed_time,
        "num_batches": len(dataloader),
    }
    results["model_conifg"] = OmegaConf.to_container(config, resolve=True)

    save_results(results, args.output_path)
    print("Average perplexity: ", results["average_perplexity"])
    print("Stats: ", results["stats"])
    print(f"Results for {args.split} saved under {args.output_path}.")

if __name__ == "__main__":
    main()

# {
#    "main":{
#       "seed":1,
#       "path_to_output_dir":"/share/pi/nigam/migufuen/hf_ehr/cache/runs/gpt-base-1024--clmbr-10-epochs/",
#       "is_force_restart":false
#    },
#    "callbacks":{
#       "early_stopping":{
#          "metric_mode":"min",
#          "patience":3
#       },
#       "model_ checkpointing":{
#          "save_top_k_val_loss":1,
#          "save_most_recent_k":1,
#          "save_most_recent_every_n_train_steps":10000,
#          "every_n_train_steps":30000,
#          "every_n_train_nonPAD_tokens":300000000,
#          "every_n_flops":10000000000000000,
#          "is_ run_eval_on_checkpoint":true
#       }
#    },
#    "data":{
#       "dataset":{
#          "path_to_femr_extract":"/local-scratch/nigam/hf_ehr/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_02_08_extract_v8_no_notes",
#          "is_debug":false,
#          "name":"FEMRDataset"
#       },
#       "dataloader":{
#          "mode":"approx",
#          "batch_size":4,
#          "approx_batch_sampler":{
#             "max_tokens":16384,
#             "bucket_size":100,
#             "is_random_shuffle_across_buckets":true,
#             "is_random_shuffle_within_buckets":true
#          },
#          "n_workers":4,
#          "max_lengt h":1024,
#          "is_truncation_random":true
#       },
#       "tokenizer":{
#          "name":"CLMBRTokenizer",
#          "path_to_config":"/share/pi/nigam/mwornow/hf_ehr/cache/tokenizers/clmbr_v8/tokenizer_config.json"
#       }
#    },
#    "trainer":{
#       "accumulate_grad_batches":4,
#       "g radient_clip_value":1.0,
#       "gradient_clip_algorithm":"norm",
#       "devices":[
#          0
#       ],
#       "distributed_backend":"ddp",
#       "min_epochs":1,
#       "max_epochs":20,
#       "limit_train_batches":"None",
#       "limit_val_batches":"None",
#       "val_check_interval":0.45,
#       " check_val_every_n_epoch":1,
#       "optimizer":{
#          "type":"AdamW",
#          "betas":"(0.9, 0.95)",
#          "weight_decay":0.1,
#          "lr":0.0002
#       },
#       "scheduler":{
#          "num_warmup_steps":40000,
#          "num_decay_steps":4000000,
#          "initial_lr":1e-06,
#          "final_lr":1e-05
#       }
#    },
#    "logging":{
#       "wandb":{
#          "is_wandb":true,
#          "name":"gpt-base-1024--clmbr-10-epochs",
#          "is_force_create_wandb_run_from_scratch":false
#       },
#       "mlflow":{
#          "is_mlflow":false,
#          "name":"None"
#       },
#       "is_log_grad_norm":false,
#       "log_every_n_steps ":1
#    },
#    "model":{
#       "name":"gpt2",
#       "hf_name":"gpt2",
#       "config_kwargs":{
#          "n_positions":1024,
#          "n_layer":12,
#          "n_head":12,
#          "n_embd":768
#       }
#    }
# }