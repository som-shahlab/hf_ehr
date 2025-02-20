import json
import random
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
from torch.utils.data import DataLoader
import datetime
import traceback
import pandas as pd
from hf_ehr.data.datasets import AllTokensFEMRDataset, FEMRDataset
from hf_ehr.data.tokenization import BaseTokenizer, collate_femr_timelines
from hf_ehr.utils import load_config_from_ckpt, load_tokenizer_from_config, load_model_from_path, load_ckpt
from loguru import logger

def parse_args() -> Namespace:
    """"Parse command-line arguments."""
    parser = ArgumentParser(description='Calculate PPL for a model checkpoint.')
    parser.add_argument('--path_to_ckpt_dir', type=str, required=True, help='Path to model checkpoint directory')
    parser.add_argument('--device', type=str, default="cuda", help='Device to run validation on')
    parser.add_argument('--split', type=str, default="val", help='Split on which to calculate PPL')
    parser.add_argument('--dataset', type=str, default="FEMRDataset", help='Type of dataset -- AllTokensFEMRDataset or AllTokensDataset')
    parser.add_argument('--datasource', type=str, default="starr", help='Source of data')
    parser.add_argument('--stride', type=int, default=32, help='Stride')
    parser.add_argument('--n_patients', type=int, default=20_000, help='# of val patients')
    parser.add_argument('--is_debug', action='store_true', default=False, help='Debug setting')
    parser.add_argument('--is_load_from_config', action='store_true', default=False,  help='If TRUE, load dataset based on config')
    return parser.parse_args()

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
    results = []

    model.eval()
    model.to(device)
    with torch.no_grad():
        for p_idx in tqdm(p_idxs, total=len(p_idxs), desc='eval() | Iterating over patients...'):
            results_for_pid = []
            # Tokenize this patients timeline
            pid, events = dataset[p_idx]
            tokens: Dict[str, Float[torch.Tensor, 'B max_length']] = tokenizer([ events ], 
                                                                                truncation=False, 
                                                                                padding=False, 
                                                                                max_length=max_length,
                                                                                is_truncation_random=False,
                                                                                add_special_tokens=False,
                                                                                return_tensors='pt')
            seq_len: int = tokens['input_ids'].shape[1]
            
            if seq_len < 2:
                # Need at least 2 tokens to calculate PPL
                continue

            # Split timeline into batches of length `max_length` for model to ingest
            prev_end_idx: int = 0
            for start_idx in range(0, seq_len, stride):
                if start_idx + max_length > seq_len and seq_len > max_length:
                    # Shift start_idx to the left to ensure that the last batch is of length `max_length`
                    start_idx = seq_len - max_length
                end_idx: int = min(start_idx + max_length, seq_len)
                trg_len: int = end_idx - prev_end_idx # may be different from stride on last loop
                assert start_idx >= 0, f"Error -- start_idx={start_idx} must be >= 0"
                assert start_idx <= end_idx, f"Error -- start_idx={start_idx} must be <= end_idx={end_idx}"
                assert end_idx <= seq_len, f"Error -- end_idx={end_idx} must be <= seq_len={seq_len}"
                assert trg_len <= max_length, f"Error -- trg_len={trg_len} must be <= max_length={max_length}"

                # Tokens
                input_ids: Float[torch.Tensor, 'B L'] = tokens['input_ids'][:,start_idx:end_idx].to(device)

                # Attention mask
                if 'hyena' in config['model']['name']:
                    attention_mask = None
                    outputs = model.model(input_ids=input_ids)
                else:
                    attention_mask: Float[torch.Tensor, 'B L'] = tokens['attention_mask'][:,start_idx:end_idx].to(device)
                    assert attention_mask.sum() == attention_mask.numel(), "Error -- attention_mask must be all 1's"
                    outputs = model.model(input_ids=input_ids, attention_mask=attention_mask)

                # Run model
                logits: Float[torch.Tensor, 'B L V'] = outputs.logits

                # Calculate log probs
                shift_logits: Float[torch.Tensor, 'B L-1 V'] = logits[:, :-1, :].contiguous()
                shift_labels: Float[torch.Tensor, 'B L-1'] = input_ids[:, 1:].contiguous()
                log_probs: Float[torch.Tensor, 'B L-1 V'] = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                log_probs_for_labels: Float[torch.Tensor, 'B L-1'] = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
                if attention_mask is not None:
                    attention_mask: Float[torch.Tensor, 'B L-1'] = attention_mask[:, :-1].contiguous()
                    log_probs_for_labels *= attention_mask
                
                # Save results depending on trg_len
                log_probs_for_labels = log_probs_for_labels[:,-trg_len:][0]
                shift_labels = shift_labels[:,-trg_len:][0]
                shift_logits = shift_logits[:,-trg_len:][0]
                log_probs = log_probs[:,-trg_len:][0]
                if start_idx == 0:
                    assert log_probs_for_labels.shape[0] == min(seq_len, max_length) - 1, f"Error -- log_probs_for_labels.shape[0]={log_probs_for_labels.shape[0]} must equal min(seq_len, max_length) - 1={min(seq_len, max_length) - 1}"
                    assert trg_len-1 == log_probs_for_labels.shape[0], f"Error -- trg_len-1={trg_len-1} must equal log_probs_for_labels.shape[0]={log_probs_for_labels.shape[0]}"
                        
                results_for_pid += [ {
                    'pid' : pid,
                    'n_events' : len(events),
                    'n_tokens' : seq_len,
                    'token_idx' : prev_end_idx + token_idx + (-1 if start_idx > 0 else 0), # account for shift by 1
                    # what the model is supposed to predict....
                    'label' : shift_labels[token_idx].item(),
                    'label_log_prob' : log_prob,
                    # what the model actually wants to predict....
                    'argmax_label' : shift_logits[token_idx].argmax().item(),
                    'argmax_log_prob' : log_probs[token_idx].max().item(),
                } for token_idx, log_prob in enumerate(log_probs_for_labels.detach().cpu().numpy().tolist()) ]
                # print(f"pid={pid} | n_events={len(events)} | n_tokens={seq_len} | start={start_idx} | end={end_idx} | n_tokens_for_ppl_calc={log_probs_for_labels.shape[0]} | ppl={np.exp(-log_probs_for_labels.detach().cpu().numpy().mean())}")

                prev_end_idx = end_idx
                if end_idx >= seq_len:
                    break

            # Sanity checks
            assert len(results_for_pid) == seq_len - 1, f"Error -- len(results_for_pid)={len(results_for_pid)} must equal seq_len-1={seq_len - 1}"
            assert results_for_pid[0]['token_idx'] == 0, f"Error -- first token_idx={results_for_pid[0]['token_idx']} must equal 0"
            assert results_for_pid[-1]['token_idx'] == seq_len - 2, f"Error -- last token_idx{results_for_pid[-1]['token_idx']} must equal seq_len-2={seq_len - 2}"
            assert len(set([ x['token_idx'] for x in results_for_pid ])) == len(results_for_pid), f"Error -- duplicate token_idx's found in results_for_pid"
            assert [ int(x['token_idx']) for x in results_for_pid ] == list(range(seq_len - 1)), f"Error -- token_idx's not contiguous in results_for_pid"

            # Save results
            results += results_for_pid
            if is_debug and p_idx > 10:
                break

    mean_loss_per_token: float = np.mean([ x['label_log_prob'] for x in results ])
    median_loss_per_token: float = np.median([ x['label_log_prob'] for x in results ])
    std_loss_per_token: float = np.std([ x['label_log_prob'] for x in results ])
    mean_ppl_per_token: float = np.mean([ np.exp(x['label_log_prob']) for x in results ])
    median_ppl_per_token: float = np.median([ np.exp(x['label_log_prob']) for x in results ])
    std_ppl_per_token: float = np.std([ np.exp(x['label_log_prob']) for x in results ])
    n_tokens: int = len(results)
    ppl: float = float(np.exp(-mean_loss_per_token))

    return {
        "mean_loss_per_token" : mean_loss_per_token,
        "median_loss_per_token" : median_loss_per_token,
        "std_loss_per_token" : std_loss_per_token,
        "mean_ppl_per_token" : mean_ppl_per_token,
        "median_ppl_per_token" : median_ppl_per_token,
        "std_ppl_per_token" : std_ppl_per_token,
        "ppl": ppl,
        "n_tokens": n_tokens,
        "n_batches" : len(p_idxs),
        "results" : results,
    }

def map_datasource_to_femr_extract(datasource: str) -> str:
    """Maps data source name (e.g. 'mimic4') to the proper path to FEMR extract"""
    if datasource == 'starr':
        return '/share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_02_08_extract_v8_no_notes'
    elif datasource == 'ehrshot':
        return '/share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/femr/extract'
    elif datasource == 'mimic4':
        return '/share/pi/nigam/data/femr_mimic_4_extract'
    else:
        raise ValueError(f"Unknown datasource: {datasource}")

def add_calcs_to_df(df: pd.DataFrame, datasource: str, tokenizer) -> pd.DataFrame:
    import femr.datasets
    femr_db = femr.datasets.PatientDatabase(map_datasource_to_femr_extract(datasource))
    def safe_get_description(x):
        try:
            return femr_db.get_ontology().get_text_description(x.split(" || ")[0])
        except Exception as e:
            return None  # or return some other default value
    df['label_ppl'] = np.exp(-df['label_log_prob']).astype(float)
    df['label_as_token'] = df['label'].apply(lambda x: tokenizer.idx_2_token[x]).astype(str)
    df['label_as_token_desc'] = df['label_as_token'].apply(lambda x: safe_get_description(x)).astype(str)
    df['argmax_ppl'] = np.exp(-df['argmax_log_prob']).astype(float)
    df['argmax_as_token'] = df['argmax_label'].apply(lambda x: tokenizer.idx_2_token[x]).astype(str)
    df['argmax_as_token_desc'] = df['argmax_as_token'].apply(lambda x: safe_get_description(x)).astype(str)
    df['pid'] = df['pid'].astype(int)
    df['token_idx'] = df['token_idx'].astype(int)
    df['n_events'] = df['n_events'].astype(int)
    df['n_tokens'] = df['n_tokens'].astype(int)
    df['label'] = df['label'].astype(int)
    df['argmax_label'] = df['argmax_label'].astype(int)
    return df

def get_path_to_output_dir(path_to_ckpt_dir: str, split: str, dataset: str) -> str:
    """Create output directory if it does not exist."""
    path_to_output_dir = os.path.abspath(os.path.join(path_to_ckpt_dir, ".." ,"ppl", split, dataset))
    os.makedirs(path_to_output_dir, exist_ok=True)
    return path_to_output_dir

def run_ckpt(path_to_ckpt: str, 
             path_to_output: str, 
             datasource: str,
             dataset_name: str,
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
    logger.info(f"Model config: {config}")
    
    if is_load_from_config:
        # Load dataset/dataloader for this split exactly as loaded during training based on `config`
        OmegaConf.set_struct(config, False)
        config.data.dataset.name = 'FEMRDataset'
        logger.info("Start | Loading dataset")
        start = time.time()
        datasets: Dict[str, Any] = load_datasets(config, tokenizer)
        dataset = datasets[split]
        logger.info(f"Finish | Loading dataset | t={time.time() - start}")
    else:
        # Load dataset/dataloader using fixed settings
        logger.info("Start | Loading dataset")
        start = time.time()
        path_to_femr_extract: str = map_datasource_to_femr_extract(datasource)
        max_length: int = config.data.dataloader.max_length
        is_debug: bool = getattr(config.data.dataset, 'is_debug', False)
        seed: int = config.main.seed
        if dataset_name == 'FEMRDataset':
            dataset = FEMRDataset(path_to_femr_extract, split=split, is_debug=is_debug, seed=seed)
        elif dataset_name == 'AllTokensFEMRDataset':
            dataset = AllTokensFEMRDataset(tokenizer, max_length, path_to_femr_extract, split=split, is_debug=is_debug, seed=seed)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        logger.info(f"Finish | Loading dataset | t={time.time() - start}")
    assert stride <= max_length, f"Error -- stride={stride} must be <= max_length={max_length} of model, otherwise tokens will be skipped"

    # Run eval
    logger.info("Start | Calculating average perplexity")
    start = time.time()
    random.seed(0)
    p_idxs: List[int] = random.sample(range(len(dataset)), n_patients)
    assert len(p_idxs) == n_patients, f"Error -- len(p_idxs)={len(p_idxs)} must equal n_patients={n_patients}"
    raw_results = eval(model, dataset, tokenizer, max_length, p_idxs, stride, config, device, is_debug=is_eval_debug)
    logger.info(f"PPL: {raw_results['ppl']}")
    logger.info(f"Total tokens: {raw_results['n_tokens']}")
    logger.info(f"Finish | Calculating average perplexity | t={time.time() - start}")

    # Save results
    results = {
        'timestamp' : str(datetime.datetime.now()),
        "runtime_seconds": time.time() - initial_start,
        'split' : split,
        'dataset' : 'FEMRDataset',
        'stride' : stride,
        'path_to_ckpt' : path_to_ckpt,
        'config' : OmegaConf.to_container(config, resolve=True),
        'is_debug' : is_eval_debug,
        'results' : {
            'mean_loss_per_token' : raw_results['mean_loss_per_token'],
            'median_loss_per_token' : raw_results['median_loss_per_token'],
            'std_loss_per_token' : raw_results['std_loss_per_token'],
            'mean_ppl_per_token' : raw_results['mean_ppl_per_token'],
            'median_ppl_per_token' : raw_results['median_ppl_per_token'],
            'std_ppl_per_token' : raw_results['std_ppl_per_token'],
            'ppl' : raw_results['ppl'],
            'n_tokens' : raw_results['n_tokens'],
        }
    }
    # Save json dump
    with open(path_to_output + ".json", 'w') as f:
        json.dump(results, f, indent=2)
    logger.warning(f"Saved results to `{path_to_output}.json`")
    # Save .parquet with token-level ppl's
    df = pd.DataFrame(raw_results['results'])
    df = add_calcs_to_df(df, datasource, tokenizer)
    df.to_parquet(path_to_output + '.parquet', index=False)
    logger.warning(f"Saved results to `{path_to_output}.parquet`")

def main() -> None:
    args = parse_args()
    path_to_ckpt_dir: str = args.path_to_ckpt_dir
    path_to_ckpt: str = None
    if not os.path.isdir(path_to_ckpt_dir):
        logger.info(f"Assuming path_to_ckpt_dir is a file: {path_to_ckpt_dir}")
        path_to_ckpt = path_to_ckpt_dir
        path_to_ckpt_dir = os.path.dirname(path_to_ckpt)
    if not 'ckpts' in path_to_ckpt_dir:
        # Go to `ckpts/` subfolder if not directly specified
        path_to_ckpt_dir = os.path.join(path_to_ckpt_dir, 'ckpts/')
    device: str = args.device
    split: str = args.split
    dataset: str = args.dataset
    datasource: str = args.datasource
    stride: int = args.stride
    n_patients: int = args.n_patients
    is_load_from_config: bool = args.is_load_from_config
    is_debug: bool = args.is_debug
    path_to_output_dir: str = get_path_to_output_dir(path_to_ckpt_dir, f"{datasource}/{split}", f"dataset={dataset}-stride={stride}-n_patients={n_patients}-is_config={is_load_from_config}")
    logger.critical(f"Output directory: {path_to_output_dir}")

    # Find all .ckpt files in `path_to_ckpt_dir`
    skipped_ckpts: List[str] = []
    for file in os.listdir(path_to_ckpt_dir):
        if not file.endswith(".ckpt"):
            continue
        
        if path_to_ckpt is not None and file != os.path.basename(path_to_ckpt):
            # User requested a specific ckpt
            continue

        # TODO - remove
        if path_to_ckpt is None and not (
            file.startswith('train-tokens-total_nonPAD')
            and 'ckpt_val=2000000000-persist' in file
        ):
            continue

        # Get paths
        path_to_ckpt: str = os.path.join(path_to_ckpt_dir, file)
        path_to_output: str = os.path.join(path_to_output_dir, f"{file.replace('.ckpt', '')}")

        # Run eval
        logger.info("#"* 50)
        logger.info(f"Start | Processing model ckpt @ `{path_to_ckpt}`")
        try:
            run_ckpt(path_to_ckpt, path_to_output, datasource, dataset, split, n_patients, stride, device, is_load_from_config, is_debug)
        except Exception as e:
            logger.critical(f"Error processing checkpoint @ `{path_to_ckpt}`: {e}")
            traceback.print_exc()
            skipped_ckpts.append(path_to_ckpt)
            raise e
            
        logger.success(f"Finish | Processing model ckpt @ `{path_to_ckpt}`")
        logger.info("#"* 50)
        break
    
    if len(skipped_ckpts) > 0:
        logger.critical(f"Skipped checkpoints: {skipped_ckpts}")

if __name__ == "__main__":
    main()
