from functools import partial
import os
from pathlib import Path
from typing import List, Tuple, Any, Dict, Optional
import torch
import uuid
import hashlib
import yaml


def get_rel_path(path: str) -> str:
    """Get the relative path from the root of the project"""
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), path)

def convert_lab_value_to_token_from_ranges(code: str, unit: str, value: float, ranges: List[Tuple[float, float]], is_tokenize_out_of_range: bool = False) -> Optional[str]:
    # Given a list of ranges (i.e. tuples of [start, end] values), remaps the code to the index in the `ranges` array corresponds
    # to this code's value, i.e. "code" => "{code} || {idx}"
    # If the value doesn't fit in any of the ranges, returns the code itself, i.e. "{code}"
    for idx, (start_val, end_val) in enumerate(ranges):
        if start_val <= value <= end_val:
            return f"{code} || {unit} || R{idx + 1}" # "STANFORD_OBS/123 | mmol | R3"
    # Token is out of range, so decide whether to return `None` or special `R0` code
    if is_tokenize_out_of_range:
        return f"{code} || {unit} || R0" # "STANFORD_OBS/123 | mmol | R0"
    else:
        return None

def convert_lab_value_to_token_from_quantiles(code: str, unit: str, value: float, quantiles: List[float], is_tokenize_out_of_range: bool = False) -> Optional[str]:
    # Note: If we have Q1, Q2, Q3, Q4, then `len(quantiles) == 3` b/c have [0.25, 0.5, 0.75]
    for q_idx, q in enumerate(quantiles):
        if value <= q: 
            return get_lab_value_token_name(code, unit, str(q_idx + 1))
    # Token is out of range, so decide whether to return `None` or special `R0` code
    if is_tokenize_out_of_range:
        return get_lab_value_token_name(code, unit, "0") # "STANFORD_OBS/123 | mmol | Q0"
    else:
        return None

def get_lab_value_token_name(code: str, unit: str, quantile: str) -> str:
    return f"{code} || {unit} || Q{quantile}" # "STANFORD_OBS/123 | mmol | Q4"

def _get_linear_schedule_with_warmup_lr_lambda(current_step: int, 
                                               *, 
                                               num_warmup_steps: int, 
                                               num_decay_steps: int,
                                               initial_lr: float,
                                               peak_lr: float,
                                               final_lr: float) -> float:
    """Note that this needs to return a multiplier on `initial_lr` as set in the optimizer"""
    if current_step < num_warmup_steps:
        # Linear warmup from `initial_lr` to `peak_lr`
        new_lr: float = (peak_lr - initial_lr) / num_warmup_steps * current_step + initial_lr
    elif current_step < num_warmup_steps + num_decay_steps:
        # Linear decay from `peak_lr` to `final_lr`
        new_lr: float = (final_lr - peak_lr) / num_decay_steps * (current_step - num_warmup_steps) + peak_lr
    else:
        # Plateau at `final_lr`
        new_lr: float = final_lr
    multiplier: float = new_lr / peak_lr
    return multiplier

def lr_warmup_with_constant_plateau(optimizer, 
                                    num_warmup_steps: int, 
                                    num_decay_steps: int,
                                    initial_lr: float,
                                    final_lr: float, 
                                    last_epoch: int = -1):
    """
    Create a schedule with a learning rate that decreases linearly from the peak lr set in the optimizer to `final_lr`, after
    a warmup period during which it increases linearly from `initial_lr` to the peak lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_decay_steps (`int`):
            The total number of steps to decay the lr.
        initial_lr (`float`):
            The initial learning rate before the warmup phase
        final_lr (`float`):
            The final learning rate after the warmup and decay phases that we plateau at
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    assert num_warmup_steps > 0, f"num_warmup_steps must be > 0, got {num_warmup_steps}"
    assert num_decay_steps > 0, f"num_decay_steps must be > 0, got {num_decay_steps}"

    peak_lr: float = optimizer.param_groups[0]['lr']
    
    lr_lambda = partial(
        _get_linear_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_decay_steps=num_decay_steps,
        initial_lr=initial_lr,
        peak_lr=peak_lr,
        final_lr=final_lr,
    )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def hash_string_to_uuid(input: Any) -> str:
    """Create a MD5 hash of the stringified input"""
    input_string: str = str(input)
    md5_hash = hashlib.md5(input_string.encode()).hexdigest()
    
    # Generate a UUID from the MD5 hash
    generated_uuid = uuid.UUID(md5_hash)
    
    return str(generated_uuid)

def load_config_from_ckpt(ckpt) -> Dict[str, Any]:
    """Given a model checkpoint (from torch.load()), returns its config"""
    config = ckpt['hyper_parameters']['config']
    def recurse(d: Dict[str, Any]):
        for k, v in d.items():
            if v == 'None':
                d[k] = None
            elif isinstance(v, dict):
                recurse(v)
    recurse(config)
    return config

def load_tokenizer_from_config(config):
    """Load tokenizer from config."""
    from hf_ehr.data.tokenization import CookbookTokenizer, CLMBRTokenizer, DescTokenizer, CEHRTokenizer
    from transformers import AutoTokenizer
    
    # Load config
    name = config.data.tokenizer.name
    path_to_config: str = config.data.tokenizer.path_to_config
    tokenizer__excluded_vocabs: Optional[List[str]] = getattr(config.data.tokenizer, 'excluded_vocabs', [])
    tokenizer__min_code_count: Optional[int] = getattr(config.data.tokenizer, 'min_code_count', None)
    tokenizer__is_remap_numerical_codes: bool = getattr(config.data.tokenizer, 'is_remap_numerical_codes', False)
    tokenizer__desc_emb_tokenizer: bool = getattr(config.data.tokenizer, 'desc_emb_tokenizer', False)

    if name == 'CLMBRTokenizer':
        # CLMBR
        tokenizer = CLMBRTokenizer(path_to_config)
    elif name == 'DescTokenizer':
        # DescTokenizer
        tokenizer = DescTokenizer(path_to_config, AutoTokenizer.from_pretrained(tokenizer__desc_emb_tokenizer))
    elif name == 'CookbookTokenizer':
        # CookbookTokenizer
        tokenizer = CookbookTokenizer(path_to_config, 
                                        excluded_vocabs=tokenizer__excluded_vocabs,
                                        is_remap_numerical_codes=tokenizer__is_remap_numerical_codes,
                                        min_code_count=tokenizer__min_code_count)
    elif name == 'CEHRTokenizer':
        # CEHRTokenizer
        tokenizer = CEHRTokenizer(path_to_config, config.get('data', {}).get('tokenizer', {}).get('metadata', {}))
    return tokenizer

def load_tokenizer_from_path(path_to_ckpt: str):
    """Given a path to a model checkpoint, load the tokenizer."""
    ckpt: Dict[str, Any] = torch.load(path_to_ckpt, map_location='cpu', weights_only=False)
    config: Dict[str, Any] = load_config_from_ckpt(ckpt)
    return load_tokenizer_from_config(config)

def load_config_from_path(path_to_ckpt: str) -> Dict[str, Any]:
    """Given a path to a model checkpoint, load the config."""
    ckpt: Dict[str, Any] = torch.load(path_to_ckpt, map_location='cpu', weights_only=False)
    config: Dict[str, Any] = load_config_from_ckpt(ckpt)
    return config

def load_ckpt(path_to_ckpt: str) -> Dict[str, Any]:
    """Given a path to a model checkpoint, load the checkpoint."""
    ckpt = torch.load(path_to_ckpt, map_location='cpu', weights_only=False)
    return ckpt

def load_model_from_path(path_to_ckpt: str) -> torch.nn.Module:
    """Given a path to a model checkpoint, load the model."""
    from hf_ehr.models.gpt import GPTLanguageModel
    from hf_ehr.models.bert import BERTLanguageModel
    from hf_ehr.models.hyena import HyenaLanguageModel
    from hf_ehr.models.mamba import MambaLanguageModel
    from hf_ehr.models.llama import LlamaLanguageModel
    from hf_ehr.models.t5 import T5LanguageModel

    # Load checkpoint
    ckpt: Dict[str, Any] = load_ckpt(path_to_ckpt)
    config = load_config_from_ckpt(ckpt)
    
    # Load tokenizer
    tokenizer = load_tokenizer_from_path(path_to_ckpt)

    # Determine type of model based on config.model.name
    model_map = {
        'bert': BERTLanguageModel,
        'gpt': GPTLanguageModel,
        'hyena': HyenaLanguageModel,
        'mamba': MambaLanguageModel,
        'llama': LlamaLanguageModel,
        't5': T5LanguageModel
    }
    model_name: str = config['model']['name']
    model_class = next((m for k, m in model_map.items() if k in model_name), None)
    if not model_class: raise ValueError(f"Model `{model_name}` not supported.")

    # Load model
    model = model_class(**ckpt['hyper_parameters'], vocab_size=tokenizer.vocab_size, pad_token_id=tokenizer.pad_token_id)
    model.load_state_dict(ckpt['state_dict'])
    return model

def get_most_recent_ckpt_from_output_dir(path_to_output_dir: str) -> Optional[str]:
    """NOTE: Not used currently, but could be a useful helper function"""
    path_to_ckpts = os.path.join(path_to_output_dir, 'ckpts')
    if os.path.exists(path_to_ckpts):
        # Loop through all ckpt files, choose most recent one
        ckpt_files: List = list(Path(path_to_ckpts).glob('*.ckpt'))
        if ckpt_files is not None and len(ckpt_files) > 0:
            max_ckpt, max_tokens = None, None
            for ckpt_file in ckpt_files:
                path_to_ckpt = os.path.join(path_to_ckpts, ckpt_file)
                ckpt = torch.load(path_to_ckpt, map_location='cpu', weights_only=False)
                if max_tokens is None or ckpt['train_total_tokens_nonPAD'] > max_tokens:
                    max_ckpt = path_to_ckpt
                    max_tokens = ckpt['train_total_tokens_nonPAD']
            assert max_ckpt is not None, f"Error -- max_ckpt is None. Couldn't find most recent ckpt."
            return max_ckpt
    return None

def get_tokenizer_info_from_config_yaml(path_to_config_yaml: str) -> Tuple[str, str]:
    """Given the path to a tokenizer config .yaml file, load the .yaml file and return the path to the tokenizer_config.json and type of tokenizer.
    
    Args:
        path_to_config_yaml (str): The path to the tokenizer config .yaml file. Usually found in `hf_ehr/configs/tokenizer/`.

    Returns:
        Tuple[str, str]: A tuple containing the path to the tokenizer_config.json and the type of tokenizer.
    """
    config_tokenizer: str = yaml.safe_load(open(args.path_to_tokenizer_config, 'r'))
    assert 'data' in config_tokenizer, f"Expected 'data' in config_tokenizer, got {config_tokenizer.keys()}"
    assert 'tokenizer' in config_tokenizer['data'], f"Expected 'tokenizer' in config_tokenizer['data'], got {config_tokenizer['data'].keys()}"
    assert 'path_to_config' in config_tokenizer['data']['tokenizer'], f"Expected 'path_to_config' in config_tokenizer['data']['tokenizer'], got {config_tokenizer['data']['tokenizer'].keys()}"
    assert 'name' in config_tokenizer['data']['tokenizer'], f"Expected 'name' in config_tokenizer['data']['tokenizer'], got {config_tokenizer['data']['tokenizer'].keys()}"
    path_to_tokenizer_config: str = config_tokenizer['data']['tokenizer']['path_to_config']
    tokenizer_name: str = config_tokenizer['data']['tokenizer']['name']
    return path_to_tokenizer_config, tokenizer_name

def get_dataset_info_from_config_yaml(path_to_config_yaml: str) -> Tuple[str, str]:
    """Given the path to a dataset config .yaml file, load the .yaml file and return the type of dataset and the path to the extract.
    
    Args:
        path_to_config_yaml (str): The path to the dataset config .yaml file. Usually found in `hf_ehr/configs/data/`.

    Returns:
        Tuple[str, str]: A tuple containing the path to the extract and the type of dataset.
    """
    config_dataset: str = yaml.safe_load(open(args.path_to_dataset_config, 'r'))
    assert 'data' in config_dataset, f"Expected 'data' in config_dataset, got {config_dataset.keys()}"
    assert 'dataset' in config_dataset['data'], f"Expected 'dataset' in config_dataset['data'], got {config_dataset['data'].keys()}"
    assert 'name' in config_dataset['data']['dataset'], f"Expected 'name' in config_dataset['data']['dataset'], got {config_dataset['data']['dataset'].keys()}"
    dataset_cls: str = config_dataset['data']['dataset']['name']
    
    if dataset_cls == 'FEMRDataset':
        path_to_extract: str = config_dataset['data']['dataset']['path_to_femr_extract']
    elif dataset_cls == 'MEDSDataset':
        path_to_extract: str = config_dataset['data']['dataset']['path_to_meds_reader_extract']
    else:
        raise NotImplementedError(f'Dataset `{dataset_cls}` not yet implemented in `get_dataset_info_from_config_yaml()` function.')
    
    return path_to_extract, dataset_cls