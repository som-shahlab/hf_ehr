import os
from typing import TypedDict, Dict, Optional, List
from omegaconf import DictConfig, OmegaConf
from loguru import logger

SPLIT_SEED: int = 97
SPLIT_TRAIN_CUTOFF: float = 70
SPLIT_VAL_CUTOFF: float = 85

H100_BASE_DIR: str = '/local-scratch/nigam/users/hf_ehr/'
A100_BASE_DIR: str = '/local-scratch/nigam/hf_ehr/'
V100_BASE_DIR: str = '/local-scratch-nvme/nigam/hf_ehr/'
GPU_BASE_DIR: str = '/home/hf_ehr/'

PATH_TO_CACHE_DIR: str = '/share/pi/nigam/mwornow/hf_ehr/cache/'
PATH_TO_TOKENIZER_v8_DIR: str = os.path.join(PATH_TO_CACHE_DIR, 'tokenizer_v8/')
PATH_TO_TOKENIZER_v9_DIR: str = os.path.join(PATH_TO_CACHE_DIR, 'tokenizer_v9/')
PATH_TO_TOKENIZER_v9_LITE_DIR: str = os.path.join(PATH_TO_CACHE_DIR, 'tokenizer_v9_lite/')
PATH_TO_RUNS_DIR: str = os.path.join(PATH_TO_CACHE_DIR, 'runs/')
PATH_TO_DATASET_CACHE_DIR = os.path.join(PATH_TO_CACHE_DIR, 'dataset/')
PATH_TO_FEMR_EXTRACT_v9 = '/share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_08_13_extract_v9'
PATH_TO_FEMR_EXTRACT_v8 = '/share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_02_08_extract_v8_no_notes'

class Detail(TypedDict):
    token_2_count: Dict[str, int] # mapping [key] = token, [val] = count of that token
    unit_2_quartiles: Optional[List[float]] # mapping [key] = unit, [val] = list of quartiles
    is_numeric: Optional[bool] # if TRUE, then code is a lab value

class Code2Detail(TypedDict):
    """JSON file named `code_2_detail.json` which is a dict with [key] = code from FEMR, [val] = Detail dict"""
    code: Detail
    
def copy_file(src: str, dest: str, is_overwrite_if_exists: bool = False) -> None:
    """Copy a file or directory if it does not exist."""
    if is_overwrite_if_exists or not os.path.exists(os.path.join(dest, os.path.basename(src))):
        if os.path.isdir(src):
            logger.info(f"Copying directory from `{src}` to `{dest}`.")
            os.system(f'cp -r {src} {dest}')
        else:
            logger.info(f"Copying file from `{src}` to `{dest}`.")
            os.system(f'cp {src} {dest}')

def copy_resources_to_local(base_dir: str, is_overwrite_if_exists: bool = False) -> None:
    """Copy resources to local-scratch directories."""
    os.makedirs(base_dir, exist_ok=True)
    tokenizer_v8_dir = os.path.join(base_dir, 'tokenizer_v8')
    tokenizer_v9_dir = os.path.join(base_dir, 'tokenizer_v9')
    tokenizer_v9_lite_dir = os.path.join(base_dir, 'tokenizer_v9_lite')
    os.makedirs(tokenizer_v8_dir, exist_ok=True)
    os.makedirs(tokenizer_v9_dir, exist_ok=True)
    os.makedirs(tokenizer_v9_lite_dir, exist_ok=True)
    copy_file('/share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_08_13_extract_v9', base_dir, is_overwrite_if_exists=False)
    copy_file(os.path.join(PATH_TO_TOKENIZER_v9_LITE_DIR, 'code_2_detail.json'), tokenizer_v9_lite_dir, is_overwrite_if_exists)
    copy_file(os.path.join(PATH_TO_TOKENIZER_v9_DIR, 'code_2_detail.json'), tokenizer_v9_dir, is_overwrite_if_exists)
    copy_file('/share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_02_08_extract_v8_no_notes', base_dir, is_overwrite_if_exists=False)
    copy_file(os.path.join(PATH_TO_TOKENIZER_v8_DIR, 'code_2_detail.json'), tokenizer_v8_dir, is_overwrite_if_exists)

def get_path_to_code_2_details(config: DictConfig, base_dir: str) -> str:
    """Get the path to code_2_detail.json."""
    path_to_code_2_detail: str = ''
    if "tokenizer_v9_lite" in config.data.tokenizer.path_to_code_2_detail:
        path_to_code_2_detail = config.data.tokenizer.path_to_code_2_detail.replace('/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9_lite/', f"{os.path.join(base_dir, 'tokenizer_v9_lite')}/")
    elif "tokenizer_v9" in config.data.tokenizer.path_to_code_2_detail:
        path_to_code_2_detail = config.data.tokenizer.path_to_code_2_detail.replace('/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9/', f"{os.path.join(base_dir, 'tokenizer_v9')}/")
    elif "tokenizer_v8" in config.data.tokenizer.path_to_code_2_detail:
        path_to_code_2_detail = config.data.tokenizer.path_to_code_2_detail.replace('/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v8/', f"{os.path.join(base_dir, 'tokenizer_v8')}/")
    else:
        raise ValueError("Invalid tokenizer version.")
    return path_to_code_2_detail
    
def rewrite_paths_for_carina_from_config(config: DictConfig) -> DictConfig:
    """Rewrite paths for Carina partitions to use local-scratch directories."""
    if os.environ.get('SLURM_JOB_PARTITION') == 'nigam-v100':
        copy_resources_to_local(V100_BASE_DIR, is_overwrite_if_exists=True)
        config.data.tokenizer.path_to_code_2_detail = get_path_to_code_2_details(config, V100_BASE_DIR)
        config.data.dataset.path_to_femr_extract = config.data.dataset.path_to_femr_extract.replace('/share/pi/nigam/data/', V100_BASE_DIR)
        logger.info(f"Loading data from local-scratch: `{V100_BASE_DIR}`.")
    elif os.environ.get('SLURM_JOB_PARTITION') == 'nigam-a100':
        copy_resources_to_local(A100_BASE_DIR, is_overwrite_if_exists=True)
        config.data.tokenizer.path_to_code_2_detail = get_path_to_code_2_details(config, A100_BASE_DIR)
        config.data.dataset.path_to_femr_extract = config.data.dataset.path_to_femr_extract.replace('/share/pi/nigam/data/', A100_BASE_DIR)
        logger.info(f"Loading data from local-scratch: `{A100_BASE_DIR}`.")
    elif os.environ.get('SLURM_JOB_PARTITION') == 'nigam-h100':
        copy_resources_to_local(H100_BASE_DIR, is_overwrite_if_exists=True)
        config.data.tokenizer.path_to_code_2_detail = get_path_to_code_2_details(config, H100_BASE_DIR)
        config.data.dataset.path_to_femr_extract = config.data.dataset.path_to_femr_extract.replace('/share/pi/nigam/data/', H100_BASE_DIR)
        logger.info(f"Loading data from local-scratch: `{H100_BASE_DIR}`.")
    elif os.environ.get('SLURM_JOB_PARTITION') == 'gpu':
        copy_resources_to_local(GPU_BASE_DIR, is_overwrite_if_exists=True)
        config.data.tokenizer.path_to_code_2_detail = get_path_to_code_2_details(config, GPU_BASE_DIR)
        config.data.dataset.path_to_femr_extract = config.data.dataset.path_to_femr_extract.replace('/share/pi/nigam/data/', GPU_BASE_DIR)
        logger.info(f"Loading data from local-scratch: `{GPU_BASE_DIR}`.")
    else:
        logger.info("No local-scratch directory found. Using default `/share/pi/` paths.")
    return config
