import os
import shutil
from omegaconf import DictConfig, OmegaConf


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

def copy_if_not_exists(src: str, dest: str) -> None:
    """Copy a file or directory if it does not exist."""
    if not os.path.exists(os.path.join(dest, os.path.basename(src))):
        if os.path.isdir(src):
            print("Copying directory to destination.", src, dest)
            shutil.copytree(src, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dest)

def copy_resources_to_local(base_dir: str) -> None:
    """Copy resources to local-scratch directories."""
    os.makedirs(base_dir, exist_ok=True)
    copy_if_not_exists('/share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_08_13_extract_v9', base_dir)
    copy_if_not_exists('/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9_lite/code_2_detail.json', base_dir)
    copy_if_not_exists('/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9_lite/code_2_count.json', base_dir)
    copy_if_not_exists('/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9/code_2_detail.json', base_dir)
    copy_if_not_exists('/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9/code_2_count.json', base_dir)
    copy_if_not_exists('/share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_02_08_extract_v8_no_notes', base_dir)
    copy_if_not_exists('/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v8/code_2_detail.json', base_dir)
    copy_if_not_exists('/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v8/code_2_count.json', base_dir)

def rewrite_paths_for_carina_from_config(config: DictConfig) -> DictConfig:
    """Rewrite paths for Carina partitions to use local-scratch directories."""
    if os.environ.get('SLURM_JOB_PARTITION') == 'nigam-v100':
        copy_resources_to_local(V100_BASE_DIR)
        config.data.tokenizer.path_to_code_2_detail = config.data.tokenizer.path_to_code_2_detail.replace('/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9_lite/', V100_BASE_DIR)
        config.data.tokenizer.path_to_code_2_detail = config.data.tokenizer.path_to_code_2_detail.replace('/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v8/', V100_BASE_DIR)
        # config.data.tokenizer.path_to_code_2_count = config.data.tokenizer.path_to_code_2_count.replace('/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9_lite/', V100_BASE_DIR)
        config.data.dataset.path_to_femr_extract = config.data.dataset.path_to_femr_extract.replace('/share/pi/nigam/data/', V100_BASE_DIR)
        print(f"Loading data from local-scratch: `{V100_BASE_DIR}`.")
    elif os.environ.get('SLURM_JOB_PARTITION') == 'nigam-a100':
        copy_resources_to_local(A100_BASE_DIR)
        config.data.tokenizer.path_to_code_2_detail = config.data.tokenizer.path_to_code_2_detail.replace('/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9_lite/', A100_BASE_DIR)
        config.data.tokenizer.path_to_code_2_detail = config.data.tokenizer.path_to_code_2_detail.replace('/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v8/', A100_BASE_DIR)
        # config.data.tokenizer.path_to_code_2_count = config.data.tokenizer.path_to_code_2_count.replace('/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9_lite/', A100_BASE_DIR)
        config.data.dataset.path_to_femr_extract = config.data.dataset.path_to_femr_extract.replace('/share/pi/nigam/data/', A100_BASE_DIR)
        print(f"Loading data from local-scratch: `{A100_BASE_DIR}`.")
    elif os.environ.get('SLURM_JOB_PARTITION') == 'nigam-h100':
        copy_resources_to_local(H100_BASE_DIR)
        config.data.tokenizer.path_to_code_2_detail = config.data.tokenizer.path_to_code_2_detail.replace('/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9_lite/', H100_BASE_DIR)
        config.data.tokenizer.path_to_code_2_detail = config.data.tokenizer.path_to_code_2_detail.replace('/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v8/', H100_BASE_DIR)
        # config.data.tokenizer.path_to_code_2_count = config.data.tokenizer.path_to_code_2_count.replace('/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9_lite/', H100_BASE_DIR)
        config.data.dataset.path_to_femr_extract = config.data.dataset.path_to_femr_extract.replace('/share/pi/nigam/data/', H100_BASE_DIR)
        print(f"Loading data from local-scratch: `{H100_BASE_DIR}`.")
    elif os.environ.get('SLURM_JOB_PARTITION') == 'gpu':
        copy_resources_to_local(GPU_BASE_DIR)
        config.data.tokenizer.path_to_code_2_detail = config.data.tokenizer.path_to_code_2_detail.replace('/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9_lite/', GPU_BASE_DIR)
        config.data.tokenizer.path_to_code_2_detail = config.data.tokenizer.path_to_code_2_detail.replace('/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v8/', GPU_BASE_DIR)
        # config.data.tokenizer.path_to_code_2_count = config.data.tokenizer.path_to_code_2_count.replace('/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9_lite/', GPU_BASE_DIR)
        config.data.dataset.path_to_femr_extract = config.data.dataset.path_to_femr_extract.replace('/share/pi/nigam/data/', GPU_BASE_DIR)
        print(f"Loading data from local-scratch: `{GPU_BASE_DIR}`.")
    else:
        print("No local-scratch directory found. Using default `/share/pi/` paths.")
    return config
