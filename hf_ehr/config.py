import os
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

def rewrite_paths_for_carina_from_config(config: DictConfig) -> DictConfig:
    """Rewrite paths for Carina partitions to use local-scratch directories."""
    if os.environ.get('SLURM_JOB_PARTITION') == 'nigam-v100':
        if not os.path.exists(V100_BASE_DIR):
            os.makedirs(V100_BASE_DIR, exist_ok=True)
            os.system(f'cp -r /share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_08_13_extract_v9 {V100_BASE_DIR}')
            os.system(f'cp /share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9_lite/code_2_detail.json {V100_BASE_DIR}')
            os.system(f'cp /share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9_lite/code_2_count.json {V100_BASE_DIR}')
            os.system(f'cp /share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9/code_2_detail.json {V100_BASE_DIR}')
            os.system(f'cp /share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9/code_2_count.json {V100_BASE_DIR}')
        config.data.tokenizer.path_to_code_2_detail = config.data.tokenizer.path_to_code_2_detail.replace('/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9_lite/', V100_BASE_DIR)
        config.data.tokenizer.path_to_code_2_count = config.data.tokenizer.path_to_code_2_count.replace('/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9_lite/', V100_BASE_DIR)
        config.data.dataset.path_to_femr_extract = config.data.dataset.path_to_femr_extract.replace('/share/pi/nigam/data/', V100_BASE_DIR)
        print(f"Loading data from local-scratch: `{V100_BASE_DIR}`.")
    elif os.environ.get('SLURM_JOB_PARTITION') == 'nigam-a100':
        if not os.path.exists(A100_BASE_DIR):
            # Copy over the cache files
            os.makedirs(A100_BASE_DIR, exist_ok=True)
            os.system(f'cp -r /share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_08_13_extract_v9 {A100_BASE_DIR}')
            os.system(f'cp /share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9_lite/code_2_detail.json {A100_BASE_DIR}')
            os.system(f'cp /share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9_lite/code_2_count.json {A100_BASE_DIR}')
            os.system(f'cp /share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9/code_2_detail.json {A100_BASE_DIR}')
            os.system(f'cp /share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9/code_2_count.json {A100_BASE_DIR}')
        config.data.tokenizer.path_to_code_2_detail = config.data.tokenizer.path_to_code_2_detail.replace('/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9_lite/', A100_BASE_DIR)
        config.data.tokenizer.path_to_code_2_count = config.data.tokenizer.path_to_code_2_count.replace('/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9_lite/', A100_BASE_DIR)
        config.data.dataset.path_to_femr_extract = config.data.dataset.path_to_femr_extract.replace('/share/pi/nigam/data/', A100_BASE_DIR)
        print(f"Loading data from local-scratch: `{A100_BASE_DIR}`.")
    elif os.environ.get('SLURM_JOB_PARTITION') == 'nigam-h100':
        if not os.path.exists(H100_BASE_DIR):
            # Copy over the cache files
            os.makedirs(H100_BASE_DIR, exist_ok=True)
            os.system(f'cp -r /share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_08_13_extract_v9 {H100_BASE_DIR}')
            os.system(f'cp /share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9_lite/code_2_detail.json {H100_BASE_DIR}')
            os.system(f'cp /share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9_lite/code_2_count.json {H100_BASE_DIR}')
            os.system(f'cp /share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9/code_2_detail.json {H100_BASE_DIR}')
            os.system(f'cp /share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9/code_2_count.json {H100_BASE_DIR}')
        config.data.tokenizer.path_to_code_2_detail = config.data.tokenizer.path_to_code_2_detail.replace('/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9_lite/', H100_BASE_DIR)
        config.data.tokenizer.path_to_code_2_count = config.data.tokenizer.path_to_code_2_count.replace('/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9_lite/', H100_BASE_DIR)
        config.data.dataset.path_to_femr_extract = config.data.dataset.path_to_femr_extract.replace('/share/pi/nigam/data/', H100_BASE_DIR)
        print(f"Loading data from local-scratch: `{H100_BASE_DIR}`.")
    elif os.environ.get('SLURM_JOB_PARTITION') == 'gpu':
        if not os.path.exists(GPU_BASE_DIR):
            os.makedirs(GPU_BASE_DIR, exist_ok=True)
            os.system(f'cp -r /share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_08_13_extract_v9 {GPU_BASE_DIR}')
            os.system(f'cp /share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9_lite/code_2_detail.json {GPU_BASE_DIR}')
            os.system(f'cp /share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9_lite/code_2_count.json {GPU_BASE_DIR}')
            os.system(f'cp /share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9/code_2_detail.json {GPU_BASE_DIR}')
            os.system(f'cp /share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9/code_2_count.json {GPU_BASE_DIR}')
        config.data.tokenizer.path_to_code_2_detail = config.data.tokenizer.path_to_code_2_detail.replace('/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9_lite/', GPU_BASE_DIR)
        config.data.tokenizer.path_to_code_2_count = config.data.tokenizer.path_to_code_2_count.replace('/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9_lite/', GPU_BASE_DIR)
        config.data.dataset.path_to_femr_extract = config.data.dataset.path_to_femr_extract.replace('/share/pi/nigam/data/', GPU_BASE_DIR)
        print(f"Loading data from local-scratch: `{GPU_BASE_DIR}`.")
    else:
        print("No local-scratch directory found. Using default `/share/pi/` paths.")
    return config