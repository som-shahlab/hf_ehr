import os

V100_BASE_DIR: str = '/local-scratch-nvme/nigam/hf_ehr/'
A100_BASE_DIR: str = '/local-scratch/nigam/hf_ehr/'
GPU_BASE_DIR: str = '/home/hf_ehr/'

PATH_TO_CACHE_DIR: str = '/share/pi/nigam/mwornow/hf_ehr/cache/'
PATH_TO_TOKENIZER_v8_DIR: str = os.path.join(PATH_TO_CACHE_DIR, 'tokenizer_v8/')
PATH_TO_TOKENIZER_v9_DIR: str = os.path.join(PATH_TO_CACHE_DIR, 'tokenizer_v9_lite/')
PATH_TO_RUNS_DIR: str = os.path.join(PATH_TO_CACHE_DIR, 'runs/')