import os

PATH_TO_FEMR_EXTRACT: str = '/share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_08_13_extract_v9_lite'

PATH_TO_CACHE_DIR: str = '/share/pi/nigam/mwornow/hf_ehr/cache/'
PATH_TO_TOKENIZER_DIR: str = os.path.join(PATH_TO_CACHE_DIR, 'tokenizer/')
PATH_TO_RUNS_DIR: str = os.path.join(PATH_TO_CACHE_DIR, 'runs/')