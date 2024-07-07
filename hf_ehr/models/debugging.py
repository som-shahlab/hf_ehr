from tqdm import tqdm
from omegaconf import OmegaConf
import femr.datasets
from hf_ehr.data.datasets import FEMRDataset
from hf_ehr.data.tokenizers import FEMRTokenizer
import numpy as np
from hf_ehr.trainer.loaders import load_datasets, load_dataloaders
import pickle

data = pickle.load(open('/share/pi/nigam/mwornow/hf_ehr/cache/runs/2024-07-04_22-08-58/ckpts/problematic_batch.pkl', 'rb'))

# print("Loaded pickle")
# db = femr.datasets.PatientDatabase('/share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_02_08_extract_v8_no_notes')
# print("Loaded FEMR")
# tokenizer = FEMRTokenizer('/local-scratch/nigam/users/hf_ehr/tokenizer_v8/code_2_detail.json', 
#                             False, 
#                             ['STANFORD_OBS'], 
#                             None)
# print("Loaded tokenizer")
# dataset = FEMRDataset("/local-scratch/nigam/users/hf_ehr/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_02_08_extract_v8_no_notes", 
#                       "/local-scratch/nigam/users/hf_ehr/tokenizer_v8/code_2_detail.json", 
#                       'train', 
#                       None, 
#                       None, 
#                       ['STANFORD_OBS'], 
#                       False, 
#                       False, 
#                       None, 
#                       False, 
#                       False)
# print("Loaded dataset")

# # Get patient from dataset
# problem_pid = 30026585
# assert problem_pid in dataset.get_pids()
# idx = np.where(dataset.get_pids() == 30026585)
# datapoint = dataset[idx]

# print(datapoint[1])
# # Tokenize patient
# # tokenization = tokenizer([ dataset[72002][1] ])
# tokenization = tokenizer([ datapoint[1] ], is_truncation_random=True, max_length=1024, truncation=True)
# print(tokenization['input_ids'])

config_path = '/share/pi/nigam/suhana/hf_ehr_repo/hf_ehr/hf_ehr/configs/config.yaml'
config = OmegaConf.load(config_path)

# Tokenizer
path_to_tokenizer_code_2_detail = '/local-scratch/nigam/users/hf_ehr/tokenizer_v8/code_2_detail.json'

print(f"Loading FEMRTokenizer: `{path_to_tokenizer_code_2_detail}`")
tokenizer = FEMRTokenizer(path_to_tokenizer_code_2_detail)
print(f"Vocab size: `{tokenizer.vocab_size}`")

# Load datasets and dataloaders
datasets = load_datasets(config)
dataloaders = load_dataloaders(config, datasets, tokenizer)
print("Loaded dataloaders")
# Calculate batch index based on the step
batch_index = 72_000


# # Fetch the specific batch from the dataloader
# train_dataloader = dataloaders['train']
# for i, batch in tqdm(enumerate(train_dataloader), total=batch_index, desc='Looping...'):
#     if i == batch_index:
#         print(f"Batch {batch_index} @ idx={i}:\n")
#         print(batch)
#         breakpoint()

# print(f"No batch found at index {batch_index}")


# Tokenize patient
tokenization = tokenizer([ datasets['train'][72200][1] ], is_truncation_random=True, max_length=1024, truncation=True)
print(tokenization['input_ids'])