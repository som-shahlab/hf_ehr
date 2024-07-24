from tqdm import tqdm
from omegaconf import OmegaConf
import femr.datasets
from hf_ehr.data.datasets import FEMRDataset
from hf_ehr.data.tokenizers import FEMRTokenizer
import numpy as np
from hf_ehr.trainer.loaders import load_datasets, load_dataloaders
import pickle
from hf_ehr.data.datasets import collate_femr_timelines

data = pickle.load(open('/share/pi/nigam/mwornow/hf_ehr/cache/runs/2024-07-04_22-08-58/ckpts/problematic_batch.pkl', 'rb'))
data = pickle.load(open('/share/pi/nigam/mwornow/hf_ehr/cache/runs/2024-07-04_22-08-58/ckpts/problematic_batch_2.pkl', 'rb'))

# print("Loaded pickle")
# db = femr.datasets.PatientDatabase('/share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_02_08_extract_v8_no_notes')
# print("Loaded FEMR")
# tokenizer = FEMRTokenizer('/local-scratch/nigam/users/hf_ehr/tokenizer_v8/code_2_detail.json', 
#                             False, 
#                             ['STANFORD_OBS'], 
#                             None)
# print("Loaded tokenizer")
dataset = FEMRDataset("/local-scratch/nigam/users/hf_ehr/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_02_08_extract_v8_no_notes", 
                      "/local-scratch/nigam/users/hf_ehr/tokenizer_v8/code_2_detail.json", 
                      'train', 
                      None, 
                      None, 
                      ['STANFORD_OBS'], 
                      False, 
                      False, 
                      None, 
                      False, 
                      False)
print("Loaded dataset")

# Get patient from dataset
problem_pid = 30343887 # 30026585 # , 30343971,
assert problem_pid in dataset.get_pids()
idx = np.where(dataset.get_pids() == problem_pid)[0][0]
datapoint = dataset[idx]
print("idx of problem_pid in dataset=", idx)

print("batch:", collate_femr_timelines([ dataset[288000], dataset[288001], dataset[288002], dataset[288003] ], tokenizer, 1024, True, False, 0, 1))

# Load from config
config_path = '/share/pi/nigam/suhana/hf_ehr_repo/hf_ehr/hf_ehr/configs/config.yaml'
config = OmegaConf.load(config_path)
path_to_tokenizer_code_2_detail = '/local-scratch/nigam/users/hf_ehr/tokenizer_v8/code_2_detail.json'
print(f"Loading FEMRTokenizer: `{path_to_tokenizer_code_2_detail}`")
tokenizer = FEMRTokenizer(path_to_tokenizer_code_2_detail)
print(f"Vocab size: `{tokenizer.vocab_size}`")
datasets = load_datasets(config)
dataloaders = load_dataloaders(config, datasets, tokenizer)
print("Loaded dataloaders")

# Fetch the specific batch from the dataloader
train_dataloader = dataloaders['train']
n_all_zero_attns = {}
for i, batch in tqdm(enumerate(train_dataloader), total=batch_index, desc='Looping...'):
    n_all_zero_attn: int = (batch['tokens']['attention_mask'].sum(dim=1) == 0).sum().item()
    if n_all_zero_attn > 0:
        n_all_zero_attns[i] = {
            'n_all_zero_attn' : n_all_zero_attn,
            'batch' : batch,
        }
    if i % 5000 == 0:
        print("len(n_all_zero_attns)=", len(n_all_zero_attns))
        pickle.dump(n_all_zero_attns, open('/share/pi/nigam/mwornow/hf_ehr/cache/runs/2024-07-04_22-08-58/ckpts/problematic_batch_n_all_zero_attns.pkl', 'wb'))
    if i == batch_index:
        print(f"Batch {batch_index} @ idx={i}:\n")
        print(batch)



collate_femr_timelines([ datapoint ], tokenizer, 1024)

# print(datapoint[1])
# # Tokenize patient
# # tokenization = tokenizer([ dataset[72002][1] ])
# tokenization = tokenizer([ datapoint[1] ], is_truncation_random=True, max_length=1024, truncation=True)
# print(tokenization['input_ids'])

# Calculate batch index based on the step
batch_index = 72_000

# Tokenize patient
tokenization = tokenizer([ datasets['train'][72002][1] ], is_truncation_random=True, max_length=1024, truncation=True)
print("TOKENIZATION:", tokenization)
