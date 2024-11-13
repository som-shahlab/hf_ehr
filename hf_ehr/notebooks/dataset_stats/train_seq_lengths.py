from hf_ehr.utils import load_config_from_path
from hf_ehr.data.tokenization import CLMBRTokenizer
from hf_ehr.data.datasets import BaseDataset
from torch.utils.data import DataLoader
from hf_ehr.trainer.loaders import load_datasets, load_dataloaders
from omegaconf import OmegaConf
import os
from hf_ehr.config import PATH_TO_FEMR_EXTRACT_v8
from typing import Dict, List
from tqdm import tqdm
import json
 
# dataset_name: str = 'FEMRDataset'
dataset_name: str = 'AllTokensFEMRDataset'

# Load config
config = load_config_from_path('/share/pi/nigam/suhana/hf_ehr/cache/runs_backup/mamba-tiny-16384--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt')
OmegaConf.set_struct(config, False)
config.data.dataset.path_to_femr_extract = PATH_TO_FEMR_EXTRACT_v8

# FEMR
config.data.dataset.name = dataset_name

# Load dataloader
tokenizer = CLMBRTokenizer( config.data.tokenizer.path_to_config )
datasets: Dict[str, BaseDataset] = load_datasets(config, tokenizer)
dataloaders: Dict[str, DataLoader] = load_dataloaders(config, datasets, tokenizer)

# Loop through train dataloader, keeping track of all sequence lengths seen
file_name: str = f'../cache/train_seq_lengths-mamba-16k-{dataset_name}.json'
if os.path.exists(file_name):
    data = json.load(open(file_name, 'r'))
    train_seq_lengths: List[int] = data['train_seq_lengths']
else:
    train_seq_lengths: List[int] = []
    for batch in tqdm(dataloaders['train']):
        lengths = batch['tokens']['attention_mask'].sum(dim=1)
        assert len(lengths) == len(batch['patient_ids'])
        train_seq_lengths.extend(lengths)
    train_seq_lengths = [ x.item() for x in train_seq_lengths ]
    json.dump({ 'train_seq_lengths' : train_seq_lengths, 'dataset_name' : dataset_name }, open(file_name, 'w'))
    print("# of batches:", len(dataloaders['train']))
print("# of seqs:", len(train_seq_lengths))