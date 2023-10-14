import random
from typing import Dict, List, Optional, Union
import pandas as pd
import torch
from torch.utils.data import Dataset
import femr.datasets
import os
import numpy as np
from jaxtyping import Float
from hf_ehr.config import PATH_TO_FEMR_EXTRACT, PATH_TO_TOKENIZER_DIR
import json
from omegaconf import DictConfig

SPLIT_SEED: int = 97
SPLIT_TRAIN_CUTOFF: float = 70
SPLIT_VAL_CUTOFF: float = 85

class FEMRTokenizer():
    def __init__(self, atoi: Dict[str, int]) -> None:
        self.atoi: Dict[str, int] = atoi # [key] = "ICD/10", [value] = 103
        self.itoa: Dict[int, str] = { v: k for k, v in self.atoi.items()} # [key] = 103, [value] = "ICD/10"

        # Special tokens
        if '[PAD]' not in self.atoi.keys():
            raise ValueError("Could not find [PAD] token in self.atoi")
        if '[BOS]' not in self.atoi.keys():
            raise ValueError("Could not find [BOS] token in self.atoi")
        if '[EOS]' not in self.atoi.keys():
            raise ValueError("Could not find [EOS] token in self.atoi")
        if '[UNK]' not in self.atoi.keys():
            raise ValueError("Could not find [UNK] token in self.atoi")
        self.pad_token_id = self.atoi['[PAD]']
        self.bos_token_id = self.atoi['[BOS]']
        self.eos_token_id = self.atoi['[EOS]']
        self.unk_token_id = self.atoi['[UNK]']
        self.special_tokens: List[str] = [ '[PAD]', '[BOS]', '[EOS]', '[UNK]']
        
        # Set attributes
        self.vocab_size: int = len(self.atoi)
    
    def add_special_token(self, token_name: str, token: str):
        max_curr_token: int = max([ int(x) for x in self.itoa.keys() ])
        self.itoa[max_curr_token + 1] = token
        self.atoi[token] = max_curr_token + 1
        setattr(self, f'{token_name}_token_id', self.atoi[token])
        self.special_tokens.append(token)
        self.vocab_size = len(self.atoi)

    def get_vocab(self, is_include_special_tokens: bool = True) -> List[str]:
        if is_include_special_tokens:
            return list(self.atoi.keys())
        else:
            return [ x for x in self.atoi.keys() if x not in self.special_tokens ]

    def get_vocab_tokens(self, is_include_special_tokens: bool = True) -> torch.Tensor:
        if is_include_special_tokens:
            return torch.tensor(self.atoi.values(), dtype=torch.int64)
        else:
            return torch.tensor([ x for x in self.atoi.values() if x not in self.special_tokens ], dtype=torch.int64)

    def tokenize(self, 
                 batch: Union[List[str], List[List[str]]],
                 truncation: bool = False,
                 padding: bool = False,
                 max_length: Optional[int] = None,
                 is_truncation_random: bool = False,
                 add_special_tokens: bool = False,
                 seed: int = 1) -> Dict[str, torch.Tensor]:
        '''Tokenize a batch of patient timelines, where each timeline is a list of event codes.'''
        assert truncation == False or (truncation == True and max_length is not None), "If truncation is True, max_length must be specified"
        
        if not isinstance(batch[0], list):
            # Single timeline - batch to size 1
            batch = [ batch ]
        
        # Tokenize
        tokenized_batch: List[List[int]] = []
        for timeline in batch:
            tokenized_batch.append([ self.atoi[x] for x in timeline ])
        
        # Special tokens
        if add_special_tokens:
            tokenized_batch = [ [self.bos_token_id] + timeline + [self.eos_token_id] for timeline in tokenized_batch ]
        
        # Truncate
        if truncation:
            if is_truncation_random:
                random.seed(seed)
                # Truncate at random positions
                truncated_batch: List[List[int]] = []
                for timeline in tokenized_batch:
                    if len(timeline) > max_length:
                        # Calculate a random start index
                        start_index: int = random.randint(0, len(timeline) - max_length)
                        truncated_batch.append(timeline[start_index:start_index + max_length])
                    else:
                        truncated_batch.append(timeline)
                tokenized_batch = truncated_batch
            else:
                # Truncate on right hand side of sequence
                tokenized_batch = [ timeline[:max_length] for timeline in tokenized_batch ]

        # Pad
        if padding:
            max_batch_length: int = max([ len(timeline) for timeline in tokenized_batch ])
            tokenized_batch = [ timeline + [self.pad_token_id] * (max_batch_length - len(timeline)) for timeline in tokenized_batch ]

        # Input IDs
        input_ids: Float[torch.Tensor, 'B max_length'] = torch.tensor(tokenized_batch)

        # Attention masks
        attention_mask: Float[torch.Tensor, 'B max_length'] = (input_ids != self.pad_token_id).int()
        
        return { "input_ids": input_ids, "attention_mask": attention_mask }

class FEMRDataset(Dataset):
    '''Dataset that returns patients in a FEMR extract.
        Note: Takes 1.5 hrs to loop through all event.code of all 3769353 patients in STARR-OMOP-deid-lite.
    '''
    def __init__(self, 
                 path_to_femr_extract: str, 
                 split: str = 'train', 
                 min_events: Optional[int] = 1):
        assert os.path.exists(path_to_femr_extract), f"{path_to_femr_extract} is not a valid path"
        assert split in ['train', 'val', 'test'], f"{split} not in ['train', 'val', 'test']"
        self.path_to_femr_extract: str = path_to_femr_extract
        self.femr_db = femr.datasets.PatientDatabase(path_to_femr_extract)
        self.split: str = split
        
        # Pre-calculate canonical splits based on patient ids
        all_pids: np.ndarray = np.array([ pid for pid in self.femr_db ])
        hashed_pids: np.ndarray = np.array([ self.femr_db.compute_split(SPLIT_SEED, pid) for pid in all_pids ])
        self.train_pids: np.ndarray = all_pids[np.where(hashed_pids < SPLIT_TRAIN_CUTOFF)[0]]
        self.val_pids: np.ndarray = all_pids[np.where((SPLIT_TRAIN_CUTOFF <= hashed_pids) & (hashed_pids < SPLIT_VAL_CUTOFF))[0]]
        self.test_pids: np.ndarray = all_pids[np.where(hashed_pids >= SPLIT_VAL_CUTOFF)[0]]
        
        # Filter out patients
        # if min_events is not None:
        #     # Filter out patients with timelines shorter than `min_events`
            

        # Confirm disjoint train/val/test
        assert np.intersect1d(self.train_pids, self.val_pids).shape[0] == 0
        assert np.intersect1d(self.train_pids, self.test_pids).shape[0] == 0
        assert np.intersect1d(self.val_pids, self.test_pids).shape[0] == 0

    def __len__(self):
        if self.split == 'train':
            return len(self.train_pids)
        elif self.split == 'val':
            return len(self.val_pids)
        elif self.split == 'test':
            return len(self.test_pids)
        else:
            raise ValueError(f"Invalid split: {self.split}")
    
    def __getitem__(self, idx: int) -> List[int]:
        '''Return all event codes for this patient at `idx` in `self.split`'''
        if self.split == 'train':
            pid = self.train_pids[idx]
        elif self.split == 'val':
            pid = self.val_pids[idx]
        elif self.split == 'test':
            pid = self.test_pids[idx]
        else:
            raise ValueError(f"Invalid split: {self.split}")
        # For negative `idx`, we need to unwrap `pid`
        if len(pid.shape) > 0:
            pid = pid[0]
        return [ e.code for e in self.femr_db[pid].events ]

def collate_femr_timelines(batch: List[List[int]], 
                             tokenizer: FEMRTokenizer, 
                             max_length: int,
                             is_truncation_random: bool = False,
                             seed: int = 1):
    '''Collate function for FEMR timelines
        Truncate or pad to max length in batch.
    '''
    
    # Otherwise, truncate on right hand side of sequence
    tokens: Float[torch.Tensor, 'B max_length'] = tokenizer.tokenize(batch, 
                                                                        truncation=True, 
                                                                        padding=True, 
                                                                        max_length=max_length,
                                                                        is_truncation_random=is_truncation_random,
                                                                        seed=seed, 
                                                                        add_special_tokens=True)
    return tokens


if __name__ == '__main__':
    # Tokenizer
    atoi: Dict[str, int] = json.load(open(os.path.join(PATH_TO_TOKENIZER_DIR, 'code_2_int.json'), 'r'))
    tokenizer = FEMRTokenizer(atoi)
    
    # Dataset
    train_dataset = FEMRDataset(PATH_TO_FEMR_EXTRACT, split='train')
    val_dataset = FEMRDataset(PATH_TO_FEMR_EXTRACT, split='val')
    test_dataset = FEMRDataset(PATH_TO_FEMR_EXTRACT, split='test')
    
    # Stats
    print('train', len(train_dataset))
    print('val', len(val_dataset))
    print('test', len(test_dataset))

    # Sanity checking
    print(train_dataset)
    print(train_dataset[-1])
    print(tokenizer.tokenize(train_dataset[-1:])['input_ids'].tolist())
    assert tokenizer.tokenize(train_dataset[-1:])['input_ids'].tolist() == [[246, 406, 1259, 395, 1195, 239, 911, 1588, 14, 56, 2335, 1292, 179]]