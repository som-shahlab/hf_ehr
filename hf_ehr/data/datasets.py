import random
from typing import Dict, List, Optional, Tuple, Union, Any
import torch
from torch.utils.data import Dataset
import femr.datasets
import os
import numpy as np
from jaxtyping import Float
import json
from hf_ehr.config import GPU_BASE_DIR
from transformers import PreTrainedTokenizer

SPLIT_SEED: int = 97
SPLIT_TRAIN_CUTOFF: float = 70
SPLIT_VAL_CUTOFF: float = 85

class FEMRTokenizer(PreTrainedTokenizer):
    def __init__(self, code_2_count: Dict[str, int], min_code_count: Optional[int] = None) -> None:
        self.code_2_count = code_2_count
        # Only keep codes with >= `min_code_count` occurrences in our dataset
        self.code_2_count = code_2_count
        codes: List[str] = sorted(list(code_2_count.keys()))
        if min_code_count is not None:
            codes = [ x for x in codes if self.code_2_count[x] >= min_code_count ]
            
        # Create vocab
        self.special_tokens = [ '[BOS]', '[EOS]', '[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']
        self.non_special_tokens = codes
        self.vocab = self.special_tokens + self.non_special_tokens
        
        # Map tokens -> idxs
        self.token_2_idx = { x: idx for idx, x in enumerate(self.vocab) }
        self.idx_2_token = { idx: x for idx, x in enumerate(self.vocab) }

        # Create tokenizer
        super().__init__(
            bos_token='[BOS]',
            eos_token='[EOS]',
            unk_token='[UNK]',
            sep_token='[SEP]',
            pad_token='[PAD]',
            cls_token='[CLS]',
            mask_token='[MASK]',
        )
        self.add_tokens(self.non_special_tokens)

    def __call__(self, 
                 batch: Union[List[str], List[List[str]]],
                 is_truncation_random: bool = False,
                 seed: int = 1,
                 **kwargs) -> Dict[str, torch.Tensor]:
        '''Tokenize a batch of patient timelines, where each timeline is a list of event codes.
            We add the ability to truncate seqs at random time points.
            
            Expects as input a list of codes in either the format of:
                A list of codes (List[str])
                A list of lists of codes (List[str])
            NOTE: Must set `is_split_into_words=True` b/c we've already pre-tokenized our inputs (i.e. we're passing in a List of tokens, not a string)
        '''
        if isinstance(batch[0], str):
            # List[str] => List[str]
            batch = [ batch ]

        if is_truncation_random:
            max_length: int = kwargs.get("max_length")
            if not max_length:
                raise ValueError(f"If you specify `is_truncation_random`, then you must also provide a non-None value for `max_length`")

            # Tokenize without truncation
            kwargs.pop('max_length')
            kwargs.pop('truncation')
            tokenized_batch: Dict[str, torch.Tensor] = super().__call__(batch, **kwargs, truncation=None, is_split_into_words=True)
            
            # Truncate at random positions
            random.seed(seed)
            for key in tokenized_batch.keys():
                truncated_batch: List[List[int]] = []
                for timeline in tokenized_batch[key]:
                    if len(timeline) > max_length:
                        # Calculate a random start index
                        start_index: int = random.randint(0, len(timeline) - max_length)
                        new_timeline = timeline[start_index:start_index + max_length]
                        assert new_timeline.shape[0] == max_length, f"Error in truncating by random positions: new_timeline.shape = {new_timeline.shape[0]} != max_length={max_length}"
                        truncated_batch.append(new_timeline)
                    else:
                        truncated_batch.append(timeline)
                if kwargs.get('return_tensors') == 'pt':
                    tokenized_batch[key] = torch.stack(truncated_batch, dim=0)
                else:
                    tokenized_batch[key] = truncated_batch
        else:
            tokenized_batch: Dict[str, torch.Tensor] = super().__call__(batch, **kwargs, is_split_into_words=True)

        return tokenized_batch

    """Mandatory overwrites of base class"""
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        return self.token_2_idx

    def _tokenize(self, text: str, **kwargs):
        """Default to splitting by ' ' since the tokenizer will join together tokens using a space"""
        raise Exception("We shouldn't ever get here (FEMRTokenizer._tokenize()")

    def _convert_token_to_id(self, token: str) -> int:
        return self.token_2_idx[token]

    def _convert_id_to_token(self, index: int) -> str:
        raise self.idx_2_token[index]

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
    
    def __getitem__(self, idx: int) -> Tuple[int, List[int]]:
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
        return (pid, [ e.code for e in self.femr_db[pid].events ])

def torch_mask_tokens(tokenizer: FEMRTokenizer, inputs: Any, mlm_prob: float, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    
    Taken from: https://github.com/huggingface/transformers/blob/09f9f566de83eef1f13ee83b5a1bbeebde5c80c1/src/transformers/data/data_collator.py#L782
    """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for MLM training (with probability `mlm_prob`)
    probability_matrix = torch.full(labels.shape, mlm_prob)
    if special_tokens_mask is None:
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    else:
        special_tokens_mask = special_tokens_mask.bool()

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

def collate_femr_timelines(batch: List[Tuple[int, List[int]]], 
                             tokenizer: FEMRTokenizer, 
                             max_length: int,
                             is_truncation_random: bool = False,
                             is_mlm: bool = False,
                             mlm_prob: float = 0.15,
                             seed: int = 1) -> Dict[str, Any]:
    '''Collate function for FEMR timelines
        Truncate or pad to max length in batch.
    '''

    # Otherwise, truncate on right hand side of sequence
    tokens: Dict[str, Float[torch.Tensor, 'B max_length']] = tokenizer([ x[1] for x in batch ], 
                                                                        truncation=True, 
                                                                        padding=True, 
                                                                        max_length=max_length,
                                                                        is_truncation_random=is_truncation_random,
                                                                        seed=seed, 
                                                                        add_special_tokens=True,
                                                                        return_tensors='pt')
    if is_mlm:
        tokens["input_ids"], tokens["labels"] = torch_mask_tokens(tokenizer, tokens["input_ids"], mlm_prob)
    return {
        'patient_ids' : [ x[0] for x in batch ],
        'tokens' : tokens,
    }


if __name__ == '__main__':

    path_to_femr_extract: str = '/share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_08_13_extract_v9_lite'
    path_to_femr_extract = path_to_femr_extract.replace('/share/pi/nigam/data/', GPU_BASE_DIR)
    path_to_code_2_count: str = '/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9_lite/code_2_count.json'
    path_to_code_2_count = path_to_code_2_count.replace('/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9_lite/', GPU_BASE_DIR)
    
    # Tokenizer
    code_2_count: Dict[str, int] = json.load(open(path_to_code_2_count, 'r'))
    tokenizer = FEMRTokenizer(code_2_count)
    
    # Dataset
    train_dataset = FEMRDataset(path_to_femr_extract, split='train')
    val_dataset = FEMRDataset(path_to_femr_extract, split='val')
    test_dataset = FEMRDataset(path_to_femr_extract, split='test')
    
    # Stats
    print('train', len(train_dataset))
    print('val', len(val_dataset))
    print('test', len(test_dataset))

    # Sanity checking
    print(train_dataset)
    print(train_dataset[-1])
    print(tokenizer(train_dataset[-1:][1])['input_ids'])
    print(tokenizer.batch_decode(tokenizer(train_dataset[-1:][1])['input_ids']))
    assert tokenizer(train_dataset[-1:][1])['input_ids'] == [[109803, 8187, 8185, 93995, 91564, 95332, 154435, 155073, 91689, 8184, 155175, 49815, 167230]]
    
    long_seq = [x for i in range(10) for x in train_dataset[i][1] ]
    assert len(long_seq) == 2846
    print(tokenizer(long_seq, is_truncation_random=True, max_length=3, seed=1)['input_ids'])
    assert tokenizer(long_seq, is_truncation_random=True, max_length=3, seed=1)['input_ids'] == [[150436, 135719, 147624]]
    assert tokenizer(long_seq, is_truncation_random=True, max_length=3, seed=2)['input_ids'] == [[91787, 97637, 97429]]
    assert tokenizer(long_seq, is_truncation_random=True, max_length=3, seed=3)['input_ids'] == [[167230, 98027, 98027]]    