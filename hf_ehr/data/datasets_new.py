from typing import Dict, List, Optional, Set, Tuple, Union, Any, TypedDict
import torch
from torch.utils.data import Dataset
import femr.datasets
import os
import numpy as np
from jaxtyping import Float
import json
from transformers import AutoTokenizer
from tqdm import tqdm
import datetime
from hf_ehr.config import GPU_BASE_DIR, Event, SPLIT_TRAIN_CUTOFF, SPLIT_VAL_CUTOFF, SPLIT_SEED
from hf_ehr.data.tokenization_new import CookbookTokenizer, DescTokenizer

class FEMRDataset(Dataset):
    '''Dataset that returns patients in a FEMR extract.
        Note: Takes 1.5 hrs to loop through all event.code of all 3769353 patients in STARR-OMOP-deid-lite.
    '''
    def __init__(self, 
                 path_to_femr_extract: str, 
                 split: str = 'train',
                 is_debug: bool = False,
                 seed: int = 1):
        assert os.path.exists(path_to_femr_extract), f"{path_to_femr_extract} is not a valid path"
        assert split in ['train', 'val', 'test'], f"{split} not in ['train', 'val', 'test']"
        self.path_to_femr_extract: str = path_to_femr_extract
        self.femr_db = femr.datasets.PatientDatabase(path_to_femr_extract)
        self.split: str = split
        self.is_debug: bool = is_debug
        self.seed: int = seed

        # Pre-calculate canonical splits based on patient ids
        all_pids: np.ndarray = np.array([ pid for pid in self.femr_db ])
        hashed_pids: np.ndarray = np.array([ self.femr_db.compute_split(SPLIT_SEED, pid) for pid in all_pids ])
        self.train_pids: np.ndarray = all_pids[np.where(hashed_pids < SPLIT_TRAIN_CUTOFF)[0]]
        self.val_pids: np.ndarray = all_pids[np.where((SPLIT_TRAIN_CUTOFF <= hashed_pids) & (hashed_pids < SPLIT_VAL_CUTOFF))[0]]
        self.test_pids: np.ndarray = all_pids[np.where(hashed_pids >= SPLIT_VAL_CUTOFF)[0]]
        
        # Confirm disjoint train/val/test
        assert np.intersect1d(self.train_pids, self.val_pids).shape[0] == 0
        assert np.intersect1d(self.train_pids, self.test_pids).shape[0] == 0
        assert np.intersect1d(self.val_pids, self.test_pids).shape[0] == 0

        # If debug, then shrink to 1k patients
        if is_debug:
            self.train_pids = self.train_pids[:1000]
            self.val_pids = self.val_pids[:1000]
            self.test_pids = self.test_pids[:1000]
    
    def get_pids(self) -> np.ndarray:
        """Return patient ids for this split"""
        if self.split == 'train':
            pids = self.train_pids
        elif self.split == 'val':
            pids = self.val_pids
        elif self.split == 'test':
            pids = self.test_pids
        else:
            raise ValueError(f"Invalid split: {self.split}")
        return pids

    def __len__(self) -> int:
        return len(self.get_pids())
    
    def __getitem__(self, idx: int) -> Tuple[int, List[Event]]:
        '''Return all event codes for this patient at `idx` in `self.split`.
            Does any preprocessing necessary for e.g. converting numerical/desc codes.
        '''
        pids: np.ndarray = self.get_pids()
        pid: int = pids[idx]

        # For negative `idx`, we need to unwrap `pid`
        if len(pid.shape) > 0:
            pid = pid[0]

        # Get data for each clinical event in patient timeline
        events: List[Event] = [
            Event(code=e.code, value=e.value, unit=e.unit, start=e.start, end=e.end, omop_table=e.omop_table)
            for e in self.femr_db[pid].events
        ]
        return (pid, events)

def torch_mask_tokens(tokenizer: CookbookTokenizer, inputs: Any, mlm_prob: float, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
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
                             tokenizer: Union[CookbookTokenizer, DescTokenizer], 
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
    
    # Set labels
    if is_mlm:
        # Masked LM
        tokens["input_ids"], tokens["labels"] = torch_mask_tokens(tokenizer, tokens["input_ids"], mlm_prob)
    else:
        # Causal LM
        tokens['labels'] = tokens['input_ids']

    return {
        'patient_ids' : [ x[0] for x in batch ],
        'tokens' : tokens,
    }


if __name__ == '__main__':
    path_to_femr_extract: str = '/share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_02_08_extract_v8_no_notes/'.replace('/share/pi/nigam/data/', GPU_BASE_DIR)
    path_to_code_2_detail: str = '/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v8/code_2_detail.json'
    #path_to_code_2_detail: str = '/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v8/code_2_detail.json'.replace('/share/pi/nigam/mwornow/hf_ehr/cache/', GPU_BASE_DIR)
    
    # Tokenizer
    tokenizer = CookbookTokenizer(path_to_code_2_detail)
    desc_tokenizer = DescTokenizer(AutoTokenizer.from_pretrained("bert-base-uncased"))
    biogpt_tokenizer = DescTokenizer(AutoTokenizer.from_pretrained("microsoft/biogpt"))
    pubmed_tokenizer = DescTokenizer(AutoTokenizer.from_pretrained("stanford-crfm/pubmed_gpt_tokenizer"))
    # breakpoint()
    # Dataset
    train_dataset = FEMRDataset(path_to_femr_extract, path_to_code_2_detail, split='train', is_remap_numerical_codes=False)
    #val_dataset = FEMRDataset(path_to_femr_extract, path_to_code_2_detail, split='val', is_remap_numerical_codes=True)
    #test_dataset = FEMRDataset(path_to_femr_extract, path_to_code_2_detail, split='test', is_remap_numerical_codes=True)

    # Stats
    print('train', len(train_dataset))
    #print('val', len(val_dataset))
    #print('test', len(test_dataset))

    # t1 = time.time()
    # event_count = 0
    # for pid in tqdm(train_dataset.get_pids()[:100000]):
    #     for e in train_dataset.femr_db[pid].events:
    #         event_count += 1
    #         train_dataset.femr_db.get_ontology().get_text_description(e.code)
    # t2 = time.time()
    # print("Time to loop through all events in train_dataset: ", t2 - t1)
    # # Print average time per event
    # print("Average time per patient: ", (t2 - t1) / 100000)
    # print("Average time per event: ", (t2 - t1) / event_count)
    """
    # Dataset with numerical lab remapping
    train_dataset_numerical = FEMRDataset(path_to_femr_extract, path_to_code_2_detail, split='train', is_remap_numerical_codes=True)
    # Dataset with textual desc code remapping
    train_dataset_desc = FEMRDataset(path_to_femr_extract, path_to_code_2_detail, split='train', is_remap_codes_to_desc=True)
    
    # Check numerical codes
    breakpoint()
    print("bert tokenizer")
    print(train_dataset_desc[-1])
    print(desc_tokenizer(train_dataset_desc[-1:][1])['input_ids'])
    print(desc_tokenizer.batch_decode(desc_tokenizer(train_dataset_desc[-1:][1])['input_ids']))
    breakpoint()
    print("pubmed tokenizer")
    print(train_dataset_desc[-1])
    print(pubmed_tokenizer(train_dataset_desc[-1:][1])['input_ids'])
    print(pubmed_tokenizer.batch_decode(pubmed_tokenizer(train_dataset_desc[-1:][1])['input_ids']))
    breakpoint()
    print("biogpt tokenizer")
    print(train_dataset_desc[-1])
    print(biogpt_tokenizer(train_dataset_desc[-1:][1])['input_ids'])
    print(biogpt_tokenizer.batch_decode(biogpt_tokenizer(train_dataset_desc[-1:][1])['input_ids']))
    breakpoint()
    
    exit()    
    train_seq_lengths: List[int] = train_dataset.get_seq_lengths()
    val_seq_lengths: List[int] = val_dataset.get_seq_lengths()
    test_seq_lengths: List[int] = test_dataset.get_seq_lengths()
    assert len(train_seq_lengths) == len(train_dataset)
    assert len(val_seq_lengths) == len(val_dataset)
    assert len(test_seq_lengths) == len(test_dataset)

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
    """
