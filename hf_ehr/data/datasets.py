import random
from typing import Dict, List, Optional, Tuple, Union, Any, TypedDict
import torch
from torch.utils.data import Dataset
import femr.datasets
import os
import numpy as np
from jaxtyping import Float
import json
from transformers import PreTrainedTokenizer
from tqdm import tqdm
import datetime
from omegaconf import OmegaConf
from hf_ehr.config import GPU_BASE_DIR, PATH_TO_DATASET_CACHE_DIR
from hf_ehr.utils import convert_lab_value_to_token

class Detail(TypedDict):
    count: int
    unit_2_quartiles: Optional[List[float]]
    is_numeric: Optional[bool]

class Code2Detail(TypedDict):
    code: Detail

SPLIT_SEED: int = 97
SPLIT_TRAIN_CUTOFF: float = 70
SPLIT_VAL_CUTOFF: float = 85

class FEMRTokenizer(PreTrainedTokenizer):
    def __init__(self, 
                 path_to_code_2_detail: str, 
                 min_code_count: Optional[int] = None) -> None:
        self.code_2_detail: Code2Detail = json.load(open(path_to_code_2_detail, 'r'))
        # Only keep codes with >= `min_code_count` occurrences in our dataset
        codes: List[str] = sorted(list(code_2_detail.keys()))
        if min_code_count is not None:
            codes = [ x for x in codes if self.code_2_detail[x]['count'] >= min_code_count ]

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
                 path_to_code_2_detail: Optional[str],
                 split: str = 'train',
                 sampling_strat: Optional[str] = None,
                 sampling_kwargs: Optional[Dict] = None,
                 is_remap_numerical_codes: bool = False, # if TRUE, then remap numericals to buckets based on quantile of value
                 is_debug: bool = False,
                 seed: int = 1):
        assert os.path.exists(path_to_femr_extract), f"{path_to_femr_extract} is not a valid path"
        assert split in ['train', 'val', 'test'], f"{split} not in ['train', 'val', 'test']"
        self.path_to_femr_extract: str = path_to_femr_extract
        self.femr_db = femr.datasets.PatientDatabase(path_to_femr_extract)
        self.split: str = split
        self.sampling_strat: Optional[str] = sampling_strat
        self.sampling_kwargs: Optional[Dict] = sampling_kwargs
        self.is_debug: bool = is_debug
        self.seed: int = seed
        
        # Augmentations
        self.code_2_detail: Code2Detail = json.load(open(path_to_code_2_detail, 'r')) if path_to_code_2_detail is not None else None # type: ignore
        self.is_remap_numerical_codes: bool = is_remap_numerical_codes
        
        # Sanity check
        if self.is_remap_numerical_codes:
            assert self.code_2_detail is not None, f"self.code_2_detail cannot be NONE if self.is_remap_numerical_codes=True"
        
        # Pre-calculate canonical splits based on patient ids
        all_pids: np.ndarray = np.array([ pid for pid in self.femr_db ])
        hashed_pids: np.ndarray = np.array([ self.femr_db.compute_split(SPLIT_SEED, pid) for pid in all_pids ])
        if split == 'train':
            train_pids = all_pids[np.where(hashed_pids < SPLIT_TRAIN_CUTOFF)[0]]
            if not sampling_strat:
                self.train_pids: np.ndarray = all_pids[np.where(hashed_pids < SPLIT_TRAIN_CUTOFF)[0]]
            else:
                self.train_pids: np.ndarray = self.get_sampled_pids(train_pids)
        elif split == 'val':
            self.val_pids: np.ndarray = all_pids[np.where((SPLIT_TRAIN_CUTOFF <= hashed_pids) & (hashed_pids < SPLIT_VAL_CUTOFF))[0]]
        elif split == 'test':
            self.test_pids: np.ndarray = all_pids[np.where(hashed_pids >= SPLIT_VAL_CUTOFF)[0]]
        else:
            raise ValueError(f"Invalid split: {split}")
        
        # Confirm disjoint train/val/test
        assert np.intersect1d(self.train_pids, self.val_pids).shape[0] == 0
        assert np.intersect1d(self.train_pids, self.test_pids).shape[0] == 0
        assert np.intersect1d(self.val_pids, self.test_pids).shape[0] == 0

        # If debug, then shrink to 1k patients
        if is_debug:
            self.train_pids = self.train_pids[:1000]
            self.val_pids = self.val_pids[:1000]
            self.test_pids = self.test_pids[:1000]

    def get_sampled_pids(self, pids: np.ndarray, is_force_refresh: bool = False) -> np.ndarray:
        """Returns sampled patient_ids based on the sample strategy"""
        # Check if cache exists
        np.random.seed(self.seed)
        path_to_cache_file: str = os.path.join(self.get_path_to_cache_folder(), 'sample_splits.json')
        if not is_force_refresh:
            if os.path.exists(path_to_cache_file):
                data: Dict = json.load(open(path_to_cache_file, 'r'))
                if data['uuid'] == self.get_uuid(): # confirm UUID matches
                    return pids[data['pids']]

        # Generate from scratch
        if self.sampling_strat == 'random':
            # Random sampling -- i.e. select a random X% subset of patients (without replacement)
            assert self.sampling_kwargs.percent is not None, "If sampling_strat is 'random', then you must provide a value for `percent`"
            size: int = len(pids) * self.sampling_kwargs.percent // 100
            indices: np.ndarray = np.random.choice(len(pids), size=size, replace=False)
            pids: np.ndarray = pids[indices]
        elif self.sampling_strat == "stratified":
            # Stratified sampling based on demographics
            assert self.sampling_kwargs.demographic in ['age', 'race', 'sex'], "If sampling_strat is 'stratified', then you must provide a value for `age`, `race`, or `sex`"
            demographics = self._get_demographics_dict(pids)
            indices: np.ndarray = self._get_stratified_indices(demographics)
        else:
            raise ValueError(f"Unsupported sampling strategy: {self.sampling_strat}")
        
        # Save to cache
        os.makedirs(os.path.dirname(path_to_cache_file), exist_ok=True)
        json.dump({ 'uuid' : self.get_uuid(), 'pids' : indices.tolist() }, open(path_to_cache_file, 'w'))
        print("Successfully saved sampled pids to cache: ", path_to_cache_file)
        return pids

    def _get_demographics_dict(self, pids: np.ndarray) -> Dict[str, Any]:
        '''Returns dict of demographics'''
        demographics = {
            'age': {
                'age_20': [],
                'age_40': [],
                'age_60': [],
                'age_80': [],
                'age_plus': []
            },
            'race': {
                'white': [],
                'pacific_islander': [],
                'black': [],
                'asian': [],
                'american_indian': [],
                'unknown': []
            },
            'sex': {
                'male': [],
                'female': []
            }
        }
        for pid in pids:
            if self.sampling_kwargs.demographic == 'age':
                end_age = self.femr_db[pid].events[-1].start
                start_age = self.femr_db[pid].events[0].start
                age = end_age - start_age
                if age <= datetime.timedelta(days=20*365):
                    demographics['age']['age_20'].append(pid)
                elif datetime.timedelta(days=20*365) < age <= datetime.timedelta(days=40*365):  
                    demographics['age']['age_40'].append(pid)
                elif datetime.timedelta(days=40*365) < age < datetime.timedelta(days=60*365):
                    demographics['age']['age_60'].append(pid)
                elif datetime.timedelta(days=60*365) < age < datetime.timedelta(days=80*365):
                    demographics['age']['age_80'].append(pid)
                elif datetime.timedelta(days=80*365) < age:
                    demographics['age']['age_plus'].append(pid)
            elif self.sampling_kwargs.demographic == 'race':
                race_codes = {'Race/5': 'white', 'Race/4': 'pacific_islander', 
                          'Race/3': 'black', 'Race/2': 'asian', 'Race/1': 'american_indian'}
                race = 'unknown'
                for e in self.femr_db[pid].events:
                    if e.code in race_codes:
                        demographics['race'][race_codes[e.code]].append(pid)
                        race = race_codes[e.code]
                        break
                if race == 'unknown':
                    demographics['race']['unknown'].append(pid)
            elif self.sampling_kwargs.demographic == 'sex':
                for e in self.femr_db[pid].events:
                    if e.code == 'Gender/M':
                        demographics['sex']['male'].append(pid)
                        break
                    elif e.code == 'Gender/F':
                        demographics['sex']['female'].append(pid)
                        break
        return demographics
        
    def _get_stratified_indices(self, demographics: Dict) ->  np.ndarray:
        '''Returns stratified patient_ids based on the selected demographic'''
        np.random.seed(self.seed)
        pids = []
        demographic = self.sampling_kwargs.demographic
        min_key = min(demographics[demographic], key=lambda k: len(demographics[demographic][k]))
        for val in demographics[demographic].values():
            sampled_pids = np.random.choice(val, len(demographics[demographic][min_key]), replace=False)
            pids.extend(sampled_pids)
        return np.array(pids)

    def __len__(self) -> int:
        return len(self.get_pids())
    
    def __getitem__(self, idx: int) -> Tuple[int, List[str]]:
        '''Return all event codes for this patient at `idx` in `self.split`'''
        pids: np.ndarray = self.get_pids()
        pid: int = pids[idx]

        # For negative `idx`, we need to unwrap `pid`
        if len(pid.shape) > 0:
            pid = pid[0]
        
        # Get codes
        codes: List[str] = []
        for e in self.femr_db[pid].events:
            if self.is_remap_numerical_codes:
                # if we hit a numerical code and need to remap it, follow tokenizer template format
                # to map (code, unit, value) => quantile for (code, unit)
                if (
                    hasattr(e, 'value') # `e` has a `value`
                    and e.value is not None # `value` is not None
                    and ( # `value` is numeric
                        isinstance(e.value, float)
                        or isinstance(e.value, int)
                    )
                ):
                    unit: str = str(e.unit)
                    quantiles: List[float] = self.code_2_detail[e.code]['unit_2_quartile'][unit]

                    # Determine quantile for (code, unit, value)
                    token: str = convert_lab_value_to_token(e.code, unit, e.value, quantiles)
                    
                    # Remap code => token
                    codes.append(token)
                    continue
            codes.append(e.code)
        return (pid, codes)

    def get_uuid(self) -> str:
        """Returns unique UUID for this dataset version. Useful for caching files"""
        uuid = f'starr_omop_v9-{self.split}'
        if self.sampling_strat is not None:
            uuid += f'-{self.sampling_strat}'
            if self.sampling_strat == 'random':
                uuid += f'-{str(self.sampling_kwargs.percent)}'
            if self.sampling_strat == 'stratified':
                uuid += f'-{self.sampling_kwargs.demographic}'
        if self.is_debug:
            uuid += f'-is_debug'
        return uuid

    def get_path_to_cache_folder(self) -> str:
        """Returns path to cache folder for this dataset (e.g. storing things like sampling split, seq lengths, etc.)"""
        path_to_cache_dir: str = os.path.join(PATH_TO_DATASET_CACHE_DIR, self.get_uuid(), self.split)
        return path_to_cache_dir
    
    def get_pids(self) -> np.ndarray:
        if self.split == 'train':
            pids = self.train_pids
        elif self.split == 'val':
            pids = self.val_pids
        elif self.split == 'test':
            pids = self.test_pids
        else:
            raise ValueError(f"Invalid split: {self.split}")
        return pids

    def get_seq_lengths(self, is_force_refresh: bool = False) -> List[int]:
        """Return a list of sequence lengths for all patients in dataset"""
        pids: np.ndarray = self.get_pids()

        # Check if cache exists (otherwise takes ~10 mins to iterate over 500k patients)
        path_to_cache_file: str = os.path.join(self.get_path_to_cache_folder(), 'seq_lengths.json')
        if not is_force_refresh:
            if os.path.exists(path_to_cache_file):
                print(f"Loading seq_lengths from `{path_to_cache_file}`")
                data = json.load(open(path_to_cache_file, 'r'))
                if data['uuid'] == self.get_uuid(): # confirm UUID matches
                    lengths: List[int] = data['lengths']
                    if len(lengths) == len(pids):
                        return lengths
                print(f"The seq_lengths in `{path_to_cache_file}` didn't match this dataset's `uuid` or exact `pids`, so recreating from scratch")
            else:
                print(f"No cache file found at `{path_to_cache_file}` for uuid={self.get_uuid()}`")

        # Calculate seq lengths
        lengths: List[int] = [ len(self.__getitem__(idx)[1]) for idx in tqdm(range(len(pids)), desc='get_seq_lengths()') ]
        os.makedirs(os.path.dirname(path_to_cache_file), exist_ok=True)
        json.dump({ 'uuid' : self.get_uuid(), 'lengths' : lengths }, open(path_to_cache_file, 'w'))
        return lengths

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


def _generate_dataset(args):
    dataset: FEMRDataset = args[0]
    tokenizer: FEMRTokenizer = args[1]
    path_to_femr_extract: str = args[2]
    n_tokens_per_file: str = args[3]
    pids: np.ndarray = args[4]

    tokens = tokenizer.encode([ e.code for e in self.femr_db[pid] ], truncate=False)
    queue.extend(tokens)
    if len(queue) > 10000:
        np.save(np.array(queue))
        queue = []

    # # Tokenizer
    # code_2_detail: Dict[str, int] = json.load(open(path_to_code_2_detail, 'r'))
    # tokenizer = FEMRTokenizer(code_2_detail)
    
    # FEMR DB
    femr_db = femr.datasets.PatientDatabase(path_to_femr_extract)
    
    for pid in pids.tolist():
        events = [ e.code for e in femr_db[pid].events ]
        tokens += tokenizer.encode(events)['input_ids']

class AllTokensDataset(Dataset):
    '''Dataset that merges all patients into one long sequence of tokens, with each patient sandwiched by a [BOS] and [EOS] token
    '''
    def __init__(self,
                 path_to_femr_extract: str, 
                 dataset: FEMRDataset,
                 is_debug: bool = False):
        assert os.path.exists(path_to_femr_extract), f"{path_to_femr_extract} is not a valid path"
        self.path_to_femr_extract: str = path_to_femr_extract
        self.dataset: FEMRDataset = dataset

    def generate_dataset(self, tokenizer: FEMRTokenizer, n_tokens_per_file: int = 10_000_000, n_procs: int = 10):
        """Default to 10M tokens => 10 * 4 => 40MB files"""
        # Chunk pids
        pids: np.ndarray = self.dataset.get_pids()
        tasks: List[Tuple] = [
            (self.dataset, tokenizer, self.path_to_femr_extract, n_tokens_per_file, pids[chunk_start:chunk_start + len(pids) // n_procs],)
            for chunk_start in range(0, len(pids), len(pids) // n_procs)
        ]
        import multiprocessing
        with multiprocessing.get_context("forkserver").Pool(n_procs) as pool:
            pool.imap_unordered(_generate_dataset, tasks)

    def __len__(self):
        if self.split == 'train':
            return len(self.train_pids)
        elif self.split == 'val':
            return len(self.val_pids)
        elif self.split == 'test':
            return len(self.test_pids)
        else:
            raise ValueError(f"Invalid split: {self.split}")
    
    def __getitem__(self, idx: int) -> Tuple[int, List[str]]:
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

    def get_uuid(self) -> str:
        """Returns unique UUID for this dataset version. Useful for caching files"""
        uuid = f'starr_omop_v9-{self.split}'
        if self.sampling_strat is not None:
            uuid += f'-{self.sampling_strat}'
        if self.is_debug:
            uuid += f'-is_debug'
        return uuid

    def get_path_to_cache_folder(self) -> str:
        """Returns path to cache folder for this dataset (e.g. storing things like sampling split, seq lengths, etc.)"""
        path_to_cache_dir: str = os.path.join(PATH_TO_DATASET_CACHE_DIR, self.get_uuid(), self.split)
        return path_to_cache_dir

    
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
    path_to_femr_extract: str = '/share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_08_13_extract_v9_lite'.replace('/share/pi/nigam/data/', GPU_BASE_DIR)
    path_to_code_2_detail: str = '/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9/code_2_detail.json'.replace('/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v9/', GPU_BASE_DIR)
    
    # Tokenizer
    tokenizer = FEMRTokenizer(path_to_code_2_detail)
    
    # Dataset
    train_dataset = FEMRDataset(path_to_femr_extract, path_to_code_2_detail, split='train', is_remap_numerical_codes=True)
    val_dataset = FEMRDataset(path_to_femr_extract, path_to_code_2_detail, split='val', is_remap_numerical_codes=True)
    test_dataset = FEMRDataset(path_to_femr_extract, path_to_code_2_detail, split='test', is_remap_numerical_codes=True)

    # Stats
    print('train', len(train_dataset))
    print('val', len(val_dataset))
    print('test', len(test_dataset))
    
    # Check numerical codes
    breakpoint()
    print(train_dataset[-1])
    print(tokenizer(train_dataset[-1:][1])['input_ids'])
    print(tokenizer.batch_decode(tokenizer(train_dataset[-1:][1])['input_ids']))

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