from typing import Dict, List, Optional, Set, Tuple, Union, Any, TypedDict
import torch
from torch.utils.data import Dataset
from femr import Event
import femr.datasets
import os
import numpy as np
from jaxtyping import Float
import json
from transformers import AutoTokenizer
from tqdm import tqdm
import datetime
from hf_ehr.utils import convert_lab_value_to_token_from_quantiles, convert_lab_value_to_token_from_ranges, hash_string_to_uuid
from hf_ehr.config import GPU_BASE_DIR, PATH_TO_DATASET_CACHE_DIR, Code2Detail, SPLIT_TRAIN_CUTOFF, SPLIT_VAL_CUTOFF, SPLIT_SEED
from hf_ehr.data.tokenization import FEMRTokenizer, DescTokenizer

def convert_events_to_tokens(events: List[Event], code_2_detail: Code2Detail, **kwargs) -> List[str]:
    tokens: List[str] = []
    for e in events:
        token: Optional[str] = convert_event_to_token(e, code_2_detail, **kwargs)
        if token is not None:
            tokens.append(token)
    return tokens

def convert_event_to_token(e: Event, code_2_detail: Code2Detail, **kwargs) -> Optional[str]:
    """
        Helper function used in FEMRDataset to convert a FEMR Event `e` to a string `token` based on the dataset's configuration.
        If return `None`, then ignore this event (i.e. has no corresponding token).
    """
    # Parse kwargs
    excluded_vocabs: Set[str] = kwargs.get('excluded_vocabs', {}) or {}
    min_code_count: Optional[int] = kwargs.get('min_code_count', None)
    is_remap_numerical_codes: bool = kwargs.get('is_remap_numerical_codes', False) or False
    is_remap_codes_to_desc: bool = kwargs.get('is_remap_codes_to_desc', False) or False
    is_clmbr: bool = kwargs.get('is_clmbr', False) or False

    # Default the token to just being the literal code
    token: str = e.code # e.g. "LOINC/10230-1"

    # If exclude certain vocabs, then ignore this token if it belongs to one of those vocabs (e.g. "STANFORD_OBS/")            
    if token.split("/")[0].lower() in { x.lower() for x in excluded_vocabs }:
        return None
    
    # Check if code is above min count
    def is_code_above_min_count(token: str, min_code_count: int) -> bool:
        if token in code_2_detail:
            code: str = token
        else:
            code: str = token.split(" || ")[0]
        token_2_count = code_2_detail[code]['token_2_count']
        return sum(token_2_count.values()) >= min_code_count

    if min_code_count:
        if not is_code_above_min_count(token, min_code_count):
            return None
    
    # If CLMBR then do special mapping and continue
    if is_clmbr:
        # If code isn't in CLMBR vocab => ignore
        if e.code not in code_2_detail:
            return None

        # If numerical code...
        if code_2_detail[e.code].get('is_numeric', False) and (
            hasattr(e, 'value') # `e` has a `value`
            and e.value is not None # `value` is not None
            and ( # `value` is numeric
                isinstance(e.value, float)
                or isinstance(e.value, int)
            )
        ):
            # NOTE: CLMBR ignores units, so hardcode all to "None"
            unit: str = "None"
            assert 'unit_2_ranges' in code_2_detail[e.code], f"ERROR - Missing 'unit_2_ranges' for code={e.code} in code_2_detail: {code_2_detail[e.code]}"
            assert unit in code_2_detail[e.code]['unit_2_ranges'], f"ERROR - Missing unit={unit} in 'unit_2_ranges' for code={e.code} in code_2_detail: {code_2_detail[e.code]['unit_2_ranges']}"
            ranges: List[Tuple[float]] = code_2_detail[e.code]['unit_2_ranges'][unit]

            # Determine range for (code, unit, value)
            token = convert_lab_value_to_token_from_ranges(token, unit, e.value, ranges)

            # If code is out of range of CLMBR vocab, return None
            if token.endswith("R0"):
                return None
        
        # If textual code...
        if code_2_detail[e.code].get('is_categorical', False) and (
            hasattr(e, 'value') # `e` has a `value`
            and e.value is not None # `value` is not None
            and e.value != '' # `value` is not blank
            and ( # `value` is textual
                isinstance(e.value, str)
            )
        ):
            assert 'categorical_values' in code_2_detail[e.code], f"ERROR - Missing 'categorical_values' for code={e.code} in code_2_detail: {code_2_detail[e.code]}"
            if e.value in code_2_detail[e.code]['categorical_values']:
                token = f"{e.code} || {e.value}"
            else:
                # If value not in categorical_values, then default to code
                token = f"{e.code}"

        return token

    # First, if remap codes to textual descs => change `token` to textual definition of code
    if is_remap_codes_to_desc:
        # "LOINC/10230-1" => "Left ventricular Ejection fraction"
        token = code_2_detail[e.code]['desc']

    # Second, if remap numerical codes => change numerical codes to bucketed quantiles based on `value`...
    if is_remap_numerical_codes:
        # "LOINC/10230-1" => "LOINC/10230-1 || % (See scan or EMR data for detail) || Q1"
        if (
            hasattr(e, 'value') # `e` has a `value`
            and e.value is not None # `value` is not None
            and ( # `value` is numeric
                isinstance(e.value, float)
                or isinstance(e.value, int)
            )
        ):
            # if we hit a numerical code and need to remap it, follow tokenizer template format
            # to map (code, unit, value) => quantile for (code, unit)
            unit: str = str(e.unit)
            assert 'unit_2_quartiles' in code_2_detail[e.code], f"ERROR - Missing 'unit_2_quartiles' for code={e.code} in code_2_detail: {code_2_detail[e.code]}"
            assert unit in code_2_detail[e.code]['unit_2_quartiles'], f"ERROR - Missing unit={unit} in 'unit_2_quartiles' for code={e.code} in code_2_detail: {code_2_detail[e.code]['unit_2_quartiles']}"
            quantiles: List[float] = code_2_detail[e.code]['unit_2_quartiles'][unit]

            # Determine quantile for (code, unit, value)
            token = convert_lab_value_to_token_from_quantiles(token, unit, e.value, quantiles)
    return token

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
                 excluded_vocabs: Optional[List[str]] = None,
                 is_remap_numerical_codes: bool = False, # if TRUE, then remap numericals to buckets based on quantile of value
                 is_remap_codes_to_desc: bool = False, # if TRUE, then remap all codes to their textual descriptions
                 min_code_count: Optional[int] = None, 
                 is_clmbr: bool = False, # if TRUE, then use CLMBR-style vocab
                 is_debug: bool = False,
                 seed: int = 1):
        assert os.path.exists(path_to_femr_extract), f"{path_to_femr_extract} is not a valid path"
        assert split in ['train', 'val', 'test'], f"{split} not in ['train', 'val', 'test']"
        self.path_to_femr_extract: str = path_to_femr_extract
        self.path_to_code_2_detail: str = path_to_code_2_detail
        self.femr_db = femr.datasets.PatientDatabase(path_to_femr_extract)
        self.split: str = split
        self.sampling_strat: Optional[str] = sampling_strat
        self.sampling_kwargs: Optional[Dict] = sampling_kwargs
        self.excluded_vocabs: Set[str] = excluded_vocabs
        self.min_code_count: Optional[str] = min_code_count
        self.is_debug: bool = is_debug
        self.seed: int = seed
    
        # Code vocab
        self.code_2_detail: Code2Detail = json.load(open(self.path_to_code_2_detail, 'r')) if path_to_code_2_detail is not None else None # type: ignore
                
        # Augmentations
        self.is_remap_numerical_codes: bool = is_remap_numerical_codes
        self.is_remap_codes_to_desc: bool = is_remap_codes_to_desc
        self.is_clmbr: bool = is_clmbr
        if self.is_clmbr:
            assert self.is_remap_numerical_codes, f"ERROR - Cannot have `is_clmbr=True` and `is_remap_numerical_codes=False`"
            assert not self.is_remap_codes_to_desc, f"ERROR - Cannot have `is_clmbr=True` and `is_remap_codes_to_desc=True`"
            assert self.excluded_vocabs == ['STANFORD_OBS'], f"ERROR - If `is_clmbr=True`, then `excluded_vocabs` must be ['STANFORD_OBS']"

        # Sanity check
        if self.is_remap_numerical_codes:
            assert self.code_2_detail is not None, f"self.code_2_detail cannot be NONE if self.is_remap_numerical_codes=True"

        # Pre-calculate canonical splits based on patient ids
        all_pids: np.ndarray = np.array([ pid for pid in self.femr_db ])
        hashed_pids: np.ndarray = np.array([ self.femr_db.compute_split(SPLIT_SEED, pid) for pid in all_pids ])
        self.train_pids: np.ndarray = all_pids[np.where(hashed_pids < SPLIT_TRAIN_CUTOFF)[0]]
        if sampling_strat:
            self.train_pids: np.ndarray = self.get_sampled_pids(self.train_pids)
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
        '''Return all event codes for this patient at `idx` in `self.split`.
            Does any preprocessing necessary for e.g. converting numerical/desc codes.
        '''
        pids: np.ndarray = self.get_pids()
        pid: int = pids[idx]

        # For negative `idx`, we need to unwrap `pid`
        if len(pid.shape) > 0:
            pid = pid[0]

        # Get token for each clinical event in patient timeline
        tokens: List[str] = convert_events_to_tokens(
            self.femr_db[pid].events, 
            code_2_detail=self.code_2_detail, 
            excluded_vocabs=self.excluded_vocabs,
            min_code_count=self.min_code_count,
            is_remap_numerical_codes=self.is_remap_numerical_codes,
            is_remap_codes_to_desc=self.is_remap_codes_to_desc,
            is_clmbr=self.is_clmbr,
        )
        return (pid, tokens)
    
    def get_uuid(self) -> str:
        """Returns unique UUID for this dataset version. Useful for caching files"""
        extract: str = self.path_to_femr_extract.split("/")[-1]
        uuid = f'{extract}-{self.split}'
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
                print(f"The seq_lengths in `{path_to_cache_file}` didn't match this dataset's `uuid` or exact `pids`, so recreating `seq_lengths.json` from scratch now...")
            else:
                print(f"No cache file found at `{path_to_cache_file}` for uuid={self.get_uuid()}`. Generating `seq_lengths.json` now...")

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

def collate_femr_timelines(batch: List[Tuple[int, List[int]]], 
                             tokenizer: Union[FEMRTokenizer, DescTokenizer], 
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
    
    # Tokenizer
    dataset = FEMRDataset(path_to_femr_extract, 
                        path_to_code_2_detail, 
                        split='train', 
                        is_remap_numerical_codes=True, 
                        is_clmbr=True,
                        excluded_vocabs=['STANFORD_OBS'])
    tokenizer = FEMRTokenizer(path_to_code_2_detail, 
                                is_remap_numerical_codes=True,
                                excluded_vocabs=['STANFORD_OBS'])
        

    # Dataset
    train_dataset = FEMRDataset(path_to_femr_extract, path_to_code_2_detail, split='train', is_remap_numerical_codes=True)
    val_dataset = FEMRDataset(path_to_femr_extract, path_to_code_2_detail, split='val', is_remap_numerical_codes=True)
    test_dataset = FEMRDataset(path_to_femr_extract, path_to_code_2_detail, split='test', is_remap_numerical_codes=True)

    print(train_dataset[0])
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

    # # Dataset with numerical lab remapping
    # train_dataset_numerical = FEMRDataset(path_to_femr_extract, path_to_code_2_detail, split='train', is_remap_numerical_codes=True)
    # # Dataset with textual desc code remapping
    # train_dataset_desc = FEMRDataset(path_to_femr_extract, path_to_code_2_detail, split='train', is_remap_codes_to_desc=True)
    
    # # Check numerical codes
    # print("bert tokenizer")
    # print(train_dataset_desc[-1])
    # print(desc_tokenizer(train_dataset_desc[-1:][1])['input_ids'])
    # print(desc_tokenizer.batch_decode(desc_tokenizer(train_dataset_desc[-1:][1])['input_ids']))
    # print("pubmed tokenizer")
    # print(train_dataset_desc[-1])
    # print(pubmed_tokenizer(train_dataset_desc[-1:][1])['input_ids'])
    # print(pubmed_tokenizer.batch_decode(pubmed_tokenizer(train_dataset_desc[-1:][1])['input_ids']))
    # print("biogpt tokenizer")
    # print(train_dataset_desc[-1])
    # print(biogpt_tokenizer(train_dataset_desc[-1:][1])['input_ids'])
    # print(biogpt_tokenizer.batch_decode(biogpt_tokenizer(train_dataset_desc[-1:][1])['input_ids']))
    
    # exit()    
    # train_seq_lengths: List[int] = train_dataset.get_seq_lengths()
    # val_seq_lengths: List[int] = val_dataset.get_seq_lengths()
    # test_seq_lengths: List[int] = test_dataset.get_seq_lengths()
    # assert len(train_seq_lengths) == len(train_dataset)
    # assert len(val_seq_lengths) == len(val_dataset)
    # assert len(test_seq_lengths) == len(test_dataset)

    # # Sanity checking
    # print(train_dataset)
    # print(train_dataset[-1])
    # print(tokenizer(train_dataset[-1:][1])['input_ids'])
    # print(tokenizer.batch_decode(tokenizer(train_dataset[-1:][1])['input_ids']))
    # assert tokenizer(train_dataset[-1:][1])['input_ids'] == [[109803, 8187, 8185, 93995, 91564, 95332, 154435, 155073, 91689, 8184, 155175, 49815, 167230]]
    
    # long_seq = [x for i in range(10) for x in train_dataset[i][1] ]
    # assert len(long_seq) == 2846
    # print(tokenizer(long_seq, is_truncation_random=True, max_length=3, seed=1)['input_ids'])
    # assert tokenizer(long_seq, is_truncation_random=True, max_length=3, seed=1)['input_ids'] == [[150436, 135719, 147624]]
    # assert tokenizer(long_seq, is_truncation_random=True, max_length=3, seed=2)['input_ids'] == [[91787, 97637, 97429]]
    # assert tokenizer(long_seq, is_truncation_random=True, max_length=3, seed=3)['input_ids'] == [[167230, 98027, 98027]]    
