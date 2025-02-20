import multiprocessing
import datetime
import json
import multiprocessing.managers
import random
from typing import Dict, List, Optional, Set, Tuple, Union, Any, TypedDict
import torch
from transformers import PreTrainedTokenizer, AutoTokenizer
from hf_ehr.config import Event, TokenizerConfigEntry, load_tokenizer_config_from_path, save_tokenizer_config_to_path
import os
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig

class TokenizerSeqLengthPerPatientCache(TypedDict):
    """Typing for `seq_length_per_patient.json` cache file"""
    timestamp: str
    tokenizer_metadata: Dict[str, Any]
    dataset_metadata: Dict[str, Any]
    seq_lengths: List[int]

def filter_tokenizer_config(tokenizer_config: List[TokenizerConfigEntry],
                            excluded_vocabs: Optional[Set[str]] = None,
                            min_code_occurrence_count: Optional[int] = None,
                            keep_n_max_occurrence_codes: Optional[int] = None,
                            **kwargs) -> Tuple[List[TokenizerConfigEntry], List[TokenizerConfigEntry]]:
    """
        Given a set of filters, applies them to the `tokenizer_config`. 
        Returns two lists -- one for valid tokens, one for invalid tokens.
    """
    valid_entries: List[TokenizerConfigEntry] = []
    invalid_entries: List[TokenizerConfigEntry] = []
    for entry in tokenizer_config:
        # Remove tokens from excluded vocabs
        if (
            excluded_vocabs is not None
            and entry.code.split("/")[0].lower() in excluded_vocabs
        ):
            invalid_entries.append(entry.to_token())
            continue
        # Remove tokens with < `min_code_occurrence_count` occurrences in our dataset
        if (
            min_code_occurrence_count is not None
            and entry.get_stat('count_occurrences', None) < min_code_occurrence_count
        ):
            invalid_entries.append(entry.to_token())
            continue
    
        # If we've made it here, then we want to keep this token
        valid_entries.append(entry)
    
    # Keep only the top `keep_n_max_occurrence_codes` tokens, sorted by occurrence count (if specified)
    if keep_n_max_occurrence_codes is not None:
        sorted_entries: List[TokenizerConfigEntry] = sorted(valid_entries, key=lambda x: x.get_stat('count_occurrences', None), reverse=True)
        valid_entries = sorted_entries[:keep_n_max_occurrence_codes]
        invalid_entries = sorted_entries[keep_n_max_occurrence_codes:]

    return valid_entries, invalid_entries

def is_metadata_equal(metadata1: Dict, metadata2: Dict) -> bool:
    """Return TRUE if `metadata1` EXACTLY EQUALS `metadata2`"""
    # Handle special case of paths that get rewritten
    metadata_paths: List[str] = [ 'path_to_femr_extract' ]
    metadata1 = { key: val if not key in metadata_paths else os.path.basename(val) for key, val in metadata1.items() }
    metadata2 = { key: val if not key in metadata_paths else os.path.basename(val) for key, val in metadata2.items() }
    
    # Ignore `is_already_run`
    metadata1.pop('is_already_run', None)
    metadata2.pop('is_already_run', None)

    is_match: bool = True
    for key, val in metadata1.items():
        if key not in metadata2:
            return False
        if metadata2[key] != val:
            return False
    for key, val in metadata2.items():
        if key not in metadata1:
            return False
        if metadata1[key] != val:
            return False
    return is_match

class BaseTokenizer(PreTrainedTokenizer):
    path_to_tokenizer_config: str
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert self.path_to_tokenizer_config is not None, f"ERROR - `self.path_to_tokenizer_config` must be set on `init()`"
        assert hasattr(self, 'metadata'), f"ERROR - `self.metadata` must be set on `init()`"
        # Convert DictConfig to a regular dictionary if necessary
        if isinstance(self.metadata, DictConfig):
            self.metadata = OmegaConf.to_container(self.metadata, resolve=True)
        assert isinstance(self.metadata, dict), f"ERROR - `self.metadata` must be a dict, but got {type(self.metadata)}"
        self.path_to_tokenizer_version_dir: str = self.get_path_to_tokenizer_version_dir() # trigger creation of version folder if it doesn't exist

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Override the default `from_pretrained()` method to allow for custom tokenizer config files"""
        from huggingface_hub import hf_hub_download
        # Download the tokenizer config file from Hugging Face
        path_to_tokenizer_config = hf_hub_download(
            repo_id=pretrained_model_name_or_path,
            filename="tokenizer_config.json" # NOTE: Hardcoded
        )

        # Load the configuration
        with open(path_to_tokenizer_config, "r") as f:
            tokenizer_config = json.load(f)

        # Extract `path_to_tokenizer_config` and `metadata` from the config
        metadata = tokenizer_config.get("metadata", {})
        
        # Initialize the tokenizer
        tokenizer = cls(
            path_to_tokenizer_config=path_to_tokenizer_config,
            **kwargs,
        )
        return tokenizer
    
    ########################################################
    # Tokenization helpers
    ########################################################
    def convert_events_to_tokens(self, events: List[Event], **kwargs) -> List[str]:
        """Provide default implementation where one Event => one token"""
        tokens: List[str] = []
        
        for e in events:
            token: Optional[str] = self.convert_event_to_token(e, **kwargs)
            if token is not None:
                tokens.append(token)
        return tokens
    
    def convert_events_to_tokenized_events(self, events: List[Event], **kwargs) -> List[Event]:
        """Returns all events that DO get mapped to tokens"""
        tokenized_events: List[Event] = []
        for e in events:
            token: Optional[str] = self.convert_event_to_token(e, **kwargs)
            if token is not None:
                tokenized_events.append(e)
        return tokenized_events
    
    def convert_events_to_non_tokenized_events(self, events: List[Event], **kwargs) -> List[Event]:
        """Returns all events that DO NOT get mapped to tokens"""
        non_tokenized_events: List[Event] = []
        for e in events:
            token: Optional[str] = self.convert_event_to_token(e, **kwargs)
            if token is None:
                non_tokenized_events.append(e)
        return non_tokenized_events
    
    def convert_event_to_token(self, e: Event, **kwargs) -> Optional[str]:
        raise NotImplementedError("Must implement `self.convert_event_to_token()` in child class")

    ########################################################
    # Caching / versioning
    ########################################################
    def save(self):
        """Dump vocab for this tokenizer version"""
        json.dump({
            'timestamp' : datetime.datetime.now().isoformat(),
            'metadata' : self.metadata,
            'vocab' : self.get_vocab(),
        }, open(os.path.join(self.path_to_tokenizer_version_dir, 'vocab.json'), 'w'), indent=2)
        save_tokenizer_config_to_path(os.path.join(self.path_to_tokenizer_version_dir, 'tokenizer_config_filtered.json'), self.tokenizer_config)

    def get_path_to_tokenizer_version_dir(self) -> str:
        """
            Example path: /share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v8/versions/2021-08-10_15-00-00/

            The tokenizer can have multiple versions depending on its `self.metadata`. 
            This method returns the path to the folder containing the exact version that matches `self.metadata`.
            If no folder exists for this version, then creates a new folder and return that.
        """
        path_to_tokenizer_dir: str = os.path.dirname(self.path_to_tokenizer_config)
        path_to_versions_dir: str = os.path.join(path_to_tokenizer_dir, 'versions/')
        os.makedirs(path_to_versions_dir, exist_ok=True)
        
        # Find folder corresponding to this version
        for f in os.listdir(path_to_versions_dir):
            if not os.path.isdir(os.path.join(path_to_versions_dir, f)):
                continue
            # Read metadata in `f``
            f_metadata = json.load(open(os.path.join(path_to_versions_dir, f, 'metadata.json'), 'r'))
            
            # If `self.metadata` of this tokenizer exactly matches `metadata.json` in this folder, then return it
            if is_metadata_equal(self.metadata, f_metadata):
                return os.path.join(path_to_versions_dir, f)
        
        # No matching folders found, so create a new one for this version
        path_to_new_folder: str = os.path.join(path_to_versions_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(path_to_new_folder, exist_ok=True)
        json.dump(self.metadata, open(os.path.join(path_to_new_folder, 'metadata.json'), 'w'), indent=2)
        print(f"Creating new folder for this version of the tokenizer at `{path_to_new_folder}` with metadata={self.metadata}")
        return path_to_new_folder
    
    def get_path_to_dataset_dir(self, dataset: 'Dataset') -> str:
        """
            Example path: /share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v8/versions/2021-08-10_15-00-00/datasets/v8/
        
            The tokenizer can have certain dataset-specific properties. 
            We store those in a datasets/ folder within the versions/ folder.
            We make sure that the dataset we retrieve matches the metadata of the argument `dataset`.
        """
        path_to_datasets_dir: str = os.path.join(self.path_to_tokenizer_version_dir, 'datasets/')
        os.makedirs(path_to_datasets_dir, exist_ok=True)
        
        # Find folder corresponding to this version's dataset
        for f in os.listdir(path_to_datasets_dir):
            if not os.path.isdir(os.path.join(path_to_datasets_dir, f)):
                continue
            # Read metadata in `f``
            f_metadata = json.load(open(os.path.join(path_to_datasets_dir, f, 'metadata.json'), 'r'))
            
            # If `self.metadata` of this dataset exactly matches `metadata.json` in this folder, then return it
            if is_metadata_equal(dataset.metadata, f_metadata):
                return os.path.join(path_to_datasets_dir, f)

        # No matching folders found, so create a new one for this version's dataset
        path_to_new_folder: str = os.path.join(path_to_datasets_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(path_to_new_folder, exist_ok=True)
        json.dump(dataset.metadata, open(os.path.join(path_to_new_folder, 'metadata.json'), 'w'), indent=2)
        print(f"Creating new folder for this dataset of this version of the tokenizer at `{path_to_new_folder}` with metadata={dataset.metadata}")
        return path_to_new_folder

    def get_seq_length(self, args: Tuple['Dataset', int, int]) -> List[Tuple[int, int]]:
        """Given a dataset and a range of indices, return the sequence length of each patient in that range"""
        from hf_ehr.data.datasets import FEMRDataset, MEDSDataset
        dataset_metadata, start_idx, end_idx = args # type is: FEMRDataset, int, int
        
        dataset_cls = None
        if dataset_metadata.get('cls') == 'MEDSDataset':
            dataset_cls = MEDSDataset
        else:
            dataset_cls = FEMRDataset

        # remove extraneous keys so that we can init FEMRDataset() without errors
        dataset_metadata = { key: val for key, val in dataset_metadata.items() if key not in [ 'cls', 'tokenizer_metadata', 'max_length' ] }
        
        dataset = dataset_cls(**dataset_metadata)
        results: List[Tuple[int, int]] = []
        for idx in range(start_idx, end_idx):
            events: List[Event] = dataset.__getitem__(idx)[1]
            results.append((idx, len(self.__call__(events)['input_ids'][0])))
        return results

    def get_seq_length_per_patient(self, dataset, n_procs: int = 5, is_force_refresh: bool = False) -> List[int]:
        """
            Fetch the sequence length of every patient in `dataset`, save to cache, and return the list of lengths.
            If cache exists, then load from cache (unless `is_force_refresh` is set to TRUE).
            If cache doesn't exist or `is_force_refresh` is TRUE, then calculate the lengths in parallel. 
                For CLMBRTokenizer, `n_procs=5`, this takes ~5 hrs for 2.5M patients (train dataset)
                For DescTokenizer, `n_procs=5`, this takes ~11 hrs for 2.5M patients (train dataset)
        """
        # Check if cache exists
        path_to_dataset_dir: str = self.get_path_to_dataset_dir(dataset)
        path_to_cache_file: str = os.path.join(path_to_dataset_dir, 'seq_length_per_patient.json')

        if not is_force_refresh:
            # If NOT force refresh, try to load from cache
            if os.path.exists(path_to_cache_file):
                print(f"Loading `seq_length_per_patient.json` from `{path_to_cache_file}` for split=`{dataset.split}`")
                data: TokenizerSeqLengthPerPatientCache = json.load(open(path_to_cache_file, 'r'))
                seq_lengths: List[int] = data['seq_lengths']
                if (
                    len(seq_lengths) == dataset.get_n_patients() 
                    and is_metadata_equal(self.metadata, data.get('tokenizer_metadata'))
                    and is_metadata_equal(dataset.metadata, data.get('dataset_metadata'))
                ):
                    return seq_lengths
                print(f"The # of `seq_lengths` in `{path_to_cache_file}` didn't match this dataset's length ({len(seq_lengths)} != {dataset.get_n_patients()}) or the `metadata` differed for tokenizer ({self.metadata} != {data['tokenizer_metadata']}) or dataset ({dataset.metadata} != {data['dataset_metadata']}), so recreating `seq_length_per_patient.json` from scratch now...")
            else:
                print(f"No `seq_length_per_patient.json` found at `{path_to_cache_file}` for split=`{dataset.split}`. Generating `seq_length_per_patient.json` now...")

        # Calculate seq lengths in parallel
        n_procs = 1 # TODO - remove
        if n_procs == 1:
            tasks: List[Tuple] = [(dataset.metadata, start, min(dataset.get_n_patients(), start + 1),) for start in range(0, dataset.get_n_patients(), 1) ]
            results: List[List[Tuple[int,int]]] = [ self.get_seq_length(t) for t in tqdm(tasks, total=len(tasks), desc=f"tokenizer.get_seq_length_per_patient() | n_procs={n_procs}") ]
        else:
            chunk_size: int = 5_000
            tasks: List[Tuple] = [(dataset.metadata, start, min(dataset.get_n_patients(), start + chunk_size),) for start in range(0, dataset.get_n_patients(), chunk_size) ]
            with multiprocessing.Pool(processes=n_procs) as pool:
                results: List[List[Tuple[int,int]]] = list(tqdm(pool.imap_unordered(self.get_seq_length, tasks), total=len(tasks), desc=f"tokenizer.get_seq_length_per_patient() | n_procs={n_procs}"))
        flattened_results: List[Tuple[int, int]] = [ x for sublist in results for x in sublist ] # type: ignore
        seq_lengths: List[int] = [ x[1] for x in sorted(flattened_results, key=lambda x: x[0]) ]
        json.dump({ 
            'timestamp' : datetime.datetime.now().isoformat(), 
            'tokenizer_metadata' : self.metadata, 
            'dataset_metadata' : dataset.metadata, 
            'seq_lengths' : seq_lengths 
        }, open(path_to_cache_file, 'w'), indent=2)
        return seq_lengths

class BaseCodeTokenizer(BaseTokenizer):

    def __init__(self) -> None:
        # Create vocab
        self.special_tokens = [ '[BOS]', '[EOS]', '[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']
        self.vocab = self.special_tokens + self.non_special_tokens

        # Map tokens -> idxs
        self.token_2_idx: Dict[str, int] = { x: idx for idx, x in enumerate(self.vocab) }
        self.idx_2_token: Dict[int, str] = { idx: x for idx, x in enumerate(self.vocab) }

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
        self.clean_up_tokenization_spaces = False # to avoid HuggingFace deprecation warning
    
    def __call__(self, 
                 batch_of_events: Union[List[Event], List[List[Event]]],
                 is_truncation_random: bool = False,
                 seed: int = 1,
                 **kwargs) -> Dict[str, torch.Tensor]:
        """Tokenize a batch of patient timelines, where each timeline is a list of event codes.
            We add the ability to truncate seqs at random time points.
            
            Expects as input a list of Events

            NOTE: Must set `is_split_into_words=True` b/c we've already pre-tokenized our inputs (i.e. we're passing in a List of tokens, not a string)
        """
        if not isinstance(batch_of_events[0], list):
            # List[Event] => List[List[Event]]
            batch_of_events = [ batch_of_events ] # type: ignore
        
        # First, convert all Events => ProtoTokens
        batch: List[List[str]] = [ self.convert_events_to_tokens(x) for x in batch_of_events ]

        # Second, add special tokens (if applicable)
        if kwargs.get("add_special_tokens", False):
            batch = [ [ self.cls_token, self.bos_token ] + x + [ self.eos_token ] for x in batch ]

        # NOTE: When `is_split_into_words=True`, then __call__() will fail when a list of tokens has length 0.
        # Replace those lists with length 0 with a list of length 1 (i.e. [self.pad_token_id])
        batch = [ x if len(x) > 0 else [ self.pad_token ] for x in batch ]

        # Third, tokenize the batch
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
            start_idxs: List[int] = []
            for timeline in tokenized_batch['input_ids']:
                length: int = (timeline != self.pad_token_id).sum() # count of non-PAD tokens
                if length > max_length:
                    # Calculate a random start index
                    start_index: int = random.randint(0, length - max_length)
                    start_idxs.append(start_index)
                else:
                    start_idxs.append(0)
                    
            for key in tokenized_batch.keys():
                truncated_batch: List[List[int]] = []
                for idx, timeline in enumerate(tokenized_batch[key]):
                    new_timeline = timeline[start_idxs[idx]:start_idxs[idx] + max_length]
                    assert new_timeline.shape[0] <= max_length, f"Error in truncating by random positions: new_timeline.shape = {new_timeline.shape[0]} !<= max_length={max_length}"
                    truncated_batch.append(new_timeline)
                if kwargs.get('return_tensors') == 'pt':
                    tokenized_batch[key] = torch.stack(truncated_batch, dim=0)
                else:
                    tokenized_batch[key] = truncated_batch
        else:
            try:
                tokenized_batch: Dict[str, torch.Tensor] = super().__call__(batch, **kwargs, is_split_into_words=True)
            except Exception as e:
                print("Exception in BaseCodeTokenizer.__call__():", e)
                raise e

        return tokenized_batch

    """Mandatory overwrites of base class"""
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        return self.token_2_idx
    
    def tokenize(self, text: str, **kwargs) -> int:
        """Here, `text` will be a single code (e.g. "LOINC/13"), so just map it back to itself"""
        return [ text ]

    def _tokenize(self, text: str, **kwargs) -> int:
        """Here, `text` will be a single code (e.g. "LOINC/13"), so just map it back to itself"""
        return [ text ]

    def _convert_token_to_id(self, token: str) -> int:
        return self.token_2_idx[token]

    def _convert_id_to_token(self, index: int) -> str:
        raise self.idx_2_token[index]

class CookbookTokenizer(BaseCodeTokenizer):
    """
        Settings:
            is_remap_numerical_codes_to_quantiles: bool
                - If TRUE, then remap numericals to buckets based on quantile of value
            excluded_vocabs: Optional[List[str]]
                - List of vocabs to exclude from the tokenizer. Determined by the first part of the code before the '/' (e.g. "STANFORD_OBS" in "STANFORD_OBS/1234")
            min_code_occurrence_count: Optional[int]
                - Only keep tokens with >= `min_code_occurrence_count` total occurrences in our dataset
    """
    def __init__(self, 
                 path_to_tokenizer_config: str, 
                 metadata: Optional[Dict[str, Any]] = None) -> None:
        self.path_to_tokenizer_config: str = path_to_tokenizer_config
        self.tokenizer_config: List[TokenizerConfigEntry] = load_tokenizer_config_from_path(path_to_tokenizer_config)
        
        # Set metadata
        self.metadata: Dict[str, Any] = {} if metadata is None else dict(metadata)
        self.metadata['cls'] = 'CookbookTokenizer'

        # Fetches ATT metadata - By default, it'll be done in CEHR-GPT style
        self.is_add_visit_start: bool = metadata.get('is_add_visit_start', True)
        self.is_add_visit_end: bool = metadata.get('is_add_visit_end', True)
        self.is_add_day_att: bool = metadata.get('is_add_day_att', True)
        self.is_add_day_week_month_att: bool = metadata.get('is_add_day_week_month_att', False)

        # Pop the flags from metadata after initializing them to avoid recreating the config
        for key in ['is_add_visit_start', 'is_add_visit_end', 'is_add_day_att', 'is_add_day_week_month_att']:
            if key in self.metadata:
                self.metadata.pop(key)
        
        #  # Initialize special tokens
        # # TODO -- prepend all these attributes with 'token_' for readibility
        # self.visit_start = "[VISIT START]"
        # self.visit_end = "[VISIT END]"
        # self.day_atts_cehr_gpt = [f"[DAY {i}]" for i in range(1, 1081)]
        # self.long_att_cehr_gpt = "[LONG TERM]"
        # self.day_atts_cehr_bert = [f"[DAY {i}]" for i in range(1, 7)]
        # self.week_atts = [f"[WEEK {i}]" for i in range(1, 4)]
        # self.month_atts = [f"[MONTH {i}]" for i in range(1, 12)]
        # self.long_att_cehr_bert = "[LONG TERM]"

        # # Add special tokens to the vocabulary
        # self.special_tokens.extend(self.day_atts_cehr_gpt)
        # self.special_tokens.append(self.long_att_cehr_gpt)
        # self.special_tokens.extend(self.day_atts_cehr_bert)
        # self.special_tokens.extend(self.week_atts)
        # self.special_tokens.extend(self.month_atts)
        # self.special_tokens.append(self.long_att_cehr_bert)
        # self.special_tokens.extend([self.visit_start, self.visit_end])

        # Metadata
        self.is_remap_numerical_codes_to_quantiles: bool = metadata.get('is_remap_numerical_codes_to_quantiles', False)
        self.excluded_vocabs: Optional[Set[str]] = { x.lower() for x in metadata.get('excluded_vocabs', {}) } if metadata.get('excluded_vocabs', {}) else None # type: ignore
        self.min_code_occurrence_count: Optional[int] = metadata.get('min_code_occurrence_count', None)
        self.keep_n_max_occurrence_codes: Optional[int] = metadata.get('keep_n_max_occurrence_codes', None)

        # Apply filtering
        self.tokenizer_config, self.excluded_tokens = filter_tokenizer_config(self.tokenizer_config, 
                                                                              self.excluded_vocabs, 
                                                                              self.min_code_occurrence_count,
                                                                              self.keep_n_max_occurrence_codes)
        # Tokens
        self.code_2_token = {} # [key] = token; [val] = { 'type' : str, 'tokenization' : dict, 'token' : str }
        self.non_special_tokens: List[str] = []
        
        # Preprocess tokenizer config for quick access
        for entry in self.tokenizer_config: # NOTE: Takes ~10 seconds for 1.5M tokens
            if entry.code not in self.code_2_token: self.code_2_token[entry.code] = {}
            if entry.type not in self.code_2_token[entry.code]: self.code_2_token[entry.code][entry.type] = []
            self.code_2_token[entry.code][entry.type].append({
                'tokenization': entry.tokenization,
                'token' : entry.to_token(),
            })
            self.non_special_tokens.append(entry.to_token())
        
        # Create tokenizer
        super().__init__()

    def convert_event_to_token(self, e: Event, **kwargs) -> Optional[str]:
        """NOTE: This is basically the same as the CLMBR tokenizer's version."""
        event_code = e.code
        # If code isn't in vocab => ignore
        if event_code not in self.code_2_token:
            return None
        
        event_value = e.value
        # If numerical code...
        if (
            'numerical_range' in self.code_2_token[event_code] # `numerical_range` is a valid type for this code
            and event_value is not None # `value` is not None
            and ( # `value` is numeric
                isinstance(event_value, float)
                or isinstance(event_value, int)
            )
        ):
            for token_range in self.code_2_token[event_code]['numerical_range']:
                assert 'token' in token_range, f"ERROR - Missing 'token' for code={e.code},type=numerical_range in self.code_2_token: {self.code_2_token[e.code]['numerical_range']}"
                assert 'tokenization' in token_range, f"ERROR - Missing 'tokenization' for code={e.code},type=numerical_range in self.code_2_token: {self.code_2_token[e.code]['numerical_range']}"
                token: str = token_range['token']
                unit: str = token_range['tokenization']['unit']
                range_start: float = token_range['tokenization']['range_start']
                range_end: float = token_range['tokenization']['range_end']
                if range_start <= event_value <= range_end and e.unit == unit:
                    return token
            return None

        # If textual code...
        if (
            'categorical' in self.code_2_token[event_code] # `categorical` is a valid type for this code
            and event_value is not None # `value` is not None
            and event_value != '' # `value` is not blank
            and ( # `value` is textual
                isinstance(event_value, str)
            )
        ):
            for categorical_value in self.code_2_token[event_code]['categorical']:
                assert 'token' in categorical_value, f"ERROR - Missing 'token' for code={event_code},type=categorical in self.code_2_token: {self.code_2_token[event_code]['categorical']}"
                assert 'tokenization' in categorical_value, f"ERROR - Missing 'tokenization' for code={event_code},type=categorical in self.code_2_token: {self.code_2_token[event_code]['categorical']}"
                if event_value in categorical_value['tokenization']['categories']:
                    token: str = categorical_value['token']
                    return token
            return None

        # If just vanilla code...
        if (
            'code' in self.code_2_token[event_code] # `code` is a valid type for this code
        ):
            token: str = self.code_2_token[event_code]['code'][0]['token']
            return token

        return None
    
    def convert_events_to_tokens(self, events: List[Event], **kwargs) -> List[str]:
        tokens: List[str] = []
        current_visit_end: Optional[datetime.datetime] = None # track the end time of the currently active visit
        previous_visit_end: Optional[datetime.datetime] = None # track the end time of the immediately preceding visit

        for e in events:

            # Check if we need to add a visit end token
            if current_visit_end is not None and (
                e.start > current_visit_end # If we have [VISIT A = { TOKEN 1, TOKEN 2 }] [TOKEN 3], then add a visit end token before [TOKEN 3]
                or "Visit" in e.code # If we have [VISIT A = { TOKEN 1, TOKEN 2 }] [VISIT B = { TOKEN 3, TOKEN 4 }], then add a visit end token after [VISIT A]
            ):
                # This token occurs after the currently active visit ends, so end it (if exists)
                if self.is_add_visit_end:
                    tokens.append(self.visit_end)
                current_visit_end = None

            # Check if the event is a visit
            if "Visit" in e.code:
                    
                # Add ATT Tokens, if applicable
                # This will be inserted between the prior visit and the current visit
                if previous_visit_end is not None:
                    interval: float = (e.start - previous_visit_end).days # Time (in days) between this visit's start and the immediately preceding visit's end
                    assert interval >= 0, f"Interval has value = {interval} but should always be positive, but fails on {e}."
                    
                    if self.is_add_day_att:
                        if interval <= 1080:
                            att = self.day_atts_cehr_gpt[interval - 1]
                        else:
                            att = self.long_att_cehr_gpt
                        tokens.append(att)
                    elif self.is_add_day_week_month_att:
                        if interval < 7:
                            att = self.day_atts_cehr_bert[interval - 1]
                        elif 7 <= interval < 30:
                            att = self.week_atts[(interval // 7) - 1]
                        elif 30 <= interval < 360:
                            att = self.month_atts[(interval // 30) - 1]
                        else:
                            att = self.long_att_cehr_bert
                        tokens.append(att)

                # Add visit start token, if applicable
                # if self.is_add_visit_start:
                #     tokens.append(self.visit_start)
            
                # Add token itself
                token = self.convert_event_to_token(e, **kwargs)
                if token:
                    tokens.append(token)

                # Keep track of this visit's end
                current_visit_end = e.end
                previous_visit_end = e.end
            else:
                token = self.convert_event_to_token(e, **kwargs)
                if token:
                    tokens.append(token)
        
        return tokens

class CLMBRTokenizer(BaseCodeTokenizer):
    def __init__(self, path_to_tokenizer_config: str) -> None:
        self.path_to_tokenizer_config: str = path_to_tokenizer_config
        self.tokenizer_config: List[TokenizerConfigEntry] = load_tokenizer_config_from_path(path_to_tokenizer_config)
        
        # Set metadata        
        self.metadata: Dict[str, Any] = {}
        self.metadata['cls'] = 'CLMBRTokenizer'

        # Preprocess tokenizer config for quick access
        self.code_2_token = {} # [key] = token; [val] = { 'type' : str, 'tokenization' : dict, 'token' : str }
        self.non_special_tokens: List[str] = []
        for entry in self.tokenizer_config:
            if entry.code not in self.code_2_token: self.code_2_token[entry.code] = {}
            if entry.type not in self.code_2_token[entry.code]: self.code_2_token[entry.code][entry.type] = []
            self.code_2_token[entry.code][entry.type].append({
                'tokenization': entry.tokenization,
                'token' : entry.to_token(),
            })
            self.non_special_tokens.append(entry.to_token())

        # assert len(self.non_special_tokens) == 39811, f"ERROR - Expected 39811 self.non_special_tokens, but got {len(self.non_special_tokens)}"

        # Create tokenizer
        super().__init__()

    def convert_event_to_token(self, e: Event, **kwargs) -> Optional[str]:
        # If code isn't in CLMBR vocab => ignore
        if e.code not in self.code_2_token:
            return None
        
        # If numerical code...
        if (
            'numerical_range' in self.code_2_token[e.code] # `numerical_range` is a valid type for this code
            and e.value is not None # `value` is not None
            and ( # `value` is numeric
                isinstance(e.value, float)
                or isinstance(e.value, int)
            )
        ):
            # NOTE: CLMBR ignores units
            for token_range in self.code_2_token[e.code]['numerical_range']:
                assert 'token' in token_range, f"ERROR - Missing 'token' for code={e.code},type=numerical_range in self.code_2_token: {self.code_2_token[e.code]['numerical_range']}"
                assert 'tokenization' in token_range, f"ERROR - Missing 'tokenization' for code={e.code},type=numerical_range in self.code_2_token: {self.code_2_token[e.code]['numerical_range']}"
                token: str = token_range['token']
                range_start: float = token_range['tokenization']['range_start']
                range_end: float = token_range['tokenization']['range_end']
                if range_start <= e.value <= range_end:
                    return token
            return None

        # If textual code...
        if (
            'categorical' in self.code_2_token[e.code] # `categorical` is a valid type for this code
            and e.value is not None # `value` is not None
            and e.value != '' # `value` is not blank
            and ( # `value` is textual
                isinstance(e.value, str)
            )
        ):
            for categorical_value in self.code_2_token[e.code]['categorical']:
                assert 'token' in categorical_value, f"ERROR - Missing 'token' for code={e.code},type=categorical in self.code_2_token: {self.code_2_token[e.code]['categorical']}"
                assert 'tokenization' in categorical_value, f"ERROR - Missing 'tokenization' for code={e.code},type=categorical in self.code_2_token: {self.code_2_token[e.code]['categorical']}"
                if e.value in categorical_value['tokenization']['categories']:
                    token: str = categorical_value['token']
                    return token
            return None

        # If just vanilla code...
        if (
            'code' in self.code_2_token[e.code] # `code` is a valid type for this code
        ):
            token: str = self.code_2_token[e.code]['code'][0]['token']
            return token

        return None


class CEHRTokenizer(BaseCodeTokenizer):
    def __init__(self, path_to_tokenizer_config: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.path_to_tokenizer_config: str = path_to_tokenizer_config
        self.tokenizer_config: List[TokenizerConfigEntry] = load_tokenizer_config_from_path(path_to_tokenizer_config)
        
        # Set metadata        
        self.metadata: Dict[str, Any] = {} if metadata is None else dict(metadata)
        self.metadata['cls'] = 'CEHRTokenizer'

        # Fetches ATT metadata - By default, it'll be done in CEHR-GPT style
        self.is_add_visit_start: bool = self.metadata.get('is_add_visit_start', True)
        self.is_add_visit_end: bool = self.metadata.get('is_add_visit_end', True)
        self.is_add_day_att: bool = self.metadata.get('is_add_day_att', True)
        self.is_add_day_week_month_att: bool = self.metadata.get('is_add_day_week_month_att', False)
        # Initialize special tokens for visits and time intervals (ATT tokens)
        self.visit_start = "[VISIT START]"
        self.visit_end = "[VISIT END]"
        self.day_atts_cehr_gpt = [f"[DAY {i}]" for i in range(1, 1081)]  # Days 1 to 1080
        self.long_att_cehr_gpt = "[LONG TERM]"
        self.day_atts_cehr_bert = [f"[DAY {i}]" for i in range(1, 7)]  # Days 1 to 6
        self.week_atts = [f"[WEEK {i}]" for i in range(1, 5)]  # Weeks 1 to 4
        self.month_atts = [f"[MONTH {i}]" for i in range(1, 13)]  # Months 1 to 12
        self.long_att_cehr_bert = "[LONG TERM]"

        # Add special tokens to the tokenizer
        self.special_tokens = [
            self.visit_start, self.visit_end, self.long_att_cehr_gpt, self.long_att_cehr_bert
        ] + self.day_atts_cehr_gpt + self.day_atts_cehr_bert + self.week_atts + self.month_atts

        # Preprocess tokenizer config for quick access
        self.code_2_token = {} # [key] = token; [val] = { 'type' : str, 'tokenization' : dict, 'token' : str }
        self.non_special_tokens: List[str] = []
        for entry in self.tokenizer_config:
            if entry.code not in self.code_2_token: self.code_2_token[entry.code] = {}
            if entry.type not in self.code_2_token[entry.code]: self.code_2_token[entry.code][entry.type] = []
            self.code_2_token[entry.code][entry.type].append({
                'tokenization': entry.tokenization,
                'token' : entry.to_token(),
            })
            self.non_special_tokens.append(entry.to_token())

        # assert len(self.non_special_tokens) == 39811, f"ERROR - Expected 39811 self.non_special_tokens, but got {len(self.non_special_tokens)}"

        # Create tokenizer
        super().__init__()

    def convert_event_to_token(self, e: Event, **kwargs) -> Optional[str]:
        # If code isn't in CLMBR vocab => ignore
        if e.code not in self.code_2_token:
            return None
        
        # If numerical code...
        if (
            'numerical_range' in self.code_2_token[e.code] # `numerical_range` is a valid type for this code
            and e.value is not None # `value` is not None
            and ( # `value` is numeric
                isinstance(e.value, float)
                or isinstance(e.value, int)
            )
        ):
            # NOTE: CLMBR ignores units
            for token_range in self.code_2_token[e.code]['numerical_range']:
                assert 'token' in token_range, f"ERROR - Missing 'token' for code={e.code},type=numerical_range in self.code_2_token: {self.code_2_token[e.code]['numerical_range']}"
                assert 'tokenization' in token_range, f"ERROR - Missing 'tokenization' for code={e.code},type=numerical_range in self.code_2_token: {self.code_2_token[e.code]['numerical_range']}"
                token: str = token_range['token']
                range_start: float = token_range['tokenization']['range_start']
                range_end: float = token_range['tokenization']['range_end']
                if range_start <= e.value <= range_end:
                    return token
            return None

        # If textual code...
        if (
            'categorical' in self.code_2_token[e.code] # `categorical` is a valid type for this code
            and e.value is not None # `value` is not None
            and e.value != '' # `value` is not blank
            and ( # `value` is textual
                isinstance(e.value, str)
            )
        ):
            for categorical_value in self.code_2_token[e.code]['categorical']:
                assert 'token' in categorical_value, f"ERROR - Missing 'token' for code={e.code},type=categorical in self.code_2_token: {self.code_2_token[e.code]['categorical']}"
                assert 'tokenization' in categorical_value, f"ERROR - Missing 'tokenization' for code={e.code},type=categorical in self.code_2_token: {self.code_2_token[e.code]['categorical']}"
                if e.value in categorical_value['tokenization']['categories']:
                    token: str = categorical_value['token']
                    return token
            return None

        # If just vanilla code...
        if (
            'code' in self.code_2_token[e.code] # `code` is a valid type for this code
        ):
            token: str = self.code_2_token[e.code]['code'][0]['token']
            return token

        return None
    
    def convert_events_to_tokens(self, events: List[Event]) -> List[str]:
        """
        Convert a list of events into a list of tokens, inserting ATT tokens based on time intervals between visits.
        """
        tokens = []
        current_visit_end = None
        previous_visit_end = None
        for e in events:
            # Add visit end token if the event is after the previous visit's end
            if current_visit_end is not None and e.start > current_visit_end:
                if self.is_add_visit_end:
                    tokens.append(self.visit_end)
                current_visit_end = None

            # Handling visits and adding ATT tokens
            if "Visit" in e.code:
                # Ignore visits that last 0 seconds
                if e.start == e.end:
                    # Visit is a point event, so ignore
                    continue
                
                # If another visit is currently open (but somehow overlaps with this visit), end the previous visit
                if current_visit_end is not None:
                    if self.is_add_visit_end:
                        tokens.append(self.visit_end)
                    previous_visit_end = current_visit_end
                    current_visit_end = None
                
                # Add ATT tokens between visits based on time intervals
                if previous_visit_end is not None:
                    interval = (e.start - previous_visit_end).days
                    if interval >= 0:
                        if self.is_add_day_att:
                            if interval <= 1080:
                                tokens.append(self.day_atts_cehr_gpt[interval - 1])
                            else:
                                tokens.append(self.long_att_cehr_gpt)
                        elif self.is_add_day_week_month_att:
                            if interval < 7:
                                tokens.append(self.day_atts_cehr_bert[interval - 1])
                            elif 7 <= interval < 30:
                                tokens.append(self.week_atts[(interval // 7) - 1])
                            elif 30 <= interval < 360:
                                tokens.append(self.month_atts[(interval // 30) - 1])
                            else:
                                tokens.append(self.long_att_cehr_bert)

                # Add visit start token
                if self.is_add_visit_start:
                    tokens.append(self.visit_start)

                # Convert visit event to token and add it
                token = self.convert_event_to_token(e)
                if token:
                    tokens.append(token)

                current_visit_end = e.end
                previous_visit_end = e.end
            else:
                # Convert non-visit events to tokens
                token = self.convert_event_to_token(e)
                if token:
                    tokens.append(token)
                    
        # Close any visit that isn't closed by end of timeline
        if current_visit_end is not None:
            if self.is_add_visit_end:
                tokens.append(self.visit_end)
        return tokens

class DescTokenizer(BaseTokenizer):
    """Converts codes => textual descriptions, then tokenizes using a normal text tokenizer (e.g. BERT)
    """
    def __init__(self, 
                 path_to_tokenizer_config: str, 
                 metadata: Optional[Dict[str, Any]] = None) -> None:
        assert metadata is not None, f"ERROR - `metadata` must be provided, but got {metadata}"
        assert 'desc_emb_tokenizer' in metadata, f"ERROR - `metadata` must contain a 'desc_emb_tokenizer' key, but got {metadata}"
        self.path_to_tokenizer_config: str = path_to_tokenizer_config
        self.tokenizer_config: List[TokenizerConfigEntry] = load_tokenizer_config_from_path(path_to_tokenizer_config)
        
        # Set metadata
        self.metadata: Dict[str, Any] = {} if metadata is None else dict(metadata)
        self.metadata['cls'] = 'DescTokenizer'

        # Load underlying textual tokenizer
        self.desc_emb_tokenizer: str = metadata['desc_emb_tokenizer']
        self.tokenizer = AutoTokenizer.from_pretrained(self.desc_emb_tokenizer)
        self.event_separator: str = ' ' # character that gets inserted between Events when transformed into their textual descriptions
        
        # Metadata
        metadata = {} if metadata is None else metadata
        self.excluded_vocabs: Optional[Set[str]] = { x.lower() for x in metadata.get('excluded_vocabs', {}) } if metadata.get('excluded_vocabs', {}) else None # type: ignore
        self.min_code_occurrence_count: Optional[int] = metadata.get('min_code_occurrence_count', None)
        self.keep_n_max_occurrence_codes: Optional[int] = metadata.get('keep_n_max_occurrence_codes', None)
        
        # Apply filtering
        self.tokenizer_config, self.excluded_tokens = filter_tokenizer_config(self.tokenizer_config, 
                                                                              self.excluded_vocabs, 
                                                                              self.min_code_occurrence_count,
                                                                              self.keep_n_max_occurrence_codes)
        
        # Preprocess tokenizer config for quick access
        self.code_2_desc: Dict[str, str] = {}
        # initialize non special tokens list
        self.non_special_tokens: List[str] = []
        for entry in self.tokenizer_config:
            if entry.description is not None:
                self.code_2_desc[entry.code] = entry.description
                self.non_special_tokens.append(entry.description)

        # Define special tokens 
        self.special_tokens: List[str] = [
            self.tokenizer.bos_token,
            self.tokenizer.eos_token,
            self.tokenizer.unk_token,
            self.tokenizer.sep_token,
            self.tokenizer.pad_token,
            self.tokenizer.cls_token,
            self.tokenizer.mask_token,
        ]
        
        super().__init__(
            bos_token=self.tokenizer.bos_token,
            eos_token=self.tokenizer.eos_token,
            unk_token=self.tokenizer.unk_token,
            sep_token=self.tokenizer.sep_token,
            pad_token=self.tokenizer.pad_token,
            cls_token=self.tokenizer.cls_token,
            mask_token=self.tokenizer.mask_token
        )
    
    def convert_event_to_token(self, e: Event, **kwargs) -> Optional[str]:
        if e.code not in self.code_2_desc:
            return None
        return self.code_2_desc[e.code]

    def __call__(self, 
                 batch_of_events: Union[List[Event], List[List[Event]]],
                 is_truncation_random: bool = False,
                 seed: int = 1,
                 **kwargs) -> Dict[str, torch.Tensor]:
        """
            Tokenize a batch of patient timelines, where each timeline is a list of event codes.
            We add the ability to truncate seqs at random time points.
            
            Expects as input a list of Events
        """
        if not isinstance(batch_of_events[0], list):
            # List[Event] => List[List[Event]]
            batch_of_events = [ batch_of_events ] # type: ignore
        
        # First, convert all Events => ProtoTokens
        batch: List[List[str]] = [ self.convert_events_to_tokens(x) for x in batch_of_events ] 
        
        # Second, add special tokens (if applicable)
        if kwargs.get("add_special_tokens", False):
            batch = [ [ self.cls_token, self.bos_token ] + x + [ self.eos_token ] for x in batch ]

        # Concatenate all strings together for tokenization by traditional HF tokenizer
        # List[List[str]] => List[str]
        #batch = [ self.event_separator.join(x) for x in batch ] # type: ignore
        # Ensure proper filtering of None values
        batch = [self.event_separator.join(filter(None, x)) for x in batch]

        if is_truncation_random:
            max_length: int = kwargs.get("max_length")
            if not max_length:
                raise ValueError(f"If you specify `is_truncation_random`, then you must also provide a non-None value for `max_length`")

            # Tokenize without truncation
            if 'max_length' in kwargs:
                del kwargs['max_length']
            if 'truncation' in kwargs:
                del kwargs['truncation']
            tokenized_batch: Dict[str, torch.Tensor] = self.tokenizer.__call__(batch, **kwargs, truncation=None)

            # Truncate at random positions
            random.seed(seed)
            start_idxs: List[int] = []
            for timeline in tokenized_batch['input_ids']:
                length: int = (timeline != self.pad_token_id).sum() # count of non-PAD tokens
                if length > max_length:
                    # Calculate a random start index
                    start_index: int = random.randint(0, length - max_length)
                    start_idxs.append(start_index)
                else:
                    start_idxs.append(0)
                    
            for key in tokenized_batch.keys():
                truncated_batch: List[List[int]] = []
                for idx, timeline in enumerate(tokenized_batch[key]):
                    new_timeline = timeline[start_idxs[idx]:start_idxs[idx] + max_length]
                    assert new_timeline.shape[0] <= max_length, f"Error in truncating by random positions: new_timeline.shape = {new_timeline.shape[0]} !<= max_length={max_length}"
                    truncated_batch.append(new_timeline)
                if kwargs.get('return_tensors') == 'pt':
                    tokenized_batch[key] = torch.stack(truncated_batch, dim=0)
                else:
                    tokenized_batch[key] = truncated_batch
        else:
            tokenized_batch: Dict[str, torch.Tensor] = self.tokenizer.__call__(batch, **kwargs)

        return tokenized_batch

    """Mandatory overwrites of base class"""
    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer.get_vocab())

    def get_vocab(self) -> Dict[str, int]:
        return self.tokenizer.get_vocab()

    def _tokenize(self, text: str, **kwargs):
        return self.tokenizer._tokenize(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self.tokenizer._convert_token_to_id(token)

    def _convert_id_to_token(self, index: int) -> str:
        return self.tokenizer._convert_id_to_token(index)

def torch_mask_tokens(tokenizer: BaseTokenizer, 
                      inputs: Any, 
                      mlm_prob: float, 
                      special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
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
    
    # Check if mask_token is set properly
    if tokenizer.mask_token is None:
        raise ValueError("The tokenizer's mask_token is not set.")

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    mask_token_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    if mask_token_id is None:
        raise ValueError(f"The mask token {tokenizer.mask_token} could not be converted to an ID.")
    inputs[indices_replaced] = mask_token_id

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

def collate_femr_timelines(batch: List[Tuple[int, List[Event]]],
                             tokenizer: BaseTokenizer, 
                             dataset_name: str, # 'FEMRDataset' or 'AllTokensFEMRDataset'
                             max_length: int,
                             is_truncation_random: bool = False,
                             is_mlm: bool = False,
                             mlm_prob: float = 0.15,
                             seed: int = 1) -> Dict[str, Any]:
    """
        Collate function for FEMR timelines
        Truncate or pad to max length in batch.
    """
    timelines: List[List[Event]] = [ x[1] for x in batch if len(x[1]) > 0 ] # remove empty timelines
    if dataset_name == 'AllTokensFEMRDataset':
        # For AllTokensFEMRDataset, we are given explicit (start, end) idx's to subselect from each patient's timeline
        tokens: Dict[str, Float[torch.Tensor, 'B max_length']] = tokenizer(timelines, 
                                                                            truncation=True, 
                                                                            padding=True,
                                                                            max_length=max_length,
                                                                            is_truncation_random=False,
                                                                            add_special_tokens=True,
                                                                            seed=seed, 
                                                                            return_tensors='pt')
    elif dataset_name in ['FEMRDataset', 'MEDSDataset']:
        # For FEMRDataset and MEDSDataset, truncate timeline per usual
        tokens: Dict[str, Float[torch.Tensor, 'B max_length']] = tokenizer(timelines, 
                                                                            truncation=True, 
                                                                            padding=True, 
                                                                            max_length=max_length,
                                                                            is_truncation_random=is_truncation_random,
                                                                            add_special_tokens=True,
                                                                            seed=seed, 
                                                                            return_tensors='pt')
    else:
        raise ValueError(f"ERROR - Unsupported 'dataset_name' of: `{dataset_name}`")
    
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
    from hf_ehr.data.datasets import FEMRDataset
    
    PATH_TO_CACHE_DIR: str = '/share/pi/nigam/mwornow/hf_ehr/cache/'
    # Datasets
    PATH_TO_FEMR_EXTRACT_v8 = '/share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_02_08_extract_v8_no_notes'
    # Tokenizers
    PATH_TO_TOKENIZERS_DIR: str = os.path.join(PATH_TO_CACHE_DIR, 'tokenizers/')
    PATH_TO_TOKENIZER_CLMBR_v8_DIR: str = os.path.join(PATH_TO_TOKENIZERS_DIR, 'clmbr_v8/')
    PATH_TO_TOKENIZER_CLMBR_v8_CONFIG: str = os.path.join(PATH_TO_TOKENIZER_CLMBR_v8_DIR, 'tokenizer_config.json')
    PATH_TO_TOKENIZER_DESC_v8_DIR: str = os.path.join(PATH_TO_TOKENIZERS_DIR, 'desc_v8/')
    PATH_TO_TOKENIZER_DESC_v8_CONFIG: str = os.path.join(PATH_TO_TOKENIZER_DESC_v8_DIR, 'tokenizer_config.json')
    PATH_TO_TOKENIZER_COOKBOOK_v8_DIR: str = os.path.join(PATH_TO_TOKENIZERS_DIR, 'cookbook_v8/')
    PATH_TO_TOKENIZER_COOKBOOK_v8_CONFIG: str = os.path.join(PATH_TO_TOKENIZER_COOKBOOK_v8_DIR, 'tokenizer_config.json')
    PATH_TO_TOKENIZER_CEHR_v8_DIR: str = os.path.join(PATH_TO_TOKENIZERS_DIR, 'cehr_v8/')
    PATH_TO_TOKENIZER_CEHR_v8_CONFIG: str = os.path.join(PATH_TO_TOKENIZER_CEHR_v8_DIR, 'tokenizer_config.json')

    # Load v8 dataset
    print("Loading v8 dataset...")
    path_to_femr_extract: str = PATH_TO_FEMR_EXTRACT_v8
    dataset = FEMRDataset(path_to_femr_extract, split='train', is_debug=False)
    
    # Cookbook Tokenizer
    if False:
        print("Loading tokenizer...")
        tokenizer = CookbookTokenizer(PATH_TO_TOKENIZER_COOKBOOK_v8_CONFIG, metadata={
            'is_remap_numerical_codes_to_quantiles': False, # If True, remap numerical codes to a bucketed range
            'min_code_occurrence_count': 0, # Any code that occurs < `min_code_occurrence_count` times in the train dataset will be excluded
            'keep_n_max_occurrence_codes': None, # Keep only the top `keep_n_max_occurrence_codes` codes, sorted by occurrence count in train dataset
            'excluded_vocabs': ['STANFORD_OBS'], # Exclude all codes that are in these vocabularies
        })
        tokenizer.get_seq_length_per_patient(dataset, n_procs=5, is_force_refresh=True)

        tokens = tokenizer([ dataset[288000][1], dataset[288001][1], dataset[288002][1], dataset[288003][1] ], 
                        truncation=True,
                        padding=True,
                        is_truncation_random=True, 
                        max_length=1024, 
                        seed=0, 
                        add_special_tokens=True,
                        return_tensors='pt')
        print(tokens)
    
    dataset = FEMRDataset(path_to_femr_extract, split='test', is_debug=False)
    tokenizer = DescTokenizer(PATH_TO_TOKENIZER_DESC_v8_CONFIG,
                              metadata={ 
                                'desc_emb_tokenizer' : 'bert-base-uncased', 
                                'excluded_vocabs' : ['STANFORD_OBS'] 
                              })
    tokenizer.get_seq_length_per_patient(dataset, n_procs=5)
    exit()
    
    # Desc Tokenizer
    if True:
        print("Loading tokenizer...")
        tokenizer = DescTokenizer(PATH_TO_TOKENIZER_DESC_v8_CONFIG, metadata={ 'desc_emb_tokenizer' : 'bert-base-uncased', 'excluded_vocabs' : ['STANFORD_OBS'] })
        tokenizer.get_seq_length_per_patient(dataset, n_procs=5)

        # Check that initial Event => Token transformation is correct
        transformed_events = tokenizer.convert_events_to_tokens(dataset[0][1])
    
    # CLMBR Tokenizer
    if False:
        print("Loading tokenizer...")
        tokenizer = CLMBRTokenizer(PATH_TO_TOKENIZER_CLMBR_v8_CONFIG)
        
        tokenizer.get_seq_length_per_patient(dataset, n_procs=5)

        # Check that initial Event => Token transformation is correct
        transformed_events = tokenizer.convert_events_to_tokens(dataset[0][1])
        assert transformed_events == ['SNOMED/3950001', 'Gender/F', 'Ethnicity/Hispanic', 'LOINC/10525-4', 'SNOMED/609040007', 'CPT4/82306', 'LOINC/2236-8', 'SNOMED/12199005', 'LOINC/24348-5', 'SNOMED/5113004', 'CPT4/80053', 'CPT4/84075', 'LOINC/25302-1', 'LOINC/1919-0', 'SNOMED/24509005', 'SNOMED/25197003', 'SNOMED/26758005', 'SNOMED/250564007', 'SNOMED/18207002', 'SNOMED/304383000', 'SNOMED/36048009', 'SNOMED/52302001', 'SNOMED/46511006', 'SNOMED/39748002', 'SNOMED/71878006', 'SNOMED/359986008', 'SNOMED/59573005', 'SNOMED/687005', 'SNOMED/70901006', 'SNOMED/250707004', 'LOINC/2236-8 || None || -1.7976931348623157e+308 - 4.0', 'SNOMED/12199005 || None || 26.0 - 28.899999618530273', 'CPT4/84075 || None || -1.7976931348623157e+308 - 77.0', 'LOINC/25302-1 || None || 13.0 - 17.0', 'LOINC/1919-0 || None || 17.0 - 20.0', 'SNOMED/24509005 || None || 11.0 - 13.0', 'SNOMED/25197003 || None || 136.0 - 137.0', 'SNOMED/26758005 || None || 3.799999952316284 - 4.0', 'SNOMED/250564007 || None || 22.700000762939453 - 23.799999237060547', 'SNOMED/304383000 || None || 7.0 - 7.199999809265137', 'SNOMED/52302001 || None || 84.0 - 87.0', 'SNOMED/46511006 || None || 103.0 - 104.0', 'SNOMED/71878006 || None || 9.100000381469727 - 9.300000190734863', 'SNOMED/359986008 || None || 0.5 - 0.6000000238418579', 'SNOMED/59573005 || None || 4.329999923706055 - 4.5', 'SNOMED/687005 || None || 1.2000000476837158 - 1.2999999523162842', 'SNOMED/70901006 || None || 0.800000011920929 - 0.8999999761581421', 'SNOMED/250707004 || None || 3.0 - 3.200000047683716', 'LOINC/8480-6 || None || 129.0 - 1.7976931348623157e+308', 'LOINC/8462-4 || None || 74.0 - 80.0', 'LOINC/8302-2 || None || 59.0 - 61.41699981689453', 'Domain/OMOP generated', 'LOINC/8302-2 || None || 59.0 - 61.41699981689453', 'SNOMED/110483000', 'SNOMED/228490006 || N', 'SNOMED/228510007 || N', 'SNOMED/230056004 || N', 'SNOMED/230058003 || N', 'SNOMED/230057008 || N', 'SNOMED/271649006 || None || 126.0 - 1.7976931348623157e+308', 'SNOMED/271650006 || None || 75.0 - 80.0', 'SNOMED/417662000', 'Medicare Specialty/A0', 'LOINC/11506-3', 'SNOMED/110483000', 'SNOMED/228490006 || N', 'SNOMED/228510007 || N', 'SNOMED/230056004 || N', 'SNOMED/230058003 || N', 'SNOMED/230057008 || N', 'SNOMED/417662000', 'Medicare Specialty/A0', 'LOINC/11506-3', 'Visit/OP', 'SNOMED/110483000', 'SNOMED/228490006 || N', 'SNOMED/228510007 || N', 'SNOMED/230056004 || N', 'SNOMED/230058003 || N', 'SNOMED/230057008 || N', 'SNOMED/39104002', 'SNOMED/417662000', 'Medicare Specialty/A0', 'Medicare Specialty/A0']

        tokens = tokenizer([ dataset[0][1] ], 
                        truncation=True,
                        padding=True,
                        is_truncation_random=True, 
                        max_length=1024, 
                        seed=0, 
                        add_special_tokens=True,
                        return_tensors='pt')

        # Check that main Event => Token transformation is correct
        for idx, x in enumerate(tokens['input_ids'][0][2:-1]):
            assert tokenizer.idx_2_token[x.item()] == transformed_events[idx], f"ERROR - Mis-match between transformed_events and tokens: {tokenizer.idx_2_token[x.item()]} != {transformed_events[idx]} @ idx={idx}"
        # Check that added special tokens are correct
        assert tokenizer.idx_2_token[tokens['input_ids'][0][0].item()] == '[CLS]', f"ERROR - Tokenizer [CLS] mismatch"
        assert tokenizer.idx_2_token[tokens['input_ids'][0][1].item()] == '[BOS]', f"ERROR - Tokenizer [BOS] mismatch"
        assert tokenizer.idx_2_token[tokens['input_ids'][0][-1].item()] == '[EOS]', f"ERROR - Tokenizer [EOS] mismatch"
        print(tokens)

        exit()

