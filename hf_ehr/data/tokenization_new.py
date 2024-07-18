import collections
import random
from typing import Dict, List, Optional, Set, Tuple, Union, Any, TypedDict
import torch
import json
from transformers import PreTrainedTokenizer, AutoTokenizer
from hf_ehr.config import Code2Detail, Event, TokenizerConfigEntry, load_tokenizer_config_from_path

def convert_event_to_token(e: Event, tokenizer_config: Code2Detail, **kwargs) -> Optional[str]:
    """
        Helper function used in FEMRDataset to convert a FEMR Event `e` to a string `token` based on the dataset's configuration.
        If return `None`, then ignore this event (i.e. has no corresponding token).
    """
    # Parse kwargs
    excluded_vocabs: Set[str] = kwargs.get('excluded_vocabs', {}) or {}
    min_code_occurrence_count: Optional[int] = kwargs.get('min_code_occurrence_count', None)
    is_remap_numerical_codes: bool = kwargs.get('is_remap_numerical_codes', False) or False
    is_remap_codes_to_desc: bool = kwargs.get('is_remap_codes_to_desc', False) or False
    is_clmbr: bool = kwargs.get('is_clmbr', False) or False

    # Default the token to just being the literal code
    token: str = e.code # e.g. "LOINC/10230-1"

    # If exclude certain vocabs, then ignore this token if it belongs to one of those vocabs (e.g. "STANFORD_OBS/")            
    if token.split("/")[0].lower() in { x.lower() for x in excluded_vocabs }:
        return None
    
    # Check if code is above min count
    def is_code_above_min_count(token: str, min_code_occurrence_count: int) -> bool:
        if token in tokenizer_config:
            code: str = token
        else:
            code: str = token.split(" || ")[0]
        token_2_count = tokenizer_config[code]['token_2_count']
        return sum(token_2_count.values()) >= min_code_occurrence_count

    if min_code_occurrence_count:
        if not is_code_above_min_count(token, min_code_occurrence_count):
            return None
    
    # If CLMBR then do special mapping and continue

    # First, if remap codes to textual descs => change `token` to textual definition of code
    if is_remap_codes_to_desc:
        # "LOINC/10230-1" => "Left ventricular Ejection fraction"
        token = tokenizer_config[e.code]['desc']

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
            assert 'unit_2_quartiles' in tokenizer_config[e.code], f"ERROR - Missing 'unit_2_quartiles' for code={e.code} in tokenizer_config: {tokenizer_config[e.code]}"
            assert unit in tokenizer_config[e.code]['unit_2_quartiles'], f"ERROR - Missing unit={unit} in 'unit_2_quartiles' for code={e.code} in tokenizer_config: {tokenizer_config[e.code]['unit_2_quartiles']}"
            quantiles: List[float] = tokenizer_config[e.code]['unit_2_quartiles'][unit]

            # Determine quantile for (code, unit, value)
            token = convert_lab_value_to_token_from_quantiles(token, unit, e.value, quantiles)
    return token

class BaseCodeTokenizer(PreTrainedTokenizer):
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

    def convert_events_to_tokens(self, events: List[Event], **kwargs) -> List[str]:
        """Provide default implementation where one Event => one token"""
        tokens: List[str] = []
        for e in events:
            token: Optional[str] = self.convert_event_to_token(e, **kwargs)
            if token is not None:
                tokens.append(token)
        return tokens
    
    def convert_event_to_token(self, e: Event, **kwargs) -> Optional[str]:
        raise NotImplementedError("Must implement `self.convert_event_to_token()` in child class")

    def __call__(self, 
                 batch_of_events: Union[List[Event], List[List[Event]]],
                 is_truncation_random: bool = False,
                 seed: int = 1,
                 **kwargs) -> Dict[str, torch.Tensor]:
        '''Tokenize a batch of patient timelines, where each timeline is a list of event codes.
            We add the ability to truncate seqs at random time points.
            
            Expects as input a list of Events

            NOTE: Must set `is_split_into_words=True` b/c we've already pre-tokenized our inputs (i.e. we're passing in a List of tokens, not a string)
        '''
        if not isinstance(batch_of_events[0], list):
            # List[Event] => List[List[Event]]
            batch_of_events = [ batch_of_events ] # type: ignore
        
        # First, convert all Events => ProtoTokens
        batch: List[List[str]] = [ self.convert_events_to_tokens(x) for x in batch_of_events ]
        
        # Second, add special tokens (if applicable)
        if kwargs.get("add_special_tokens", False):
            batch = [ [ self.cls_token, self.bos_token ] + x + [ self.eos_token ] for x in batch ]

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
            tokenized_batch: Dict[str, torch.Tensor] = super().__call__(batch, **kwargs, is_split_into_words=True)

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
    # TODO - move this settings upstream to the creation of the actual tokenizer_config.json
    """
        Settings:
            is_remap_numerical_codes: bool
                - If TRUE, then remap numericals to buckets based on quantile of value
            excluded_vocabs: Optional[List[str]]
                - List of vocabs to exclude from the tokenizer. Determined by the first part of the code before the '/' (e.g. "STANFORD_OBS" in "STANFORD_OBS/1234")
            min_code_occurrence_count: Optional[int]
                - Only keep tokens with >= `min_code_occurrence_count` total occurrences in our dataset
    """
    def __init__(self, 
                 path_to_tokenizer_config: str, 
                 is_remap_numerical_codes: bool = False, # if TRUE, then remap numericals to buckets based on quantile of value
                 excluded_vocabs: Optional[Union[Set[str], List[str]]] = None,
                 min_code_occurrence_count: Optional[int] = None) -> None:
        self.path_to_tokenizer_config: str = path_to_tokenizer_config
        self.tokenizer_config: List[TokenizerConfigEntry] = load_tokenizer_config_from_path(path_to_tokenizer_config)
        
        # Settings
        self.is_remap_numerical_codes: bool = is_remap_numerical_codes
        self.excluded_vocabs: Optional[Set[str]] = { x.lower() for x in excluded_vocabs } if excluded_vocabs else None # type: ignore
        self.min_code_occurrence_count: Optional[int] = min_code_occurrence_count
        
        # Tokens
        self.code_2_token = {} # [key] = token; [val] = { 'type' : str, 'tokenization' : dict, 'token' : str }
        self.non_special_tokens: List[str] = []
        self.excluded_tokens: List[str] = [] # for tracking purposes

        # Preprocess tokenizer config for quick access
        for entry in self.tokenizer_config:
            
            # Settings (if applicable)
            ## Remove tokens from excluded vocabs
            if (
                self.excluded_vocabs is not None
                and entry.code.split("/")[0].lower() in self.excluded_vocabs
            ):
                self.excluded_tokens.append(entry.to_token())
                continue
            ## Remove tokens with < `min_code_occurrence_count` occurrences in our dataset
            if (
                min_code_occurrence_count is not None
                and entry.get_stat('count_occurrences') < min_code_occurrence_count
            ):
                self.excluded_tokens.append(entry.to_token())
                continue

            # If we've made it here, then we want to keep this token
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
        # If code isn't in vocab => ignore
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
            for token_range in self.code_2_token[e.code]['numerical_range']:
                assert 'token' in token_range, f"ERROR - Missing 'token' for code={e.code},type=numerical_range in self.code_2_token: {self.code_2_token[e.code]['numerical_range']}"
                assert 'tokenization' in token_range, f"ERROR - Missing 'tokenization' for code={e.code},type=numerical_range in self.code_2_token: {self.code_2_token[e.code]['numerical_range']}"
                token: str = token_range['token']
                unit: str = token_range['tokenization']['unit']
                range_start: float = token_range['tokenization']['range_start']
                range_end: float = token_range['tokenization']['range_end']
                if range_start <= e.value <= range_end and e.unit == unit:
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

class CLMBRTokenizer(BaseCodeTokenizer):
    def __init__(self, path_to_tokenizer_config: str) -> None:
        self.path_to_tokenizer_config: str = path_to_tokenizer_config
        self.tokenizer_config: List[TokenizerConfigEntry] = load_tokenizer_config_from_path(path_to_tokenizer_config)

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

        assert len(self.non_special_tokens) == 39811, f"ERROR - Expected 39811 self.non_special_tokens, but got {len(self.non_special_tokens)}"

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

class DescTokenizer(PreTrainedTokenizer):
    """Converts codes => textual descriptions, then tokenizes
    """
    def __init__(self, path_to_tokenizer_config: str, tokenizer: AutoTokenizer) -> None:
        self.path_to_tokenizer_config: str = path_to_tokenizer_config
        self.tokenizer_config: List[TokenizerConfigEntry] = load_tokenizer_config_from_path(path_to_tokenizer_config)
        self.event_separator: str = ' ' # character that gets inserted between Events when transformed into their textual descriptions
        self.tokenizer = tokenizer
        
        # Preprocess tokenizer config for quick access
        self.code_2_desc: Dict[str, str] = {}
        for entry in self.tokenizer_config:
            if entry.description is not None:
                self.code_2_desc[entry.code] = entry.description

        super().__init__(
            bos_token=tokenizer.bos_token,
            eos_token=tokenizer.eos_token,
            unk_token=tokenizer.unk_token,
            sep_token=tokenizer.sep_token,
            pad_token=tokenizer.pad_token,
            cls_token=tokenizer.cls_token
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
        '''Tokenize a batch of patient timelines, where each timeline is a list of event codes.
            We add the ability to truncate seqs at random time points.
            
            Expects as input a list of Events
        '''
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
        batch = [ self.event_separator.join(x) for x in batch ] # type: ignore

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

if __name__ == '__main__':
    from hf_ehr.data.datasets_new import FEMRDataset
    from hf_ehr.config import PATH_TO_FEMR_EXTRACT_v8, PATH_TO_TOKENIZER_CLMBR_v8_CONFIG
    
    # Load v8 dataset
    print("Loading v8 dataset...")
    path_to_femr_extract: str = PATH_TO_FEMR_EXTRACT_v8
    dataset = FEMRDataset(path_to_femr_extract, split='train', is_debug=True)
    
    # Cookbook Tokenizer
    if False:
        path_to_tokenizer_config: str = '/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v8/tokenizer_config.json'
        print("Loading tokenizer...")
        tokenizer = FEMRTokenizer(path_to_tokenizer_config, excluded_vocabs=['STANFORD_OBS'])
        
        tokens = tokenizer([ dataset[288000][1], dataset[288001][1], dataset[288002][1], dataset[288003][1] ], 
                        truncation=True,
                        padding=True,
                        is_truncation_random=True, 
                        max_length=1024, 
                        seed=0, 
                        add_special_tokens=True,
                        return_tensors='pt')
        print(tokens)
        
    # Desc Tokenizer
    if False:
        print("Loading tokenizer...")
        desc_tokenizer = DescTokenizer(AutoTokenizer.from_pretrained("bert-base-uncased"))
    
    # CLMBR Tokenizer
    if True:
        print("Loading tokenizer...")
        tokenizer = CLMBRTokenizer(PATH_TO_TOKENIZER_CLMBR_v8_CONFIG)

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