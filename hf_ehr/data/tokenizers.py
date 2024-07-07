import random
from typing import Dict, List, Optional, Set, Tuple, Union, Any, TypedDict
import torch
import json
from transformers import PreTrainedTokenizer, AutoTokenizer
from hf_ehr.config import Code2Detail

class FEMRTokenizer(PreTrainedTokenizer):
    def __init__(self, 
                 path_to_code_2_detail: str, 
                 is_remap_numerical_codes: bool = False, # if TRUE, then remap numericals to buckets based on quantile of value
                 excluded_vocabs: Optional[List[str]] = None,
                 min_code_count: Optional[int] = None) -> None:
        self.path_to_code_2_detail: str = path_to_code_2_detail
        self.code_2_detail: Code2Detail = json.load(open(path_to_code_2_detail, 'r'))

        # Get vocabulary
        codes: List[str] = []
        if is_remap_numerical_codes:
            # Use the lab-value quantiled version of the FEMR codes
            for detail in self.code_2_detail.values():
                codes += list(detail['token_2_count'].keys())
        else:
            # Just use the raw FEMR codes as is
            codes: List[str] = sorted(list(self.code_2_detail.keys()))

        # Filter out excluded vocabs (if applicable)
        self.excluded_vocabs: Optional[List[str]] = excluded_vocabs
        if excluded_vocabs is not None:
            excluded_vocabs: Set[str] = { x.lower() for x in excluded_vocabs } # type: ignore
            codes = [ x for x in codes if x.split("/")[0].lower() not in excluded_vocabs ]

        # Only keep codes with >= `min_code_count` occurrences in our dataset
        self.min_code_count: Optional[int] = min_code_count
        if min_code_count is not None:
            codes = [x for x in codes if self.is_code_above_min_count(x)]

        # Create vocab
        self.special_tokens = [ '[BOS]', '[EOS]', '[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']
        self.non_special_tokens = codes
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
        
    def is_code_above_min_count(self, token: str):
        if token in self.code_2_detail:
            code: str = token
        else:
            code: str = token.split(" || ")[0]
        token_2_count = self.code_2_detail[code]['token_2_count']
        return sum(token_2_count.values()) >= self.min_code_count
    
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
            # List[str] => List[List[str]]
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

    def _tokenize(self, text: str, **kwargs) -> int:
        """Here, `text` will be a single code (e.g. "LOINC/13"), so directly map it to an id in our vocab"""
        return self._convert_token_to_id(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self.token_2_idx[token]

    def _convert_id_to_token(self, index: int) -> str:
        raise self.idx_2_token[index]


class DescTokenizer(PreTrainedTokenizer):
    """Converts codes => textual descriptions, then tokenizes
    """
    def __init__(self, tokenizer: AutoTokenizer) -> None:
        self.tokenizer = tokenizer
        self.code_separator: str = ' ' # separate descriptions with a space by default
        
        super().__init__(
            bos_token=tokenizer.bos_token,
            eos_token=tokenizer.eos_token,
            unk_token=tokenizer.unk_token,
            sep_token=self.code_separator,
            pad_token=tokenizer.pad_token,
            cls_token=tokenizer.cls_token
        )

    def __call__(self, 
                 batch: Union[List[str], List[List[str]]],
                 is_truncation_random: bool = False,
                 seed: int = 1,
                 **kwargs) -> Dict[str, torch.Tensor]:
        '''Tokenize a batch of patient timelines, where each timeline is a list of event codes.
            We add the ability to truncate seqs at random time points.
            
            Expects as input a list of text-fied code descriptions in either the format of:
                A list of codes (List[str])
                A list of lists of codes (List[str])
        '''
        if isinstance(batch[0], str):
            # List[str] => List[List[str]]
            batch = [ batch ]

        # Concatenate all strings together for tokenization by traditional HF tokenizer
        # List[List[str]] => List[str]git 
        batch = [ self.code_separator.join(x) for x in batch ]

        if is_truncation_random:
            max_length: int = kwargs.get("max_length")
            if not max_length:
                raise ValueError(f"If you specify `is_truncation_random`, then you must also provide a non-None value for `max_length`")

            # Tokenize without truncation
            kwargs.pop('max_length')
            kwargs.pop('truncation')
            tokenized_batch: Dict[str, torch.Tensor] = self.tokenizer.__call__(batch, **kwargs, truncation=None)

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