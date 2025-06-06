"""
Transforms the original CLMBR dictionary (from FEMRv1) into tokenizer_config.json format
Limits to top k tokens
"""
import argparse
import json
import time
from typing import List, Dict
import os
from tqdm import tqdm
from hf_ehr.config import (
    TokenizerConfigEntry, NumericalRangeTCE, CategoricalTCE, CodeTCE, 
    CountOccurrencesTCEStat, PPLTCEStat,
    save_tokenizer_config_to_path,
)
from huggingface_hub import hf_hub_download
from hf_ehr.utils import get_rel_path, get_tokenizer_info_from_config_yaml

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Create CookbookTokenizer for a dataset')
    parser.add_argument('--path_to_tokenizer_config', default=get_rel_path('configs/tokenizer/clmbr.yaml'), type=str, help='Config .yaml file for tokenizer to use. e.g. `../configs/tokenizer/clmbr.yaml`. This YAML file will contain a key called `path_to_config` where the tokenizer will be saved.')
    parser.add_argument('--path_to_orig_clmbr_json', required=False, default=None, type=str, help='Path to `clmbr_v8_original_dictionary.json`. If not specified, will download file from: https://huggingface.co/StanfordShahLab/clmbr-t-base/blob/main/clmbr_v8_original_dictionary.json. NOTE: This requires you are authenticated with Hugging Face.')
    parser.add_argument('--k', default=None, type=int, help='Desired vocab size (in thousands). If not specified, will not limit vocab size.')
    return parser.parse_args()

if __name__ == '__main__':
    start_total = time.time()
    args = parse_args()
    
    # Download original CLMBR dictionary if not specified
    if args.path_to_orig_clmbr_json is None:
        file_path = hf_hub_download(
            repo_id="StanfordShahLab/clmbr-t-base",
            filename="clmbr_v8_original_dictionary.json",
            subfolder="",
            revision="main"
        )
        args.path_to_orig_clmbr_json = file_path
        print(f"Downloaded to: {args.path_to_orig_clmbr_json}")
    
    # Load tokenizer config
    path_to_tokenizer_config, __ = get_tokenizer_info_from_config_yaml(args.path_to_tokenizer_config)
    
    # Set vocab size
    max_vocab_size: Optional[int] = None
    if args.k is not None:
        max_vocab_size = args.k * 1000
        # Create new tokenizer config directory with `_k` suffix (e.g. `clmbr_v8_8k`) b/c limiting vocab to top-k codes
        path_to_old_tokenizer_config_dir: str = os.path.dirname(path_to_tokenizer_config)
        path_to_new_tokenizer_config_dir: str = path_to_old_tokenizer_config_dir + f'_{args.k}k'
        path_to_tokenizer_config: str = os.path.join(path_to_new_tokenizer_config_dir, os.path.basename(path_to_tokenizer_config))
        os.makedirs(path_to_new_tokenizer_config_dir, exist_ok=True)
        print(f"Limiting vocab to top {args.k}k codes. Saving tokenizer config to: `{path_to_tokenizer_config}`")

    # Load original CLMBR dictionary
    clmbr: Dict[str, List] = json.load(open(args.path_to_orig_clmbr_json))

    tokenizer_config: List[TokenizerConfigEntry] = []
    for token in tqdm(clmbr['regular'], desc='Looping thru CLMBR codes...', total=len(clmbr['regular'])):
        code: str = token['code_string']
        val_start: float = token['val_start']
        val_end: float = token['val_end']
        type_ = token['type']
        text_string = token['text_string']
        
        # Skip ignored tokens
        if type_ == 'unused' or type_ == 3:
            continue

        defaults = {
            'code' : code,
            'description' : None,
            'type' : (
                'numerical_range' if (type_ == 'numeric' or type_ == 1) else
                'categorical' if (type_ == 'text' or type_ == 2) else
                'code'
            ),
            'stats' : [
                # dummy values, just to show what's possible
                CountOccurrencesTCEStat(split='train', dataset='v8'),
                PPLTCEStat(split='train', dataset='v8', model="gpt2-base-1024"),
            ],
        }
        if type_ == 'code' or type_ == 0:
            new_token = CodeTCE(
                **defaults,
            )
        elif type_ == 'text' or type_ == 2:
            new_token = CategoricalTCE(
                tokenization={
                    'categories' : [ text_string ],
                },
                **defaults,
            )
        elif type_ == 'numeric' or type_ == 1:
            unit: str = "None"
            new_token = NumericalRangeTCE(
                tokenization={
                    'unit' : unit,
                    'range_start' : val_start,
                    'range_end' : val_end,
                },
                **defaults,
            )
        else:
            raise ValueError(f"ERROR - Unknown type for code {code}: {type_}")
            
        tokenizer_config.append(new_token)
        
        if max_vocab_size is not None and len(tokenizer_config) >= max_vocab_size:
            print(f"We've reached the desired vocab size of {max_vocab_size}")
            break
        
    print(f"Saving CLMBR vocab to: `{path_to_tokenizer_config}`")
    save_tokenizer_config_to_path(path_to_tokenizer_config, tokenizer_config)
    
    n_new_tokens: int = len(tokenizer_config)
    n_old_tokens: int = len([ x for x in clmbr['regular'] if x['type'] != 'unused' ])
    print("Number of tokens in new CLMBR vocab: ", n_new_tokens)
    print("Number of tokens in old CLMBR vocab: ", n_old_tokens)
    
    print(f"Total time taken: {round(time.time() - start_total, 2)}s")
    print(f"Saved tokenizer config to: `{path_to_tokenizer_config}`")
    print("Done!")