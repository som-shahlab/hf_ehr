"""
Transforms the original CLMBR dictionary (from FEMRv1) into tokenizer_config.json format
"""
import json
import time
from typing import List, Dict
import os
from tqdm import tqdm
from hf_ehr.config import (
    TokenizerConfigEntry, NumericalRangeTCE, CategoricalTCE, CodeTCE, 
    CountOccurrencesTCEStat, PPLTCEStat,
    save_tokenizer_config_to_path,
    PATH_TO_TOKENIZER_CLMBR_v8_DIR
)

desired_vocab_size: int = 96000
PATH_TO_TOKENIZER_CLMBR_v8_DIR: str = PATH_TO_TOKENIZER_CLMBR_v8_DIR.replace("clmbr_v8", f"clmbr_v8_{desired_vocab_size // 1000}k")
PATH_TO_CLMBR_JSON: str = os.path.join(PATH_TO_TOKENIZER_CLMBR_v8_DIR, 'clmbr_v8_original_dictionary_full.json')

if __name__ == '__main__':
    start_total = time.time()
    
    # Load original CLMBR dictionary
    clmbr: Dict[str, List] = json.load(open(PATH_TO_CLMBR_JSON))
    path_to_output_dir: str = os.path.dirname(PATH_TO_CLMBR_JSON)
    print("Output dir:", path_to_output_dir)

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
                'numerical_range' if type_ == 'numeric' else
                'categorical' if type_ == 'text' else
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
        
        if len(tokenizer_config) >= desired_vocab_size:
            print(f"We've reached the desired vocab size of {desired_vocab_size}")
            break
        
    path_to_output_file: str = os.path.join(path_to_output_dir, 'tokenizer_config.json')
    print(f"Saving CLMBR vocab to: `{path_to_output_file}`")
    save_tokenizer_config_to_path(path_to_output_file, tokenizer_config)
    
    n_new_tokens: int = len(tokenizer_config)
    n_old_tokens: int = len([ x for x in clmbr['regular'] if x['type'] != 'unused' ])
    print("Number of tokens in new CLMBR vocab: ", n_new_tokens)
    print("Number of tokens in old CLMBR vocab: ", n_old_tokens)
    print(f"Total time taken: {round(time.time() - start_total, 2)}s")
    assert n_new_tokens == n_old_tokens, f"ERROR - Mismatch in vocab lengths"
    print("Done!")