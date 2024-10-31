"""
Transforms the original CLMBR dictionary (from FEMRv1) into tokenizer_config.json format for CEHR
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
    PATH_TO_TOKENIZER_CLMBR_v8_DIR, 
    PATH_TO_TOKENIZER_CEHR_v8_DIR
)
PATH_TO_CLMBR_JSON: str = os.path.join(PATH_TO_TOKENIZER_CLMBR_v8_DIR, 'clmbr_v8_original_dictionary.json')

if __name__ == '__main__':
    start_total = time.time()
    
    # Load original CLMBR dictionary
    clmbr: Dict[str, List] = json.load(open(PATH_TO_CLMBR_JSON))
    path_to_output_dir: str = os.path.dirname(PATH_TO_TOKENIZER_CEHR_v8_DIR)

    tokenizer_config: List[TokenizerConfigEntry] = []
    for token in tqdm(clmbr['regular'], desc='Looping thru CLMBR codes...', total=len(clmbr['regular'])):
        code: str = token['code_string']
        val_start: float = token['val_start']
        val_end: float = token['val_end']
        type_ = token['type']
        text_string = token['text_string']
        
        # Skip ignored tokens
        if type_ == 'unused':
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
        if type_ == 'code':
            new_token = CodeTCE(
                **defaults,
            )
        elif type_ == 'text':
            new_token = CategoricalTCE(
                tokenization={
                    'categories' : [ text_string ],
                },
                **defaults,
            )
        elif type_ == 'numeric':
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
    
    # Add CEHR-specific metadata for visit and ATT tokens
    metadata = {
        'is_add_visit_start': True,
        'is_add_visit_end': True,
        'is_add_day_att': False,
        'is_add_day_week_month_att': True
    }
    
    # Add CEHR-specific metadata for visit and ATT tokens if needed
    # For example, you can define visit-related tokens here and add them to the tokenizer config
    # Add visit-related tokens to the tokenizer config
    visit_tokens = [
        CodeTCE(code="[VISIT START]", description="Visit start token", type="code", stats=[]),
        CodeTCE(code="[VISIT END]", description="Visit end token", type="code", stats=[])
    ]

    # Add ATT tokens (e.g., day, long-term tokens)
    att_tokens = [
        CodeTCE(code=day_att, description=f"{day_att} ATT token", type="code", stats=[])
        for day_att in [f"[DAY {i}]" for i in range(1, 7)]  # Days 1 to 6
    ] + [
        CodeTCE(code=week_att, description=f"{week_att} ATT token", type="code", stats=[])
        for week_att in [f"[WEEK {i}]" for i in range(1, 5)]  # Weeks 1 to 4
    ] + [
        CodeTCE(code=month_att, description=f"{month_att} ATT token", type="code", stats=[])
        for month_att in [f"[MONTH {i}]" for i in range(1, 13)]  # Months 1 to 12
    ] + [
        CodeTCE(code="[LONG TERM]", description="Long-term ATT token", type="code", stats=[])
    ]

    # Add the visit and ATT tokens to the configuration
    tokenizer_config.extend(visit_tokens)
    tokenizer_config.extend(att_tokens)
        
    path_to_output_file: str = os.path.join(path_to_output_dir, 'tokenizer_config.json')
    print(f"Saving CLMBR vocab to: `{path_to_output_file}`")
    save_tokenizer_config_to_path(path_to_output_file, tokenizer_config)
    
    n_new_tokens: int = len(tokenizer_config)
    n_old_tokens: int = len([ x for x in clmbr['regular'] if x['type'] != 'unused' ])
    print("Number of tokens in new CLMBR vocab: ", n_new_tokens)
    print("Number of tokens in old CLMBR vocab: ", n_old_tokens)
    #assert n_new_tokens == n_old_tokens, f"ERROR - Mismatch in vocab lengths"
    
    print(f"Total time taken: {round(time.time() - start_total, 2)}s")
    print("Done!")
