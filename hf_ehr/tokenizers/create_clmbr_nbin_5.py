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

# Path to the original CLMBR dictionary
PATH_TO_CLMBR_JSON: str = os.path.join(PATH_TO_TOKENIZER_CLMBR_v8_DIR, 'clmbr_v8_original_dictionary.json')

# Path to the new numerical bins data
PATH_TO_NUMERICAL_BINS_JSON = '/share/pi/nigam/mwornow/hf_ehr/cache/create_cookbook/numerical_bins_final_output.json'

if __name__ == '__main__':
    start_total = time.time()
    
    # Load original CLMBR dictionary
    clmbr: Dict[str, List] = json.load(open(PATH_TO_CLMBR_JSON))
    path_to_output_dir: str = '/share/pi/nigam/mwornow/hf_ehr/cache/tokenizers/clmbr_v8_nbin_5/'

    # Load the new numerical bins data
    numerical_bins: List[Dict] = json.load(open(PATH_TO_NUMERICAL_BINS_JSON))

    # Create a lookup dictionary for the new numerical data based on code_string
    numerical_bins_lookup = {}
    for entry in numerical_bins:
        code_string = entry["code_string"]
        if code_string not in numerical_bins_lookup:
            numerical_bins_lookup[code_string] = []
        numerical_bins_lookup[code_string].append(entry)

    tokenizer_config: List[TokenizerConfigEntry] = []
    numerical_replacement_count = 0  # Initialize a counter for replaced numerical values

    # Process the original CLMBR dictionary
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
            'code': code,
            'description': None,
            'type': (
                'numerical_range' if type_ == 'numeric' else
                'categorical' if type_ == 'text' else
                'code'
            ),
            'stats': [
                CountOccurrencesTCEStat(split='train', dataset='v8'),
                PPLTCEStat(split='train', dataset='v8', model="gpt2-base-1024"),
            ],
        }

        if type_ == 'code':
            new_token = CodeTCE(
                **defaults,
            )
            tokenizer_config.append(new_token)
        elif type_ == 'text':
            new_token = CategoricalTCE(
                tokenization={
                    'categories': [text_string],
                },
                **defaults,
            )
            tokenizer_config.append(new_token)
        elif type_ == 'numeric':  # This is a numerical type
            # Replace with new numerical data if available
            if code in numerical_bins_lookup:
                for new_entry in numerical_bins_lookup[code]:
                    new_token = NumericalRangeTCE(
                        code=code,
                        description=None,
                        type="numerical_range",
                        stats=[
                            CountOccurrencesTCEStat(split='train', dataset='v8'),
                            PPLTCEStat(split='train', dataset='v8', model="gpt2-base-1024"),
                        ],
                        tokenization={
                            'unit': "None",
                            'range_start': new_entry["val_start"],
                            'range_end': new_entry["val_end"],
                        }
                    )
                    tokenizer_config.append(new_token)
                numerical_replacement_count += 1  # Increment the count for each replacement
            else:
                # If no new data, keep the old numerical range
                new_token = NumericalRangeTCE(
                    tokenization={
                        'unit': "None",
                        'range_start': val_start,
                        'range_end': val_end,
                    },
                    **defaults,
                )
                tokenizer_config.append(new_token)
        else:
            raise ValueError(f"ERROR - Unknown type for code {code}: {type_}")

    # Save the updated tokenizer configuration
    path_to_output_file: str = os.path.join(path_to_output_dir, 'tokenizer_config.json')
    print(f"Saving updated CLMBR vocab to: `{path_to_output_file}`")
    save_tokenizer_config_to_path(path_to_output_file, tokenizer_config)
    
    n_new_tokens: int = len(tokenizer_config)
    n_old_tokens: int = len([x for x in clmbr['regular'] if x['type'] != 'unused'])
    print("Number of tokens in new CLMBR vocab: ", n_new_tokens)
    print("Number of tokens in old CLMBR vocab: ", n_old_tokens)
    
    # Print the number of numerical values replaced
    print(f"Number of numerical values replaced: {numerical_replacement_count}")
    
    print(f"Total time taken: {round(time.time() - start_total, 2)}s")
    print("Done!")
