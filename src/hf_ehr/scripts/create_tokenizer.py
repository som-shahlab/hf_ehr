import json
from typing import Dict
import os
from tokenizers import Tokenizer, models, pre_tokenizers
from hf_ehr.config import PATH_TO_FEMR_EXTRACT_v9, PATH_TO_TOKENIZER_v9_DIR

if __name__ == '__main__':
    code_2_int: Dict[str, int] = json.load(open(os.path.join(PATH_TO_TOKENIZER_v9_DIR, 'code_2_int.json'), 'r'))
    code_2_count: Dict[str, int] = json.load(open(os.path.join(PATH_TO_TOKENIZER_v9_DIR, 'code_2_count.json'), 'r'))

    # Define the tokenizer model with the custom vocab
    # # vocab must be a dict where: [key] = token, [value] = unique integer
    # tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]", vocab=code_2_int))

    # # For simplicity, let's use a whitespace pre-tokenizer.
    # tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    # tokenizer.save(os.path.join(PATH_TO_TOKENIZER_DIR, "code_2_int_tokenizer.json"))
    # print("DONE!")

    # Train tokenizer
    # new_tokenizer = old_tokenizer.train_new_from_iterator(batch_iterator(dataset), 
    #                                                       vocab_size, 
    #                                                       length=len(dataset))
    # new_tokenizer.save_pretrained(os.path.join(path_to_output_dir, 'gpt2-tokenizer'))
    