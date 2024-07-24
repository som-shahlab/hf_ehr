# Tokenizers

Scripts for creating tokenizers.

We currently support three different tokenizers:
1. CLMBRTokenizer -- identical tokenizer as was used in the CLMBR/MOTOR/EHRSHOT papers
2. DescTokenizer -- tokenizer based on Ed Choi's DescEmb/UniHPF work
3. CookbookTokenizer -- custom tokenizer for this project


**Note:** Files marked `old--` are deprecated and should not be used. They are kept for reference only for future code.

## How to Run

Each tokenizer is created via a Python script. The script will generate a tokenizer and save it to disk, as described in the [File Structure](#file-structure) section below.

We recommend using many CPUs and lots of RAM as these scripts are highly parallelizable using the `--n_procs` flag. They default to using 5 CPUs.

### CLMBRTokenizer

To generate the CLMBR tokenizer, run:

```bash
python3 create_clmbr.py
```

### DescTokenizer

To generate the DescEmb tokenizer based on the v8 dataset, run:

```bash
python3 create_desc.py --dataset v8
```

### CookbookTokenizer

To generate the Cookbook tokenizer based on the v8 dataset, run:

```bash
python3 create_cookbook.py --dataset v8
```

## File Structure

All tokenizers will get written to `/share/pi/nigam/mwornow/hf_ehr/cache/tokenizers`. 

The folder structure is as follows:

```
tokenizers/
    {tokenizer_name}/
        tokenizer_config.json # Contains the tokenizer config with all tokens for this tokenizer
        versions/
            {datetime-1}/ # unique datetime for each tokenizer version
                metadata.json # Contains the tokenizer metadata, e.g. remap numerical codes, excluded vocabs, etc.
                tokenizer_config_filtered.json # Contains the tokenizer config with only the tokens actually kept in the vocab
                vocab.json # Maps textualized tokens to integer IDs
                datasets/
                    {datetime-1a}/ # unique datetime for each dataset version
                        metadata.json # Contains the dataset metadata, e.g. femr extract path, is_debug, etc.
                        seq_length_per_patient.json # Maps each idx in dataset to the seq_length of the patient when using this tokenizer version
                    {datetime-1b}/
                        ...
            {datetime-2}/
                ...
    {tokenizer_name-2}/
        ...
```
