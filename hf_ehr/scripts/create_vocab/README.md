# Tokenizer

Scripts for creating tokenizers.

## General Format:

All tokenizers will get written to `/share/pi/nigam/mwornow/hf_ehr/cache/tokenizers`. The folder structure is as follows:

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

## CLMBRTokenizer

To generate the CLMBR tokenizer, run:

```bash
python3 create_clmbr.py
```

## DescTokenizer

To generate the DescEmb tokenizer based on the v8 dataset, run:

```bash
python3 create_desc.py --dataset v8
```

## CookbookTokenizer

To generate the Cookbook tokenizer based on the v8 dataset, run:

```bash
python3 create_cookbook.py --dataset v8
```