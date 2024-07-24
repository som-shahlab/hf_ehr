# Tokenizers

Scripts for creating tokenizers.

We currently support three different tokenizers:
1. `CLMBRTokenizer` -- identical tokenizer as was used in the CLMBR/MOTOR/EHRSHOT papers
2. `DescTokenizer` -- tokenizer based on Ed Choi's DescEmb/UniHPF work
3. `CookbookTokenizer` -- custom tokenizer for this project

**Note:** Files marked `old--` are deprecated and should not be used. They are kept for reference only for future code.

## Quick Start

```bash
python3 tokenizers/create_clmbr.py # Takes ~5 seconds
python3 tokenizers/create_desc.py  # Takes ~30 mins
python3 tokenizers/create_cookbook.py # TBD
```

## How to Create a Tokenizer

Each tokenizer is created via a Python script. The script will generate a tokenizer and save it to disk, as described in the [File Structure](#file-structure) section below.

We recommend using many CPUs and lots of RAM as these scripts are highly parallelizable using the `--n_procs [int]` flag. 

They default to using `--n_procs 5`.

### CLMBRTokenizer

To generate the `CLMBRTokenizer`, run:

```bash
python3 create_clmbr.py
```

### DescTokenizer

To generate the `DescTokenizer` based on the v8 dataset using 10 CPUs, run:

```bash
python3 create_desc.py --dataset v8 --n_procs 10
```

### CookbookTokenizer

To generate the `CookbookTokenizer` based on the v8 dataset using 10 CPUs, run:

```bash
python3 create_cookbook.py --dataset v8 --n_procs 10
```

## File Structure

All tokenizers will get written to `/share/pi/nigam/mwornow/hf_ehr/cache/tokenizers`. 

The folder structure is as follows:

* `tokenizers/`
    * `{tokenizer_name}/` -- e.g. `clmbr_v8`, `desc_v8`, `cookbook_v8`
        * `tokenizer_config.json` -- Contains the main config with all tokens for this tokenizer
        * `versions/`
            * `{datetime-1}/` -- unique datetime for each tokenizer version
                * `metadata.json` -- Contains the tokenizer metadata, e.g. remap numerical codes, excluded vocabs, etc.
                * `tokenizer_config_filtered.json` -- Contains the tokenizer config with only the tokens actually kept in the vocab
                * `vocab.json` -- Maps textualized tokens to integer IDs
                * `datasets/`
                    * `{datetime-1a}/` -- unique datetime for each dataset version
                        * `metadata.json` -- Contains the dataset metadata, e.g. femr extract path, is_debug, etc.
                        * `seq_length_per_patient.json` -- Maps each idx in dataset (i.e. patient) to that example's sequence lengthwhen using this tokenizer version
                    * `{datetime-1b}/`
                        * `...
            * `{datetime-2}/`
                * `...
    * `{tokenizer_name-2}/`
        * `...

**!!** When loading a tokenizer, we will look for an exact match with that tokenizer + dataset's `metadata` attribute to an existing `metadata.json` file. If no match is found, then the relevant files will be recreated from scratch. 

Note that this can take some time! (e.g. 10+ hrs). However, once it is run, the tokenizer will be saved to disk and can be loaded quickly in the future.