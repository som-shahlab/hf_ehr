# Tokenizers

Scripts for creating tokenizers.

We currently support three different tokenizers:
1. `CLMBRTokenizer` -- identical tokenizer as was used in the CLMBR/MOTOR/EHRSHOT papers
2. `DescTokenizer` -- tokenizer based on Ed Choi's DescEmb/UniHPF work
3. `CookbookTokenizer` -- custom tokenizer for this project

## Quick Start

```bash
# Create CLMBRTokenizer
python3 tokenizers/create_clmbr.py # Takes ~5 seconds

# Create DescTokenizer
python3 tokenizers/create_desc.py  # Takes ~30 mins

# Create CookbookTokenizer
python3 tokenizers/create_cookbook.py # Takes ~10 mins

# Create CookbookTokenizer with top-10k codes
python3 tokenizers/create_cookbook.py --k 10 && python3 tokenizers/create_cookbook_k.py --k 10 # Takes ~1 mins
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

To limit to the top `k` thousand most frequently occurring codes, run:

```bash
python3 create_cookbook_k.py --dataset v8 --k 10 --stat count_occurrences
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

## Tokenizer Config File Format

### `tokenizer_config.json`

The `tokenizer_config.json` file contains the main config with all tokens for this tokenizer. It is a JSON object structured as follows:

* `timestamp` -- str *= datetime.datetime.now().isoformat()* -- An ISO-formatted timestamp of when this config was created
* `metadata` -- Dict[str, Any] -- Arbitrary metadata about the tokenizer, e.g. remap numerical codes, excluded vocabs, etc.
* `tokens` -- List[TokenizerConfigEntry] -- A list of `hf_ehr.config.TokenizerConfigEntry` objects. Please see the [Token Config Entry](#token-config-entry) section below for more details.

**Example:** This is from the `clmbr_v8/tokenizer_config.json`:

```json
{
  "timestamp": "2024-07-24T09:23:17.613609",
  "metadata": {},
  "tokens": [
    {
      "code": "SNOMED/3950001",
      "type": "code",
      "description": null,
      "tokenization": {},
      "stats": [
        {
          "type": "count_occurrences",
          "split": "train",
          "dataset": "v8",
          "count": null
        },
      ]
    },
    {
      "code": "Domain/OMOP generated",
      "type": "code",
      "description": null,
      "tokenization": {},
      "stats": [
        {
          "type": "count_occurrences",
          "split": "train",
          "dataset": "v8",
          "count": null
        },
      ]
    },
    ...
  ]
}
```

## `TokenizerConfigEntry`

Each token is stored as a `TokenizerConfigEntry`. It defines how to map a clinical event => a token. 

The definition for this object is in [hf_ehr/config.py](hf_ehr/config.py).