# Data

Helper functions for...
* Loading Datasets -- [datasets.py](datasets.py)
* Loading Tokenizers -- [tokenization.py](tokenization.py)

## Supported Formats

### 💊 MEDS

[Please see here](https://github.com/som-shahlab/long_context_clues?tab=readme-ov-file#meds_demo) for instructions for loading MEDS formatted data.

Example command:
```bash
python3 main.py --model llama --size base --tokenizer clmbr --context_length 1024 --dataloader approx --dataset meds_dev --is_run_local --is_force_refresh
```

### 🦴 FEMR

Example command:

```bash
python3 main.py --model llama --size base --tokenizer clmbr --context_length 1024 --dataloader approx --dataset v8 --is_run_local --is_force_refresh
```
