# How to Run

Usage: `sbatch gpt2.sh <model_size> <tokenizer> <context_length> <dataloader_mode> [<extra>] [--is_force_refresh]`

where...
- `<model_size>` is the model size (e.g., `base`, `large`)
- `<tokenizer>` is the tokenizer to use (e.g., `clmbr`, `femr`)
- `<context_length>` is the context length (e.g., `1024`, `2048`, `4096`, `8192`)
- `<dataloader_mode>` is the dataloader mode (e.g., `batch`, `approx`)
- `[<extra>]` is an optional string that will get appended to the end of the `python ../run.py` command verbatim
- `--is_force_refresh` is an optional flag to force refresh the run (i.e., delete the existing run and start from scratch)

Examples:

```sh
# Run GPT-2 base model with CLMBR tokenizer, batch dataloader, and 8192 context length; force train from scratch and not resume prior run (even if exists)
sbatch gpt2.sh base clmbr 8192 batch --is_force_refresh

# Run GPT-2 large model with FEMR tokenizer, approxbatchsampler dataloader, and 1024 context length; resume prior run if exists
sbatch gpt2.sh large femr 1024 approx

# Run GPT-2 large model with FEMR tokenizer, approxbatchsampler dataloader, and 1024 context length; resume prior run if exists; overwrite the default device assignment to GPU 1
sbatch gpt2.sh large femr 1024 approx "+trainer.devices=[1]"
```