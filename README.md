# EHR FM Cookbook

**Quick Links:**
* [Wandb Home](https://wandb.ai/ehr-fm/hf_ehr?nw=nwusermiking98)
* [Wandb Reports](https://wandb.ai/ehr-fm/hf_ehr/reportlist)
* [Experiment Tracker](https://docs.google.com/spreadsheets/u/1/d/1YTQaoaAicntzNqe0jeUoU5yiAAh-Q6UeZBT9yBBf7mc/edit#gid=0)

**Goals:**
1. Build infrastructure to train off-the-shelf HuggingFace models on structured EHR data
2. Measure how each of these modeling choices impacts model performance:
    a. Architecture (bert, gpt, mamba, hyena)
    b. Model size (120M, ...)
    c. Context window length (1k, 4k, 8k, 16k)
    d. Vocab size (...)
    e. Tokenizer choice (DescEmb, CLMBR, Custom)
    f. Tokens (...)
3. Measure "scaling laws" for Foundation Models for EHRs

**Evaluations:**
1. Val/PPL on STARR-OMOP held-out 15% dataset split (canonical FEMR split)
2. AUROC/AUPRC on EHRSHOT benchmark
3. **TODO** -- MIMIC?
3. **TODO** -- EHRSHOT labelers on all of STARR?

## Installation

1. Install packages
```bash
conda create -n hf_env python=3.10 -y
conda activate hf_env
pip install -r requirements.txt
pip install -e .
```

2. [Optional] If you haven't already created your **Tokenizers**, run the following. If you're on Carina, then skip this step.
```bash
cd hf_ehr/scripts/tokenizers
sbatch clmbr.sh # Takes ~5 seconds
sbatch desc.sh # Takes ~30 min
sbatch cookbook.sh # Takes TBD
```

## Quick Start

Launch 4 GPT-base runs on one SLURM node (in parallel), and 4 Mamba runs on another SLURM node (in parallel):

```bash
cd hf_ehr/scripts/carina
sbatch parallel_gpt.sh
sbatch parallel_mamba.sh
```

## Training

We use [Hydra](https://github.com/facebookresearch/hydra) to manage our configurations and [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) for training. 

You can either overwrite the config files in `configs/` or pass in CLI arguments to override the defaults.

There are 3 ways to launch a training run. 

### Easy

Launch multiple runs in parallel on the same SLURM node  (each job gets 1 GPU) using `hf_ehr/scripts/carina/parallel_{model}.sh`:

```bash
cd hf_ehr/scripts/carina

# Launch 4 gpt runs in parallel on the same node. See the file for the specific model versions run.
sbatch parallel_gpt.sh

# Launch 4 bert runs in parallel on the same node. See the file for the specific model versions run.
sbatch parallel_bert.sh

# Launch 4 hyena runs in parallel on the same node. See the file for the specific model versions run.
sbatch parallel_hyena.sh

# Launch 4 mamba runs in parallel on the same node. See the file for the specific model versions run.
sbatch parallel_mamba.sh
```

### Medium

Launch one run on a SLURM node using `hf_ehr/scripts/carina/{model}.sh`:

```bash
cd hf_ehr/scripts/carina

# Launch GPT-2 base model with CLMBRTokenizer, ApproxBatchSampler dataloader, and 2048 context length; force train from scratch and not resume prior run (even if exists)
sbatch gpt.sh base clmbr 2048 approx --is_force_refresh

# Launch Mamba tiny model with CookbookTokenizer, ApproxBatchSampler dataloader, and 16384 context length; resume prior run if exists
sbatch mamba.sh tiny cookbook 16384 approx

# Run BERT-base model with DescTokenizer, ApproxBatchSampler dataloader, and 4096 context length; resume prior run if exists; overwrite the default device assignment to GPU 1; give wandb run a name of `custom`
sbatch bert.sh base desc 4096 approx "+trainer.devices=[1] +logging.wandb.name=custom"
```

General usage:
```bash
sbatch {model}.sh <model_size> <tokenizer> <context_length> <dataloader_mode> [<extra>] [--is_force_refresh] [--is_skip_base]`
```

where...
- `{model}`: str -- Architecture (e.g., `gpt`, `bert`, `mamba`, `hyena`)
- `<model_size>`: str -- Model size (e.g., `base`, `large`, `tiny`, `medium`)
- `<tokenizer>`: str -- Tokenizer to use (e.g., `clmbr`, `cookbook`, `desc`)
- `<context_length>`: int -- Context length (e.g., `1024`, `2048`, `4096`, `8192`, `16384`)
- `<dataloader_mode>`: str -- Dataloader mode (e.g., `batch`, `approx`)
- `[<extra>]`: Optional[str] -- An optional string that will get appended to the end of the `python ../run.py` command verbatim
- `[--is_force_refresh]`: Optional -- An optional flag that triggers a force refresh of the run (i.e., delete the existing run and start from scratch)
- `[--is_skip_base]`: Optional -- An optional flag that skips running `source base.sh`. Useful when running `parallel.sh` and we don't want to reinit the conda environment multiple times

### Advanced

Directly call `run.py`, which allows maximum flexibility for configs. 

See the [Config README](hf_ehr/configs/README.md) for details on all config settings.

```bash
cd hf_ehr/scripts/carina

# Launch gpt with: size=base, dataset=v8, context_length=2048, tokenizer=CLMBRTokenizer, sampler=ApproxBatchSampler, max_tokens_per_batch=16384, use_cuda_devices=2,3, wandb_logging_name=gpt2-custom-run, force_restart_existing_run=True, save_to_path=/share/pi/nigam/mwornow/hf_ehr/cache/runs/bert-test/
python3 ../run.py \
    +data=v8 \
    +trainer=single_gpu \
    +model=gpt2-base \
    +tokenizer=clmbr \
    data.dataloader.mode=approx \
    data.dataloader.approx_batch_sampler.max_tokens=16384 \
    data.dataloader.max_length=2048 \
    model.config_kwargs.n_positions=2048 \
    trainer.devices=[2,3] \
    logging.wandb.name=gpt2-custom-run \
    main.is_force_restart=True \
    main.path_to_output_dir=/share/pi/nigam/mwornow/hf_ehr/cache/runs/bert-test/
```

## Evaluation

1. Identify the path (`<path_to_ckpt>`) to the model checkpoint you want to evaluate.

2. Generate patient representations with your model.

```bash
cd hf_ehr/scripts/eval
sbatch ehrshot.sh <path_to_ckpt>
```

3. In `ehrshot-benchmark`, update `utils.py` so that your model is included in the global constants at the top of the file.

4. Run `7_eval.sh`

```bash
conda activate EHRSHOT_ENV
bash 7_eval.sh --is_use_slurm
```

5. Run `8_make_results_plots.sh`

## Configurations

See the [Config README](hf_ehr/configs/README.md) for details on all config settings (models, training, dataloaders, tokenizers, etc.).

## How To...

### Create a Model

Let's say we want to create a new model called `{model}` of size `{size}`.

1. Create the Hydra config YAML for your model architecture in `hf_ehr/configs/architecture/{model}.yaml`. Copy the contents of `hf_ehr/configs/architecture/bert.yaml` and modify as needed. 

2. Create the Hydra config YAML for your model instantiation in `hf_ehr/configs/models/{model}-{size}.yaml`. Copy the contents of `hf_ehr/configs/models/bert-base.yaml` and modify as needed.

3. Create the model itself by creating a new file `hf_ehr/models/{model}.py`. Copy the contents of `models/bert.py` and modify as needed.

4. Add your model to `hf_ehr/scripts/run.py` above the line **raise ValueError(f"Model `{config.model.name}` not supported.")**

### Create a Tokenizer

See the [Tokenizer README](hf_ehr/tokenizers/README.md) for details on creating tokenizers and how they are stored on the file system.

## Miscellaneous

### Tensorboard:

Launch tensorboard with:
```bash
tensorboard --logdir="experiments/lightning_logs/"
```