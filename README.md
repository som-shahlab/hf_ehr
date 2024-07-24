# EHR FM Cookbook

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
python3 tokenizers/create_clmbr.py
python3 tokenizers/create_desc.py
python3 tokenizers/create_cookbook.py
```

## Quick Start

We use [Hydra](https://github.com/facebookresearch/hydra) to manage our configurations and [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) for training. 

You can either overwrite the config files in `configs/` or pass in CLI arguments to override the defaults.

### Training

See [Scripts README](scripts/carina/README.md) for more details.

There are 3 ways to launch a training run. 

**(A) Easy** -- Launch multiple runs in parallel on the same SLURM node using `scripts/carina/parallel_{model}.sh`:

```bash
cd scripts/carina

# Launch 4 gpt runs in parallel on the same node. See the file for the specific model versions run.
sbatch parallel_gpt.sh

# Launch 4 bert runs in parallel on the same node. See the file for the specific model versions run.
sbatch parallel_bert.sh

# Launch 4 hyena runs in parallel on the same node. See the file for the specific model versions run.
sbatch parallel_hyena.sh

# Launch 4 mamba runs in parallel on the same node. See the file for the specific model versions run.
sbatch parallel_mamba.sh
```

**(B) Medium** -- Launch one run on a SLURM node using `scripts/carina/{model}.sh`:

```bash
cd scripts/carina

# Launch gpt with: size=base, context_length=1024, tokenizer=CLMBRTokenizer, sampler=ApproxBatchSampler
sbatch gpt.sh base clmbr 1024 approx

# Launch mamba with: size=large, context_length=16384, tokenizer=CookbookTokenizer, sampler=ApproxBatchSampler
sbatch mamba.sh large cookbook 16384 approx

# Launch bert with: size=large, context_length=4096, tokenizer=DescTokenizer, sampler=ApproxBatchSampler; set CUDA_VISIBLE_DEVICES=2 and set wandb logging name to "test"
sbatch bert.sh large desc 4096 approx "+trainer.devices=[2] logging.wandb.name=test"
```

**(C) Advanced** -- Directly call `run.py`, which allows maximum flexibility for configs:

```bash
cd scripts/carina

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

### Evaluation

1. In `hf_ehr`, run the following to generate patient representations:

```bash
conda activate hf_env
export MODEL_NAME=gpt2-base-v8

python3 ehrshot.py \
    --path_to_database /share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/femr/extract \
    --path_to_labels_dir /share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/custom_benchmark \
    --path_to_features_dir /share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/custom_hf_features \
    --path_to_models_dir /share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/models \
    --model $MODEL_NAME \
    --embed_strat mean \
    --chunk_strat last \
    --is_force_refresh
```

2. In `ehrshot-benchmark`, update `utils.py` so that your model is included in the global constants at the top of the file.

3. Run `7_eval.sh`

```bash
conda activate EHRSHOT_ENV
bash 7_eval.sh
```

4. Run `8_make_results_plots.sh`

## Configurations

See the [Config README](configs/README.md) for details on all config settings.

## Guide

### Create a Model

Let's say we want to create a new model called `{model}` of size `{size}`.

1. Create the Hydra config YAML for your model architecture in `configs/architecture/{model}.yaml`. Copy the contents of `configs/architecture/bert.yaml` and modify as needed. 

2. Create the Hydra config YAML for your model instantiation in `configs/models/{model}-{size}.yaml`. Copy the contents of `configs/models/bert.yaml` and modify as needed.

Second, create the model itself by creating a new file `models/{model}.py`. Copy the contents of `models/bert.py` and modify as needed.


### Create a Tokenizer

See the [Tokenizer README](tokenizers/README.md) for details on creating tokenizers and how they are stored on the file system.

## Miscellaneous

### Tensorboard:

Launch tensorboard with:
```bash
tensorboard --logdir="experiments/lightning_logs/"
```