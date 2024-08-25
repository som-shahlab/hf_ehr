# EHR FM Cookbook

**📖 Table of Contents**

1. 📀 [Installation](#installation)
1. 🚀 [Quick Start](#quick_start)
1. 🏋️‍♀️ [Training](#training)
1. 📊 [Evaluation](#evaluation)
1. ℹ️ [Other](#other)

**🎯 Goals:**
1. Build infrastructure to train off-the-shelf HuggingFace models on structured EHR data
2. Measure how each of these modeling choices impacts model performance (in terms of **val ppl**, **EHRSHOT**, **MIMIC-IV**):
    1. Architecture (bert, gpt, mamba, hyena)
    2. Model size (120M, ...)
    3. Context window length (1k, 4k, 8k, 16k)
    4. Vocab size (32k, 64, 128k)
    5. Tokenizer choice (DescEmb, CLMBR, Custom)
    6. Finetuning (frozen, full, 1-layer)
3. Some sort of generative synthetic patient timeline generations

**🔗 External Links:**
* [Wandb Home](https://wandb.ai/ehr-fm/hf_ehr?nw=nwusermiking98)
* [Wandb Reports](https://wandb.ai/ehr-fm/hf_ehr/reportlist)
* [Experiment Tracker](https://docs.google.com/spreadsheets/u/1/d/1YTQaoaAicntzNqe0jeUoU5yiAAh-Q6UeZBT9yBBf7mc/edit#gid=0)


<a name="installation" />

## 📀 Installation

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
<a name="quick_start"/>

## 🚀 Quick Start

Launch 4 GPT-base runs on one SLURM node (in parallel), and 4 Mamba runs on another SLURM node (in parallel):

```bash
cd hf_ehr/scripts/carina

# GPT runs
sbatch parallel_gpt.sh

# Mamba runs
sbatch parallel_mamba.sh
```

<a name="training" />

## 🏋️‍♀️ Training

We use [Hydra](https://github.com/facebookresearch/hydra) to manage our configurations and [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) for training. 

You can either overwrite the config files in `configs/` or pass in CLI arguments to override the defaults.

There are 3 ways to launch a training run. 

### Easy Mode

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

### Medium Mode

Launch one run on a SLURM node using `hf_ehr/scripts/carina/{model}.sh`:

```bash
cd hf_ehr/scripts/carina

# Launch GPT-2 base model on v8 dataset with CLMBRTokenizer, ApproxBatchSampler dataloader, and 2048 context length; force train from scratch and not resume prior run (even if exists)
python3 main.py --model gpt2 --size base --tokenizer clmbr --context_length 2048 --dataloader approx --dataset v8 --is_force_refresh

# Launch Mamba tiny model on v8 dataset with CookbookTokenizer, ApproxBatchSampler dataloader, and 16384 context length; resume prior run if exists
python3 main.py --model mamba --size tiny --tokenizer cookbook --context_length 16384 --dataloader approx --dataset v8

# Launch BERT-base model on v8 dataset with DescTokenizer, ApproxBatchSampler dataloader, and 4096 context length; resume prior run if exists; overwrite the default device assignment to GPU 1; give wandb run a name of `custom`
python3 main.py --model bert --size base --tokenizer desc --context_length 4096 --dataloader approx --dataset v8 --extra "+trainer.devices=[1] logging.wandb.name=custom"

# Run locally a GPT-2 large model on v8 AllTokens dataset with CLMBRTokenizer, ApproxBatchSampler dataloader, and 1024 context length
python3 main.py --model gpt2 --size large --tokenizer clmbr --context_length 2048 --dataloader approx --dataset v8-alltokens --is_run_local

# Launch Mamba tiny model on v8 dataset with CookbookTokenizer, ApproxBatchSampler dataloader, and 16384 context length; resume prior run if exists; run on 8 H100's
python3 main.py --model mamba --size tiny --tokenizer cookbook --context_length 16384 --dataloader approx --dataset v8 --partitions nigam-h100 --extra "trainer=multi_gpu trainer.devices=[0,1,2,3,4,5,6,7]"
```

General usage:
```bash
python3 main.py --model <model> --size <size> --tokenizer <tokenizer> --context_length <context_length> --dataloader <dataloader> --dataset <dataset> [--extra <extra>] [--partitions <partitions>] [--is_force_refresh] [--is_skip_base] [--is_run_local]
```

where...
- `<model>`: str -- Architecture to use. Choices are `gpt`, `bert`, `hyena`, `mamba`
- `<size>`: str -- Model size to use. Choices are `tiny`, `small`, `base`, `medium`, `large`, `huge`
- `<tokenizer>`: str -- Tokenizer to use. Choices are `clmbr`, `desc`, `cookbook`
- `<context_length>`: int -- Context length to use
- `<dataloader>`: str -- Dataloader to use. Choices are `approx`, `exact`
- `<dataset>`: str -- Dataset to use. Choices are `v8`, `v8-alltokens`, `v9`, `v9-alltokens`
- `[--extra <extra>]`: Optional[str] -- An optional string that will get appended to the end of the `python ../run.py` command verbatim
- `[--partitions <partitions>]`: Optional[str] -- An optional string that specifies the partitions to use. Defaults to `nigam-v100,gpu` for gpt2 and BERT, and `nigam-h100,nigam-a100` for HYENA and MAMBA
- `[--is_force_refresh]`: Optional -- An optional flag that triggers a force refresh of the run (i.e., delete the existing run and start from scratch)
- `[--is_skip_base]`: Optional -- An optional flag that skips running `source base.sh`. Useful when running `parallel.sh` and we don't want to reinit the conda environment multiple times
- `[--is_run_local]`: Optional -- An optional flag that runs the script locally as `python run.py` instead of as a SLURM `sbatch` command

### Advanced Mode

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

### How to Configure Runs

See the [Config README](hf_ehr/configs/README.md) for details on all config settings (models, training, dataloaders, tokenizers, etc.).


<a name="evaluation"/>
    
## 📊 Evaluation

**Evaluations:**
1. Val/PPL on STARR-OMOP held-out 15% dataset split (canonical FEMR split)
2. AUROC/AUPRC on EHRSHOT tasks
3. AUROC/AUPRC on MIMIC-IV tasks
3. **TODO** -- EHRSHOT labelers on all of STARR?

### MIMIC-IV

TODO

### EHRSHOT

#### 1. Generate Patient Representations
This all occurs within the `hf_ehr` repo.

1. Identify the path (`<path_to_ckpt>`) to the model checkpoint you want to evaluate.

2. Generate patient representations with your model. This will create a folder in `/share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/models` for this model checkpoint.


```bash
cd hf_ehr/scripts/eval/
sbatch ehrshot.sh <path_to_ckpt>
```

#### 2. Generate EHRSHOT Results

This all occurs within the `ehrshot-benchmark` repo.

3. In `ehrshot-benchmark`, update `ehrshot/utils.py` so that your model is included in the global constants at the top of the file. Specifically, create a new entry in the `MODEL_2_INFO` dictionary at the top of the file. The **key** should be the name of the folder that's created for your model in `/share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/models`, and the **value** should be similar to existing entries.

4. Generate your model's AUROC/AUPRC results by running `7_eval.sh`:

```bash
# cd to ehrshot-benchmark/ directory

conda activate EHRSHOT_ENV
cd ehrshot/bash_scripts/

bash 7_eval.sh --is_use_slurm
```

#### 3. Generate EHRSHOT Plots

This all occurs within the `ehrshot-benchmark` repo.

5. Generate plots by running: `8_make_results_plots.sh`. You might need to modify the `--model_heads` parameter in the file before running to specify what gets included in your plots.

```bash
# cd to ehrshot-benchmark/ directory

conda activate EHRSHOT_ENV
cd ehrshot/bash_scripts/
bash 8_make_results_plots.sh
```

<a name="other" />

## ℹ️ Other

### Creating a Model

Let's say we want to create a new model called `{model}` of size `{size}`.

1. Create the Hydra config YAML for your model architecture in `hf_ehr/configs/architecture/{model}.yaml`. Copy the contents of `hf_ehr/configs/architecture/bert.yaml` and modify as needed. 

2. Create the Hydra config YAML for your model instantiation in `hf_ehr/configs/models/{model}-{size}.yaml`. Copy the contents of `hf_ehr/configs/models/bert-base.yaml` and modify as needed.

3. Create the model itself by creating a new file `hf_ehr/models/{model}.py`. Copy the contents of `models/bert.py` and modify as needed.

4. Add your model to `hf_ehr/scripts/run.py` above the line **raise ValueError(f"Model `{config.model.name}` not supported.")**

### Creating a Tokenizer

See the [Tokenizer README](hf_ehr/tokenizers/README.md) for details on creating tokenizers and how they are stored on the file system.

### Tensorboard:

Launch tensorboard with:
```bash
tensorboard --logdir="experiments/lightning_logs/"
```
