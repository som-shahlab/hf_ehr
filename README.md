# Training Long Context Models on EHR Data

This repo contains code and pretrained models for the [**Context Clues paper**](https://arxiv.org/abs/2412.16178). It is designed to enable **training any model on HuggingFace on structured EHR data.** It comes with Hydra configs + Wandb logging + PyTorch Lightning distributed training support.

It currently supports EHR data defined using the [**MEDS data standard**](https://github.com/Medical-Event-Data-Standard/) or [**FEMR package**](https://github.com/som-shahlab/femr).

### 📖 Table of Contents

1. 🤗 [Pretrained HuggingFace Models](#models)
1. 📀 [Installation](#installation)
1. 🚀 [Quick Start](#quick_start)
1. 🏋️‍♀️ [Training](#training)
1. 📊 [Evaluation](#evaluation)
1. 💊 [MEDS Demo](#meds_demo)
1. Ⓜ️ [Merative/Truven/MarketScan Demo](#truven_demo)
1. ℹ️ [Other](#other)
1. 🎓 [Citation](#citation)

<a name="models" />

## 🤗 Pretrained HuggingFace Models

Please see our [HuggingFace Collection](https://huggingface.co/collections/StanfordShahLab/context-clues-models-6757f893f6a2918c7ab809f1) to download the following models pretrained from scratch on 2 billion tokens of deidentified structured EHR data:

| Model | Context Lengths |
| ----- | ------------- |
| gpt | [512](https://huggingface.co/StanfordShahLab/gpt-base-512-clmbr), [1024](https://huggingface.co/StanfordShahLab/gpt-base-1024-clmbr), [2048](https://huggingface.co/StanfordShahLab/gpt-base-2048-clmbr), [4096](https://huggingface.co/StanfordShahLab/gpt-base-4096-clmbr) |
| llama | [512](https://huggingface.co/StanfordShahLab/llama-base-512-clmbr), [1024](https://huggingface.co/StanfordShahLab/llama-base-1024-clmbr), [2048](https://huggingface.co/StanfordShahLab/llama-base-2048-clmbr), [4096](https://huggingface.co/StanfordShahLab/llama-base-4096-clmbr) |
| mamba | [1024](https://huggingface.co/StanfordShahLab/mamba-tiny-1024-clmbr), [4096](https://huggingface.co/StanfordShahLab/mamba-tiny-4096-clmbr), [8192](https://huggingface.co/StanfordShahLab/mamba-tiny-8192-clmbr), [16384](https://huggingface.co/StanfordShahLab/mamba-tiny-16384-clmbr) |
| hyena | [1024](https://huggingface.co/StanfordShahLab/hyena-large-1024-clmbr), [4096](https://huggingface.co/StanfordShahLab/hyena-large-4096-clmbr), [8192](https://huggingface.co/StanfordShahLab/hyena-large-8192-clmbr), [16384](https://huggingface.co/StanfordShahLab/hyena-large-16384-clmbr) |

Here's a quick tutorial on how to use these models directly in your own code (i.e. outside of this repo's infra):

```python
from transformers import AutoModelForCausalLM
from hf_ehr.data.tokenization import CLMBRTokenizer
from hf_ehr.config import Event
from typing import List, Dict
import torch

####################################
# 1. Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("StanfordShahLab/gpt-base-512-clmbr")
tokenizer = CLMBRTokenizer.from_pretrained("StanfordShahLab/gpt-base-512-clmbr")

####################################
# 2. Define patient as sequence of `Event` objects. Only `code` is required.
patient: List[Event] = [
    Event(code='SNOMED/3950001', value=None, unit=None, start=None, end=None, omop_table=None),
    Event(code='Gender/F', value=None, unit=None, start=None, end=None, omop_table=None),
    Event(code='Ethnicity/Hispanic', value=None, unit=None, start=None, end=None, omop_table=None),
    Event(code='SNOMED/609040007', value=None, unit=None, start=None, end=None, omop_table=None),
    Event(code='LOINC/2236-8', value=-3.0, unit=None, start=None, end=None, omop_table=None),
    Event(code='SNOMED/12199005', value=26.3, unit=None, start=None, end=None, omop_table=None),        
]

####################################
# 3. Tokenize patient
batch: Dict[str, torch.Tensor] = tokenizer([ patient ], add_special_tokens=True, return_tensors='pt')
# > batch = {
#     'input_ids': tensor([[ 5, 0, 7, 9, 27, 2049, 6557, 22433, 1]]), 
#     'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0]]), 
#     'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]])
# }
textual_tokens: List[str] = tokenizer.convert_events_to_tokens(patient)
# > textual_tokens = ['SNOMED/3950001', 'Gender/F', 'Ethnicity/Hispanic', 'SNOMED/609040007', 'LOINC/2236-8 || None || -1.7976931348623157e+308 - 4.0', 'SNOMED/12199005 || None || 26.0 - 28.899999618530273']

####################################
# 4. Run model
logits = model(**batch).logits
# > logits.shape = torch.Size([1, 9, 39818])

####################################
# 5. Get patient representation for finetuning (usually we choose the last token's logits)
representation = logits[:, -1, :]
```
<a name="installation" />

## 📀 Installation

Direct install:
```bash
pip install hf-ehr
```

For faster Mamba runs, install:
```bash
pip install mamba-ssm causal-conv1d
```

Development install:
```bash
conda create -n hf_env python=3.10 -y
conda activate hf_env
pip install -r requirements.txt --no-cache-dir
pip install -e .

# [Optional] If you haven't already created your **Tokenizers**, run the following. If you're on Carina, then skip this step.
cd hf_ehr/scripts/tokenizers
sbatch clmbr.sh # Takes ~5 seconds
sbatch desc.sh # Takes ~30 min
sbatch cookbook.sh # Takes many hours
```

<a name="quick_start"/>

## 🚀 Quick Start

Launch a GPT training run with the ability to configure common hyperparameters (using `main.py`)

```bash
cd hf_ehr/scripts/carina
python3 main.py --model gpt2 --size base --tokenizer clmbr --context_length 1024 --dataloader approx --dataset v8 --is_run_local --is_force_refresh
```

Launch a Llama run on a MEDS dataset with more customization over configs (using `run.py`):
```bash
cd hf_ehr/scripts/carina
python3 run.py \
    +data=meds_mimic4_demo \
    +trainer=single_gpu \
    +model=llama-base \
    +tokenizer=clmbr \
    data.dataloader.mode=approx \
    data.dataloader.approx_batch_sampler.max_tokens=16384
```

To launch 4 GPT-base runs on one SLURM node (in parallel), and 4 Mamba runs on another SLURM node (in parallel):

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

### EHRSHOT

How to use this repo with EHRSHOT.

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

1. Generate your model's AUROC/AUPRC results by running `7_eval.sh`:

```bash
# cd to ehrshot-benchmark/ehrshot/bash_scripts/ directory
bash 7_eval.sh --is_use_slurm
```

#### 3. Generate EHRSHOT Plots

This all occurs within the `ehrshot-benchmark` repo.

1. Generate plots by running: `8_make_results_plots.sh`. You might need to modify the `--model_heads` parameter in the file before running to specify what gets included in your plots.

```bash
# cd to ehrshot-benchmark/ehrshot/bash_scripts/ directory
bash 8_make_results_plots.sh
```

<a name="meds_demo"/>

## 💊 MEDS Demo

We support training and inference on [MEDS formatted datasets](https://github.com/Medical-Event-Data-Standard/meds/). 

Here is a quick tutorial using the publicly available **MIMIC-IV demo dataset** (inspired by [this tutorial](https://colab.research.google.com/drive/1R1LrDIzhQyWldQWM0lyfjeF_n9I_iZT3)).

1. **Download** the [MIMIC-IV demo dataset](https://physionet.org/content/mimiciv-demo/1.4/) from PhysioNet.

```bash
export PATH_TO_DOWNLOAD=mimic4_demo
export PATH_TO_MEDS=meds_mimic4_demo
export PATH_TO_MEDS_READER=meds_mimic4_demo_reader

!wget -q -r -N -c --no-host-directories --cut-dirs=1 -np -P $PATH_TO_DOWNLOAD https://physionet.org/files/mimic-iv-demo/2.2/
```

2. **Convert** the MIMIC-IV demo dataset to [**MEDS format**](https://github.com/Medical-Event-Data-Standard/meds/).

```bash
rm -rf $PATH_TO_MEDS 2>/dev/null
meds_etl_mimic $PATH_TO_DOWNLOAD $PATH_TO_MEDS
```

3. **Convert** the MEDS dataset into a [**MEDS Reader Database**](https://github.com/som-shahlab/meds_reader) (to enable faster data ingestion during training).

```bash
rm -rf $PATH_TO_MEDS_READER 2>/dev/null
meds_reader_convert $PATH_TO_MEDS $PATH_TO_MEDS_READER --num_threads 4
```

4. **Verify** everything worked.

```bash
meds_reader_verify $PATH_TO_MEDS $PATH_TO_MEDS_READER
```

5. **Create train/val/test splits** (80/10/10) by running the below Python script:
```bash
cd hf_ehr/scripts/datasets
python split_meds_dataset.py --path_to_meds_reader $PATH_TO_MEDS_READER --train_split_size 0.8 --val_split_size 0.1
```

6. **Create** a **Hydra config** for your dataset.

```bash
cp hf_ehr/configs/data/meds_mimic4_demo.yaml hf_ehr/configs/data/meds_mimic4_demo_custom.yaml
sed -i 's|/share/pi/nigam/mwornow/mimic-iv-demo-meds-reader|$PATH_TO_MEDS_READER|g' hf_ehr/configs/data/meds_mimic4_demo_custom.yaml
```

7. **Train** a tokenizer on the dataset. Limit our vocabulary to the top-$k$ most frequently occurring codes.

```bash
cd hf_ehr/tokenizers
python create_cookbook.py --dataset meds_mimic4_demo --n_procs 5 --chunk_size 10000 --is_force_refresh
python create_cookbook_k.py --dataset meds_mimic4_demo --k 32 --stat count_occurrences
```

8. **Train** a **Llama model** on the dataset.
- You need to exchange line 315 in `scripts/carina/main.py`, with your desired output dir.
- By default, this uses WandB to track the run, please configure it beforehand by calling `wandb init` and then changing `scripts/run.py` at line 294 (and possibly elsewhere) entity and project.

```bash
cd hf_ehr/scripts/carina
python3 main.py --model llama --size base --tokenizer clmbr --context_length 1024 --dataloader approx --dataset meds_mimic4_demo_custom --is_run_local --is_force_refresh
``` 


<a name="truven_demo"/>

## Ⓜ️ Merative/Truven/MarketScan Demo

We support training and inference on the 2017 Merative MarketScan Commercial Claims and Encounters Database (OMOP CDMv5 formatted) dataset, aka "Truven" or "MarketScan".

1. **Download** the [Merative OMOP CDMv5 dataset](https://console.cloud.google.com/storage/browser/truven_backup/TRUVEN_CDMv5;tab=objects?authuser=1&project=som-nero-phi-nigam-starr). *Note: This takes ~10 mins to download and takes up 347 GB of space.*

```bash
export PATH_TO_DOWNLOAD=truven-omop
export PATH_TO_MEDS=truven-meds
export PATH_TO_MEDS_READER=truven-meds-reader
gsutil -m cp -r gs://truven_backup/TRUVEN_CDMv5 $PATH_TO_DOWNLOAD
```

2. **Convert** the Truven OMOP CDMv5 dataset to [**MEDS format**](https://github.com/Medical-Event-Data-Standard/meds/). *Note: This takes ~4.25 hrs to run and takes up 698MB of space.*

```bash
meds_etl_omop $PATH_TO_DOWNLOAD $PATH_TO_MEDS
```

3. **Convert** the MEDS dataset into a [**MEDS Reader Database**](https://github.com/som-shahlab/meds_reader) (to enable faster data ingestion during training). *Note: This takes ~15 mins to run and takes up 26GB of space.*

```bash
meds_reader_convert $PATH_TO_MEDS $PATH_TO_MEDS_READER --num_threads 10
meds_reader_verify $PATH_TO_MEDS $PATH_TO_MEDS_READER
```

4. **Create train/val/test splits** (80/10/10) by running the below Python script. *Note: This takes ~1 min to run.*
```bash
cd hf_ehr/scripts/datasets
python split_meds_dataset.py --path_to_meds_reader $PATH_TO_MEDS_READER --train_split_size 0.8 --val_split_size 0.1
```

5. **Train** a tokenizer on the dataset. Limit our vocabulary to the top-$k$ most frequently occurring codes. **TODO**

```bash
cd hf_ehr/tokenizers
python create_cookbook.py --dataset truven --n_procs 5 --chunk_size 10000 --is_force_refresh
python create_cookbook_k.py --dataset truven --k 32 --stat count_occurrences
```

6. **Train** a **Llama model** on the dataset using 2 GPUs. *Note: This takes ~5 hrs per epoch with 2 H100's.*

```bash
cd hf_ehr/scripts/carina
python3 main.py --model llama --size base --tokenizer clmbr --context_length 512 --dataloader batch --dataset truven --trainer multi_gpu_2 --is_run_local --is_force_refresh
``` 


<a name="other" />

## ℹ️ Other

### Based
To get the **based** model to run, you need to do the following installations on an A100 or above node:

```bash
pip install -v \
    --disable-pip-version-check \
    --no-cache-dir \
    --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" \
    'git+https://github.com/NVIDIA/apex@b496d85'  --no-cache-dir

pip install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir

# Install FLA triton kernel
pip install -U git+https://github.com/sustcsonglin/flash-linear-attention

pip install 'git+https://github.com/HazyResearch/flash-attention@v2.5.2' --no-build-isolation --no-cache-dir
pip install 'git+https://github.com/HazyResearch/flash-attention@v2.5.2#subdirectory=csrc/fused_dense_lib'  --no-build-isolation --no-cache-dir
pip install 'git+https://github.com/HazyResearch/flash-attention@v2.5.2#subdirectory=csrc/layer_norm' --no-build-isolation --no-cache-dir

git clone git@github.com:HazyResearch/based.git
cd based
pip install -e . --no-cache-dir
```

### 🤖 Creating a Model

Let's say we want to create a new model called `{model}` of size `{size}`.

1. Create the Hydra config YAML for your model architecture in `hf_ehr/configs/architecture/{model}.yaml`. Copy the contents of `hf_ehr/configs/architecture/bert.yaml` and modify as needed. 

2. Create the Hydra config YAML for your model instantiation in `hf_ehr/configs/models/{model}-{size}.yaml`. Copy the contents of `hf_ehr/configs/models/bert-base.yaml` and modify as needed.

3. Create the model itself by creating a new file `hf_ehr/models/{model}.py`. Copy the contents of `models/bert.py` and modify as needed.

4. Add your model to `hf_ehr/scripts/run.py` above the line `raise ValueError(f"Model `{config.model.name}` not supported.")`

### ✂️ Creating a Tokenizer

See the [Tokenizer README](hf_ehr/tokenizers/README.md) for details on creating tokenizers and how they are stored on the file system.

### 🤗 Uploading a Model to Hugging Face

See the [Hugging Face README](hf_ehr/scripts/huggingface/README.md) for details on uploading models to Hugging Face.

<a name="citation" />

### 📦 Package for PyPi

```bash
git add . && git commit -m "New version"
make release
```

### MEDS-DEV

First, create a tokenizer from the MEDS extract. This takes 834 seconds.

```bash
cd hf_ehr/tokenizers
python create_cookbook.py --dataset meds_dev --n_procs 5 --chunk_size 10000 --is_force_refresh
```

## 🎓 Citation

If you found this work useful, please consider citing it:

```
@article{wornow2024contextclues,
      title={Context Clues: Evaluating Long Context Models for Clinical Prediction Tasks on EHRs}, 
      author={Michael Wornow and Suhana Bedi and Miguel Angel Fuentes Hernandez and Ethan Steinberg and Jason Alan Fries and Christopher Ré and Sanmi Koyejo and Nigam H. Shah},
      year={2024},
      eprint={2412.16178},
      url={https://arxiv.org/abs/2412.16178}, 
}
```
