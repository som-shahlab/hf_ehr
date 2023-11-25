# Databricks notebook source
# MAGIC %%capture
# MAGIC %pip install -r /Workspace/Repos/s0353982@stanfordhealthcare.org/hf_ehr/requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %cd /Workspace/Repos/s0353982@stanfordhealthcare.org/hf_ehr
# MAGIC %pip install -e .

# COMMAND ----------

import os
import wandb
import json
import hydra
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
from pytorch_lightning.utilities import rank_zero_only

from loguru import logger
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from omegaconf import DictConfig, OmegaConf

from hf_ehr.data.datasets import FEMRDataset, FEMRTokenizer
from hf_ehr.models.bert import BERTLanguageModel
from hf_ehr.models.gpt import GPTLanguageModel
from hf_ehr.trainer.loaders import load_datasets, load_dataloaders


# COMMAND ----------

!python3 hf_ehr/scripts/run.py \
    +models=bert \
    data.dataloader.batch_size=4 \
    trainer.accumulate_grad_batches=16 \
    data.dataloader.n_workers=10 \
    trainer.devices=[0,1,2,3] \
    model.config_kwargs.num_hidden_layers=12 \
    model.config_kwargs.num_attention_heads=12 \
    model.config_kwargs.hidden_size=768 \
    main.path_to_output_dir=/FileStore/michael-hf_ehr/cache/runs/bert-base-v8/ \
    data.dataset.path_to_femr_extract=/FileStore/michael-hf_ehr/cache/femr/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_02_08_extract_v8_no_notes \
    data.tokenizer.path_to_code_2_int=/FileStore/michael-hf_ehr/cache/tokenizer_v8/code_2_int.json \
    data.tokenizer.path_to_code_2_count=/FileStore/michael-hf_ehr/cache/tokenizer_v8/code_2_count.json \
    +data.tokenizer.min_code_count=10 \
    logging.wandb.name=bert-base-v8-test

# COMMAND ----------

wandb.init()

# COMMAND ----------

wandb.run.id

# COMMAND ----------

os.environ['WANDB_API_KEY'] = ''

# COMMAND ----------


