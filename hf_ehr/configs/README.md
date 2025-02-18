# Configs

This directory contains Hydra config files for our PyTorch Lightning infrastructure.

The various folders are organized as follows:

## `config.yaml`

Default settings for all Hydra configs. Anything in this file can be overridden by a more specific config file or CLI argument.

**Arguments:** Default values shown in *italics*

* `main`
    * `seed`: int *= 1* -- Random seed for training
    * `path_to_output_dir`: str *= /share/pi/nigam/mwornow/hf_ehr/cache/runs/${now:%Y-%m-%d_%H-%M-%S}/* -- Path to the directory where the outputs will be saved
    * `is_force_restart`: bool *= False* -- If True, delete existing checkpoint and restart from scratch
* `hydra`
    * `run`
        * `dir`: str *={main.path_to_output_dir}* -- Path to the directory where the outputs will be saved
* `callbacks`
    * `early_stopping`
        * `metric_mode`: str *= min* -- If we want to min/max the monitored quantity
        * `patience`: int *= 3* -- Number of epochs with no improvement after which training will be stopped
    * `model_checkpointing`
        * `save_top_k_val_loss`: int *= 1* -- Save the top **K** best models according to val/loss. Checks every `save_most_recent_every_n_train_steps` steps.
        * `save_most_recent_k`: int *= 1* -- Save the most recent **K** models every `save_most_recent_every_n_train_steps` steps
        * `save_most_recent_every_n_train_steps`: int *= 10_000* -- Save model every **N** global steps; useful for resuming training after a crash
        * `every_n_train_steps`: int *= 30_000* -- Save model every **N** global steps; persists permanently
        * `every_n_train_nonPAD_tokens`: int *= 300_000_000* -- Save model every **N** nonPAD training tokens seen; persists permanently
        * `every_n_flops`: int *= 10_000_000_000_000_000* -- Save model every **N** FLOPs; persists permanently
        * `is_run_eval_on_checkpoint`: bool *= True* -- If TRUE, then log validation metrics when saving model checkpoints for: `every_n_train_nonPAD_tokens`, `every_n_flops`; Note that this slows things down a bit but is good for clean wandb logging plots
* `data`
    * `dataset`
        * `name`: str *=FEMRDataset* -- Name of class of dataset from [hf_ehr/data/datasets.py](../data/datasets.py) that this dataset is initialized from
        * `path_to_femr_extract`: str *=/share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_02_08_extract_v8_no_notes* -- Path to FEMR extract
        * `is_debug`: bool *= False*-- If True, use a small subset of the data for debugging
    * `dataloader`
        * `mode`: str *= approx* -- To avoid changing the config file for each run, specify the mode and keep both batch_size and approx_batch_sampler
        * `batch_size`: int *= 4* -- Batch size to be used. [note: ignored if `data.dataloader.mode=approx`]
        * `approx_batch_sampler`
            * `max_tokens`: int *= 4_096* -- Max tokens in batch to allow [note: ignored if `data.dataloader.mode=batch`]
            * `bucket_size`: int *= 100* -- Bucket size
            * `is_random_shuffle_across_buckets`: bool *= True* -- If TRUE, then shuffle buckets
            * `is_random_shuffle_within_buckets`: bool *= True* -- If TRUE, then shuffle within buckets
        * `n_workers`: int *= 4* -- Number of data loader workers to use.
        * `max_length`: int *= 4* -- Maximum sequence length that a patient's timeline will get truncated to. !! Make sure to override this if you set the context length of the model to be larger, otherwise the model will only see datapoints with `length <= data.dataloader.max_length` !!
        * `is_truncation_random`: bool *= True* -- If TRUE, then truncate patient timelines at random locations; If FALSE, always do right-hand side truncation
* `trainer`
    * `accumulate_grad_batches`: int *= 4* -- Accumulated gradients runs K small batches of size `data.dataloader.batch_size` before doing a backwards pass.
    * `gradient_clip_value`: float *= 1.0* -- Value for gradient clipping
    * `gradient_clip_algorithm`: str *= norm* -- Whether to clip the gradients based on their 'norm' or 'value'
    * `devices`: List[int] *= 0,1,2,3* -- List of devices to run on
    * `distributed_backend`: str *= ddp* -- Supports: dp, ddp, ddp2, fsdp, deepspeed
    * `min_epochs`: int *= 1* -- Limits training to a minimum number of epochs
    * `max_epochs`: int *= 20* -- Limits training to a max number number of epochs
    * `limit_train_batches`: Optional[Union[int, float]] *= null* -- Limits training to `N` batches if `int`, or `N%` of batches if `float`
    * `limit_val_batches`: Optional[Union[int, float]] *= null* -- Limits val to `N` batches if `int`, or `N%` of batches if `float`
    * `val_check_interval`: Union[int, float] *= 0.45* -- How often we check the validation set within an epoch - `int` is every N batches and `float` is a fraction of the training dataset
    * `check_val_every_n_epoch`: int *= 1* -- Log val at end of every epoch
    * `optimizer`
        * `type`: str *= AdamW* -- Name of optimizer to use
        * `betas`: Tuple[float, float] *= (0.9, 0.95)* -- Adam specific parameters
        * `weight_decay`: float *= 0.1* -- weight decay
        * `lr`: float *= 2e-4* -- Peak learning rate
    * `scheduler`
        * `num_warmup_steps`: int *= 40_000* -- Number of warmup local steps (i.e. non-global) from `initial_lr` -> `trainer.optimizer.lr`
        * `num_decay_steps`: int *= 4_000_000* -- Number of decay local steps (i.e. non-global) from `trainer.optimizer.lr` -> `final_lr`
        * `initial_lr`: float *= 1e-6* -- Initial learning rate
        * `final_lr`: float *= 1e-5* -- Final learning rate
* `logging`
    * `wandb`
        * `is_wandb`: bool *= True* -- If TRUE, then log to wandb
        * `name`: Optional[str] *= null* -- Name of wandb run; If NULL, then wandb will auto-generate a random name
        * `entity`: str *= ehr-fm* -- Name of wandb organization
        * `project`: str *= hf_ehr* -- Name of wandb project
        * `is_force_create_wandb_run_from_scratch`: bool *= False* -- If FALSE and resuming from a checkpoint, then resume the existing wandb run from the last step in that model's checkpoint; If TRUE, then create a new wandb run from scratch (even if resuming from a checkpoint); Setting to TRUE is helpful for quick debugging b/c resuming logging from an existing run can be slow (wandb needs to re-upload all data from previous run)
    * `mlflow`
        * `is_mlflow`: bool *= False* -- If TRUE, then log to mlflow
        * `name`: Optional[str] *= null* -- Name of mlflow run
        * `project`: str *= hf_ehr* -- Name of mlflow experiment
    * `is_log_grad_norm`: bool *= False* -- If TRUE, then calculate + log grad norm over all params (!! slows down training a lot)
    * `log_every_n_steps`: int *= 1* -- Log every **N** steps

## `architecture`

Model architecture definitions

**Example:** `configs/architecture/bert.yaml` contains the Hydra config for a generic BERT model.

**Arguments:**
* `model`
    * `name`: str -- Short, unique identifier for model. Used in code to determine what type of model we're using.
    * `hf_name`: str -- Name/path to pass to `AutoModel.from_pretrained()`
    * `is_keep_pretrained_weights`: bool *= False* -- If TRUE and not resuming from ckpt, then keep the HF model's pretrained weights when starting training.
* `trainer`
    * `mlm_mask_pct`: Optional[float] -- For models that use MLM training, this is the percent of tokens to randomly mask
    * `optimizer`
        * `lr`: float -- Peak learning rate
    * `scheduler`
        * `num_warmup_steps`: int -- Number of warmup steps
        * `num_decay_steps`: int -- Number of decay steps
        * `initial_lr`: float -- Initial learning rate
        * `final_lr`: float -- Final learning rate
    
    * `data`
        * `mlm_prob`: Optional[float] -- Probability of masking tokens for MLM training

## `data`

Dataset configurations

**Example:** `configs/data/v8.yaml` contains the Hydra config for the v8 `FEMRDataset`.

**Arguments:**
* `data`
    * `dataset`
        * `name`: str -- Name of class of dataset from [hf_ehr/data/datasets.py](../data/datasets.py) that this dataset is initialized from
        * `path_to_femr_extract`: str -- Path to FEMR extract
        * `is_debug`: bool *= False*-- If True, use a small subset of the data for debugging

## `model`

Model hyperparameters for different sizes / instantiations

**Example:** `configs/model/bert-base.yaml` contains the Hydra config for the `bert-base` version of BERT.

**Arguments:**

* `data`
    * `dataloader`: Dict[str, Any]
        * `max_length`: int -- Maximum sequence length that a patient's timeline will get truncated to. !! Make sure to override this if you set the context length of the model to be larger, otherwise the model will only see datapoints with `length <= data.dataloader.max_length` !!
* `model`
    * `config_kwargs`: Dict[str, Any] -- Optional settings for model; varies by model
        => BERT
        * `max_position_embeddings`: int -- Maximum sequence length that the model can handle
        * `num_hidden_layers`: int -- Number of hidden layers in the model
        * `num_attention_heads`: int -- Number of attention heads in the model
        * `hidden_size`: int -- Size of the hidden layers in the model

## `tokenizer`

Tokenizer hyperparameters

**Example:** `configs/tokenizer/clmbr.yaml` contains the Hydra config for the `CLMBRTokenizer` tokenizer.

**Arguments:**
* `data`
    * `tokenizer`
        * `name`: str -- Name of class in `data/tokenization.py` that defines the tokenizer
        * `path_to_config`: str -- Path to the tokenizer's primary `tokenizer_config.json` file
    * `metadata`: Optional[Dict[str, Any]] -- Optional settings for tokenizer; varies by tokenizer
        => CLMBRTokenizer
        * `N`/A
        => DescTokenizer
        * `excluded_vocabs`: Optional[List[str]] -- List of vocabularies from which tokens are excluded. e.g. ["STANFORD_OBS", "LOINC"]
        * `desc_emb_tokenizer`: str -- Name/path of tokenizer passed to `AutoTokenizer.from_pretrained()` to use as the base textual tokenizer
        => CookbookTokenizer
        * `excluded_vocabs`: Optional[List[str]] -- List of vocabularies from which tokens are excluded. e.g. ["STANFORD_OBS", "LOINC"]
        * `is_remap_numerical_codes_to_quantiles`: bool -- If True, remap numerical codes to a bucketed range
        * `min_code_occurrence_count`: Optional[int] # Any code that occurs < `min_code_occurrence_count` times in the train dataset will be excluded
        * `keep_n_max_occurrence_codes`: Optional[int] # Keep only the top `keep_n_max_occurrence_codes` codes, sorted by occurrence count in train dataset

## `trainer`

Settings for single v. multiple GPU training

**Example:** `configs/trainer/single_gpu.yaml` contains the Hydra config for a single GPU training run.

* `trainer`
    * `devices`: List[int] -- List of GPU IDs to use for training
    * `distributed_backend`: str -- Distributed backend to use for training. Options: `ddp`, `ddp_cpu`, `ddp2`, `ddp_spawn`
* `data`
    * `dataloader`
        * `n_workers`: int -- Number of workers to use for data loading; **0** means data loading will be done in the main process