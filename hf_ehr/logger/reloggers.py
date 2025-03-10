import wandb
import torch
import pandas as pd
import os
import shutil
from loguru import logger

class WandbRelogger:
    """
    Handles the relogging of metrics from a previous wandb run into a new run.
    """
    def __init__(self, project: str, entity: str):
        self.project = project
        self.entity = entity
        self.api = wandb.Api()

    def get_run_id(self, run_log_dir: str) -> str:
        """Go through all the prev_wandb_run_id_*.txt files (including wandb_run_id.txt) in the run_log_dir and return the one that
        has the highest step."""
        wandb_run_ids: List[str] = []
        
        # Get the previous checkpoint with the highest step
        highest_step_ckpt_path: str | None = None
        highest_step_ckpt: str | None = None
        for file in os.listdir(os.path.join(run_log_dir, '../ckpts')):
            if file.endswith('.ckpt'):
                ckpt = torch.load(os.path.join(run_log_dir, '../ckpts', file), map_location='cpu', weights_only=False)
                if (
                    highest_step_ckpt is None or 
                    ckpt['global_step'] > highest_step_ckpt['global_step']
                ):
                    highest_step_ckpt = ckpt
                    highest_step_ckpt_path = os.path.abspath(os.path.join(run_log_dir, '../ckpts', file))

        # Make sure that a checkpoint was found
        if highest_step_ckpt is None:
            raise ValueError(f'No checkpoint found in the run_log_dir @ `{run_log_dir}`')

        # Make sure that 'last.ckpt' is the highest step checkpoint
        path_to_last_ckpt = os.path.abspath(os.path.join(run_log_dir, '../ckpts', 'last.ckpt'))
        if highest_step_ckpt['global_step'] != self.get_last_step(path_to_last_ckpt):
            logger.critical(f'Current `last.ckpt` @ `{path_to_last_ckpt}` is not the highest step checkpoint. It has steps={self.get_last_step(path_to_last_ckpt)}, but the highest step checkpoint has `steps={highest_step_ckpt["global_step"]}`. Overwriting `last.ckpt` with the highest step checkpoint found at `{highest_step_ckpt_path}` (wandb_run_id={highest_step_ckpt["wandb_run_id"]}).')
            shutil.copy(highest_step_ckpt_path, path_to_last_ckpt)
        else:
            logger.critical(f'Current `last.ckpt` @ `{path_to_last_ckpt}` is the highest step checkpoint. No need to overwrite. Resuming run from checkpoint @ `{path_to_last_ckpt}`.')
        return highest_step_ckpt['wandb_run_id']

    def update_run_id(self, run_log_dir: str, run_id: str, prev_run_id: str):
        """Updates the wandb run ID in the run_log_dir."""
        def get_unique_filepath(base_dir, base_name):
            counter = 1
            new_name = base_name
            while os.path.exists(os.path.join(base_dir, new_name)):
                new_name = f'{base_name.split(".")[0]}_{counter}.txt'
                counter += 1
            return os.path.join(base_dir, new_name)

        run_id_filepath = os.path.join(run_log_dir, 'wandb_run_id.txt')
        with open(run_id_filepath, 'w') as f:
            logger.info(f'Updating wandb run ID to {run_id} in {run_id_filepath}')
            f.write(run_id)

        prev_run_id_filepath = get_unique_filepath(run_log_dir, 'prev_wandb_run_id.txt')
        with open(prev_run_id_filepath, 'w') as f:
            logger.info(f'Storing previous wandb run ID {prev_run_id} in {prev_run_id_filepath}')
            f.write(prev_run_id)
    
    def get_last_step(self, ckpt_path: str) -> int:
        """Gets the last step from the model checkpoint."""
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        return checkpoint["global_step"]
    
    def relog_metrics(self, ckpt_path: str, run_log_dir: str):
        """Relog all metrics history into a new run up to the last step 
        logged by the ckpt and returns the new run."""
        run_id = self.get_run_id(run_log_dir)
        # Last step from the model checkpoint.
        # It may be before the last logged step in wandb.
        last_step = self.get_last_step(ckpt_path)
        logger.info(f"Last step found in checkpoint: {last_step}")
        old_run = self.api.run(f'{self.entity}/{self.project}/runs/{run_id}')
        new_run = wandb.init(entity='ehr-fm',
                             project=self.project,
                             name=old_run.name, 
                             dir=run_log_dir,
                             resume='never',
                             config=old_run.config)
        new_run.define_metric('train/loss', summary='min')
        new_run.define_metric('val/loss', summary='min')
        history = pd.DataFrame([row for row in old_run.scan_history()])
        filtered_history = history[history['_step'] <= last_step]
        
        for _, row in filtered_history.iterrows():
            metrics = row.dropna().to_dict()
            if 'val/loss' in metrics:
                print(metrics)
            step = int(metrics.pop('_step'))
            new_run.log(metrics, step=step)
        self.update_run_id(run_log_dir, new_run.id, run_id)
        return new_run


if __name__ == '__main__':
    # run_log_path = '/share/pi/nigam/migufuen/hf_ehr/cache/runs/hyena-medium-log-test/logs/'
    run_log_path = '/share/pi/nigam/migufuen/hf_ehr/cache/runs/hyena-medium-log-test2/logs'
    # run_log_path = '/share/pi/nigam/migufuen/hf_ehr/cache/runs/gpt2-base-10-epochs/logs/'
    # run_log_path = '/share/pi/nigam/migufuen/hf_ehr/cache/runs/gpt2-base-lr-1e-4/logs/'
    ckpt_path = '/share/pi/nigam/migufuen/hf_ehr/cache/runs/hyena-medium-log-test2/ckpts/last.ckpt'
    reloader = WandbRelogger('hf_ehr', 'ehr-fm')
    new_run = reloader.relog_metrics(ckpt_path, run_log_path)
    print(new_run.id)
    new_run.finish()
    
    # This code loads the model checkpoint and "global_step" contains the int for the current step
    # checkpoint = torch.load(PATH_TO_MODEL, map_location='cpu', weights_only=False)
    # print(checkpoint.keys())
    # print(checkpoint["global_step"])
    # exit()
