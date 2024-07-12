from functools import partial
from typing import List, Tuple, Any
import torch
import uuid
import hashlib

def convert_lab_value_to_token_from_ranges(code: str, unit: str, value: float, ranges: List[Tuple[float]]) -> str:
    # Given a list of ranges (i.e. tuples of [start, end] values), remaps the code to the index in the `ranges` array corresponds
    # to this code's value, i.e. "code" => "{code} || {idx}"
    # If the value doesn't fit in any of the ranges, returns the code itself, i.e. "{code}"
    for idx, (start_val, end_val) in enumerate(ranges):
        if start_val <= value <= end_val:
            return f"{code} || {unit} || R{idx + 1}" # "STANFORD_OBS/123 | mmol | R3"
    return f"{code} || {unit} || R0" # out of range, "STANFORD_OBS/123 | mmol | R0"

def convert_lab_value_to_token_from_quantiles(code: str, unit: str, value: float, quantiles: List[float]) -> str:
    # Note: If we have Q1, Q2, Q3, Q4, then `len(quantiles) == 3` b/c have [0.25, 0.5, 0.75]
    for q_idx, q in enumerate(quantiles):
        if value <= q: 
            return get_lab_value_token_name(code, unit, str(q_idx + 1))
    return get_lab_value_token_name(code, unit, "0") # out of range, "STANFORD_OBS/123 | mmol | Q0"

def get_lab_value_token_name(code: str, unit: str, quantile: str) -> str:
    return f"{code} || {unit} || Q{quantile}" # "STANFORD_OBS/123 | mmol | Q4"

def _get_linear_schedule_with_warmup_lr_lambda(current_step: int, 
                                               *, 
                                               num_warmup_steps: int, 
                                               num_decay_steps: int,
                                               initial_lr: float,
                                               peak_lr: float,
                                               final_lr: float) -> float:
    """Note that this needs to return a multiplier on `initial_lr` as set in the optimizer"""
    if current_step < num_warmup_steps:
        # Linear warmup from `initial_lr` to `peak_lr`
        new_lr: float = (peak_lr - initial_lr) / num_warmup_steps * current_step + initial_lr
    elif current_step < num_warmup_steps + num_decay_steps:
        # Linear decay from `peak_lr` to `final_lr`
        new_lr: float = (final_lr - peak_lr) / num_decay_steps * (current_step - num_warmup_steps) + peak_lr
    else:
        # Plateau at `final_lr`
        new_lr: float = final_lr
    multiplier: float = new_lr / peak_lr
    return multiplier

def lr_warmup_with_constant_plateau(optimizer, 
                                    num_warmup_steps: int, 
                                    num_decay_steps: int,
                                    initial_lr: float,
                                    final_lr: float, 
                                    last_epoch: int = -1):
    """
    Create a schedule with a learning rate that decreases linearly from the peak lr set in the optimizer to `final_lr`, after
    a warmup period during which it increases linearly from `initial_lr` to the peak lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_decay_steps (`int`):
            The total number of steps to decay the lr.
        initial_lr (`float`):
            The initial learning rate before the warmup phase
        final_lr (`float`):
            The final learning rate after the warmup and decay phases that we plateau at
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    assert num_warmup_steps > 0, f"num_warmup_steps must be > 0, got {num_warmup_steps}"
    assert num_decay_steps > 0, f"num_decay_steps must be > 0, got {num_decay_steps}"

    peak_lr: float = optimizer.param_groups[0]['lr']
    
    lr_lambda = partial(
        _get_linear_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_decay_steps=num_decay_steps,
        initial_lr=initial_lr,
        peak_lr=peak_lr,
        final_lr=final_lr,
    )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def hash_string_to_uuid(input: Any) -> str:
    """Create a MD5 hash of the stringified input"""
    input_string: str = str(input)
    md5_hash = hashlib.md5(input_string.encode()).hexdigest()
    
    # Generate a UUID from the MD5 hash
    generated_uuid = uuid.UUID(md5_hash)
    
    return str(generated_uuid)