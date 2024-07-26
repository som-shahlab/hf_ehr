"""Credit: https://github.com/microsoft/protein-sequence-models/blob/main/sequence_models/samplers.py"""
from typing import List, Tuple, Generator
import math
import numpy as np
import torch
from torch.utils.data import Sampler, BatchSampler

class SortishSampler(Sampler):
    """Returns indices such that inputs with similar lengths are close together.
    Set bucket_size = 1 to do perfect sampling
    """

    def __init__(self, 
                 sequence_lengths: List[int], 
                 bucket_size: int, 
                 is_random_shuffle_across_buckets: bool = False, 
                 is_random_shuffle_within_buckets: bool = False,
                 num_replicas: int = 1):
        self.data: np.ndarray = np.argsort(-1 * np.array(sequence_lengths)) # sort longest -> shortest; NOTE: keep so that if we blow out memory, we do so earlier rather than later
        self.num_replicas: int = num_replicas
        self.num_samples: int = int(math.ceil(len(self.data) * 1.0 / self.num_replicas))
        self.bucket_size: int = bucket_size
        n_buckets: int = int(np.ceil(len(self.data) / self.bucket_size))
        self.data = [self.data[i * bucket_size: i * bucket_size + bucket_size] for i in range(n_buckets)]
        self.epoch: int = 0
        self.current_iter: int = 0
        self.total_size: int = self.num_samples * self.num_replicas
        self.is_random_shuffle_across_buckets: bool = is_random_shuffle_across_buckets
        self.is_random_shuffle_within_buckets: bool = is_random_shuffle_within_buckets

    def __iter__(self):
        np.random.seed(self.epoch * 10_000_000 + self.current_iter)
        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if self.is_random_shuffle_within_buckets:
            for bucket in self.data:
                np.random.shuffle(bucket)
        if self.is_random_shuffle_across_buckets:
            np.random.shuffle(self.data)
        indices = [item for sublist in self.data for item in sublist]
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size
        # subsample
        start = self.rank * self.num_samples
        end = start + self.num_samples
        indices = indices[start:end]
        self.current_iter += 1
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int):
        # Different shuffling for each epoch
        # ! Be sure to add a call to this function to PyTorch Lightning hook on epoch_end()
        self.epoch: int = epoch
        self.current_iter: int = 0


class ApproxBatchSampler(BatchSampler):
    """
	Parameters:
	-----------
	sampler : Pytorch Sampler
		Choose base sampler class to use for bucketing

	max_tokens : int
		Maximum number of tokens per batch

	max_examples: int
		Maximum examples in a single batch

	sample_lengths : array-like
		List of lengths of sequences in the order of the dataset
	"""

    def __init__(self, 
                sample_lengths: List[int], 
                sampler, 
                model_context_window: int, 
                max_tokens: int = 99999999, 
                max_examples: int = 99999999, 
                batch_mult: int = 1, 
                drop_last: bool = True):
        self.sample_lengths: List[int] = sample_lengths
        self.sampler = sampler
        self.model_context_window: int = model_context_window # max size of seq that model can handle, so any seq great than this will get truncated anyway
        self.max_tokens: int = max_tokens # max tokens per batch
        self.max_examples: int = max_examples # max examples per batch
        self.batch_mult: int = batch_mult # make batch sizes a multiple of this
        self.drop_last: bool = drop_last
        self.length = None # result of len(self)
        self.last_length_epoch_calc = None # for tracking self.length caching
        assert self.max_tokens >= self.model_context_window, f"ERROR: max_tokens ({self.max_tokens}) must be >= model_context_window ({self.model_context_window}). Otherwise, you could get a sequence that is too long to be included in any batch, i.e. len(seq) == model_context_window > max_tokens, which means some batches will return empty which throws an error. It doesn't make sense to limit the batch size to be less than the model context window, b/c then you'll never fully fill the model's context window."

    def __len__(self):
        if not (self.length and self.last_length_epoch_calc == self.sampler.epoch): 
            # Calculate length of batch sampler
            self.last_length_epoch_calc = self.sampler.epoch
            length = 0
            for __ in iter(self):
                length += 1
            self.length = length
        return self.length

    def __iter__(self) -> Generator[List[int], None, None]:
        batch: List[int] = [] # list of patient idx's in dataset included in this batch
        max_length: int = 0
        for idx in self.sampler:
            this_length: int = min(self.sample_lengths[idx], self.model_context_window) # min() b/c seq will get truncated to fit into context window anyway
            linear = (len(batch) + 1) * max(max_length, this_length)
            if linear <= self.max_tokens:
                batch.append(idx)
                max_length = max(max_length, this_length)
                if len(batch) == self.max_examples:
                    yield batch
                    batch = []
                    max_length = 0
            else:
                rounded_n = (len(batch) // self.batch_mult) * self.batch_mult
                rounded_n = max(1, rounded_n)
                yield batch[:rounded_n]
                batch = batch[rounded_n:] + [idx]
                max_length = max([self.sample_lengths[i] for i in batch])
        if len(batch) > 0:
            yield batch


class AllTokensApproxBatchSampler(ApproxBatchSampler):
    """
	Same as ApproxBatchSampler, but for AllTokensFEMRDataset, i.e. it returns all tokens 
    in each patient's timeline (i.e. generates multiple examples per patient) instead 
    of just one example per patient.
	-----------
	"""
    def __iter__(self) -> Generator[List[Tuple[int, int, int]], None, None]:
        batch: List[Tuple[int, int, int]] = [] # List of (patient idx, start idx in patient timeline, end idx in patient timeline) included in this batch
        max_length: int = 0
        for idx in self.sampler:
            # We want to save All Tokens, so split timeline into chunks of `model_context_window` tokens
            for chunk_idx, chunk_start in enumerate(range(0, self.sample_lengths[idx], self.model_context_window)):
                this_length: int = min(self.sample_lengths[idx] - chunk_start, self.model_context_window)
                linear = (len(batch) + 1) * max(max_length, this_length)
                if linear <= self.max_tokens:
                    batch.append(idx + chunk_idx)
                    max_length = max(max_length, this_length)
                    if len(batch) == self.max_examples:
                        yield batch
                        batch = []
                        max_length = 0
                else:
                    rounded_n = (len(batch) // self.batch_mult) * self.batch_mult
                    rounded_n = max(1, rounded_n)
                    yield batch[:rounded_n]
                    batch = batch[rounded_n:] + [ (idx, chunk_start, chunk_start + this_length) ]
                    max_length = max([chunk_end - chunk_start for (p_idx, chunk_start, chunk_end) in batch])
        if len(batch) > 0:
            yield batch
