"""Credit: https://github.com/microsoft/protein-sequence-models/blob/main/sequence_models/samplers.py"""
from typing import List, Tuple, Generator, Optional
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
                 secondary_sort_key: Optional[List[int]] = None,
                 n_replicas: int = 1):
        if secondary_sort_key is not None:
            self.data: np.ndarray = np.lexsort((np.array(secondary_sort_key), -1 * np.array(sequence_lengths))) # sort longest -> shortest; break ties by considering `secondary_sort_key`; useful if `secondary_sort_key` is patient_id for the AllTokensFEMRDataset so that we keep all subsequences from the same patient together
        else:
            self.data: np.ndarray = np.argsort(-1 * np.array(sequence_lengths)) # sort longest -> shortest; NOTE: keep so that if we blow out memory, we do so earlier rather than later
        self.n_replicas: int = n_replicas
        self.num_samples: int = int(math.ceil(len(self.data) * 1.0 / self.n_replicas))
        self.bucket_size: int = bucket_size
        n_buckets: int = int(np.ceil(len(self.data) / self.bucket_size))
        self.data = [self.data[i * bucket_size: i * bucket_size + bucket_size] for i in range(n_buckets)]
        self.epoch: int = 0
        self.total_size: int = self.num_samples * self.n_replicas
        self.is_random_shuffle_across_buckets: bool = is_random_shuffle_across_buckets
        self.is_random_shuffle_within_buckets: bool = is_random_shuffle_within_buckets

    def __iter__(self):
        """This gets called once at the start of every epoch."""
        np.random.seed(self.epoch)
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
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int):
        """Ensures different shuffling for each epoch"""
        # ! Be sure to add a call to this function to PyTorch Lightning hook on epoch_end()
        self.epoch: int = epoch


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
        self.start_batch_idx: int = 0 # batch idx to start yielding at;used for resuming samping from the last index saved in a checkpoint
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
        batch_counter: int = 0 # count number of batches we've yielded so far
        for sampler_idx in self.sampler:
            this_length: int = min(self.sample_lengths[sampler_idx], self.model_context_window) # min() b/c seq will get truncated to fit into context window anyway
            linear = (len(batch) + 1) * max(max_length, this_length)
            if linear <= self.max_tokens:
                batch.append(sampler_idx)
                max_length = max(max_length, this_length)
                if len(batch) == self.max_examples:
                    if batch_counter >= self.start_batch_idx:
                        # Only yield an actual batch if we've reached the `start_batch_idx`
                        yield batch
                    batch = []
                    max_length = 0
                    batch_counter += 1
            else:
                rounded_n = (len(batch) // self.batch_mult) * self.batch_mult
                rounded_n = max(1, rounded_n)
                if batch_counter >= self.start_batch_idx:
                    # Only yield an actual batch if we've reached the `start_batch_idx`
                    yield batch[:rounded_n]
                batch = batch[rounded_n:] + [sampler_idx]
                max_length = max([min(self.sample_lengths[i], self.model_context_window) for i in batch])
                batch_counter += 1
        if len(batch) > 0:
            yield batch

    def set_epoch(self, epoch: int):
        """Ensures different shuffling for each epoch"""
        # ! Be sure to add a call to this function to PyTorch Lightning hook on epoch_end()
        self.sampler.set_epoch(epoch)
        self.start_batch_idx = 0 # Reset starting batch idx b/c new epoch

if __name__ == '__main__':
    sequence_lengths = [ # NOTE: Sort descending, so we batch in reverse order (i.e. starting from bottom)
        4, 4,           # 0,1
        5, 5, 6, 7,     # 2,3,4,5
        10, 13,         # 6,7
        14, 14,         # 8,9
        14, 15,         # 10,11
        20,             # 12
        22,             # 13 
        30,             # 14
    ]
    sampler = SortishSampler(sequence_lengths,
                             10,
                            False,
                            False,
                            None,
                            1)
    # Confirm buckets of size 10 are sorted properly
    assert all(sampler.data[0] == [ 14, 13, 12, 11, 9, 8, 10, 7, 6, 5, ])
    assert all(sampler.data[1] == [ 4, 2, 3, 0, 1 ])
    print("Buckets:", sampler.data)
    
    approx = ApproxBatchSampler(sequence_lengths,
                                sampler, 
                                30, 
                                30, 
                                99999999, 
                                1, 
                                True)
    batches = [ x for x in approx ]
    # Confirm batching is correct
    assert batches == [
        [14],
        [13],
        [12],
        [11, 9],
        [8, 10],
        [7, 6],
        [5, 4, 2, 3],
        [0, 1],
    ]
    print("Batches:", batches)
    
    # Confirm can restart batching at specific point
    approx.start_batch_idx = 4 # start at 4th batch_idx
    batches_2 = [ x for x in approx ]
    assert batches_2 == [
        [8, 10],
        [7, 6],
        [5, 4, 2, 3],
        [0, 1],
    ]
    print("Batches starting with 4th idx:", batches_2)