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
                 n_samples_per_batch: Optional[List[int]] = None,
                 n_replicas: int = 1,
                 rank: Optional[int] = None):
        if secondary_sort_key is not None:
            self.data: np.ndarray = np.lexsort((np.array(secondary_sort_key), -1 * np.array(sequence_lengths))) # sort longest -> shortest; break ties by considering `secondary_sort_key`; useful if `secondary_sort_key` is patient_id for the AllTokensFEMRDataset so that we keep all subsequences from the same patient together
        else:
            self.data: np.ndarray = np.argsort(-1 * np.array(sequence_lengths)) # sort longest -> shortest; NOTE: keep so that if we blow out memory, we do so earlier rather than later
        self.n_replicas: int = n_replicas
        self.bucket_size: int = bucket_size
        n_buckets: int = int(np.ceil(len(sequence_lengths) / self.bucket_size))
        self.bucketed_data: List[np.ndarray] = [self.data[i * bucket_size: i * bucket_size + bucket_size] for i in range(n_buckets)]
        self.epoch: int = 0

        self.n_samples_per_batch: Optional[List[int]] = n_samples_per_batch # [idx] = batch idx, [value] = number of samples in that batch
        if self.n_samples_per_batch is not None:
            # We've been told explicitly the number of samples per batch. Use this to split batches across GPUs.
            # This allows for the possibility that there is a diff number of samples per batch 
            # e.g. if we're filling batches by token counts, and each sequence has a different length.
            # That means we need to track the number of samples we allocate to each GPU in order to get an 
            # even # of batches per GPU, b/c batches will be formed based on token counts rather than samples. 
            # If we don't evenly allocate the # of batches across GPUs, then torch trainer with DDP will stall
            self.n_batches_per_gpu: int = int(math.floor(len(self.n_samples_per_batch) * 1.0 / self.n_replicas))
            self.n_samples_per_gpu: List[int] = [ sum(self.n_samples_per_batch[x * self.n_batches_per_gpu: (x + 1) * self.n_batches_per_gpu]) for x in range(self.n_replicas) ] # [idx] = GPU idx, [value] = number of samples on that GPU
        else:
            # We don't know the number of samples per batch, so we'll just split samples evenly across GPUs
            # Assume even number of samples per batch, so can evenly split samples across GPUs to get even # of batches per GPU
            self.n_batches_per_gpu: int = int(math.floor(len(sequence_lengths) * 1.0 / self.n_replicas))
            self.n_samples_per_gpu: List[int] = [ self.n_batches_per_gpu ] * self.n_replicas # [idx] = GPU idx, [value] = number of samples on that GPU

        assert len(self.n_samples_per_gpu) == self.n_replicas, f"ERROR: len(self.n_samples_per_gpu) ({len(self.n_samples_per_gpu)}) != self.n_replicas ({self.n_replicas})"
        self.is_random_shuffle_across_buckets: bool = is_random_shuffle_across_buckets
        self.is_random_shuffle_within_buckets: bool = is_random_shuffle_within_buckets
        self.rank: int = rank if rank is not None else torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    def __iter__(self):
        """This gets called once at the start of every epoch."""
        np.random.seed(0)
        data: List[np.ndarray] = [ np.copy(x) for x in self.bucketed_data ] # copy to prevent modifying the original data; for deterministic shuffling between epochs
        if self.is_random_shuffle_within_buckets:
            for bucket in data:
                np.random.shuffle(bucket)
        if self.is_random_shuffle_across_buckets:
            np.random.shuffle(data)
        indices = [item for sublist in data for item in sublist]
        # subsample for this GPU. NOTE: the value of `start - end` might be different for each GPU 
        # b/c batches might be formed based on token counts rather than samples
        start: int = sum(self.n_samples_per_gpu[:self.rank])
        end: int = start + self.n_samples_per_gpu[self.rank]
        indices = indices[start:end]
        return iter(indices)

    def __len__(self) -> int:
        return self.n_samples_per_gpu[self.rank]

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
        self.length: int = None # result of len(self)
        self.n_samples_per_batch: List[int] = None # for tracking # of samples per batch -- used in Multi-GPU training to evenly distribute sample (b/c hard to know apriori how many samples are in a batch if splitting batches by max token limits)
        self.last_length_epoch_calc: int = None # for tracking self.length caching
        self.start_batch_idx: int = 0 # batch idx to start yielding at; used for resuming samping from the last index saved in a checkpoint
        assert self.max_tokens >= self.model_context_window, f"ERROR: max_tokens ({self.max_tokens}) must be >= model_context_window ({self.model_context_window}). Otherwise, you could get a sequence that is too long to be included in any batch, i.e. len(seq) == model_context_window > max_tokens, which means some batches will return empty which throws an error. It doesn't make sense to limit the batch size to be less than the model context window, b/c then you'll never fully fill the model's context window."

    def __len__(self):
        if not (self.length and self.last_length_epoch_calc == self.sampler.epoch): 
            # Calculate length of batch sampler
            self.last_length_epoch_calc = self.sampler.epoch
            length = 0
            n_samples_per_batch: List[int] = []
            for batch in iter(self):
                length += 1
                n_samples_per_batch.append(len(batch))
            self.length = length
            self.n_samples_per_batch = n_samples_per_batch
        rank: int = self.sampler.rank
        assert sum(self.n_samples_per_batch) == self.sampler.n_samples_per_gpu[rank], f"ERROR: sum(n_samples_per_batch) ({sum(self.n_samples_per_batch)}) != self.sampler.n_samples_per_gpu[rank] ({self.sampler.n_samples_per_gpu[rank]})"
        assert len(self.n_samples_per_batch) == self.length, f"ERROR: len(n_samples_per_batch) ({len(self.n_samples_per_batch)}) != self.length ({self.length})"
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
            # if batch_counter >= self.sampler.n_batches_per_gpu:
            #     # If we've yielded the max number of batches for this GPU, then stop
            #     # This is needed to keep the batch counts consistent across GPUs
            #     print(f"Breaking out of ranking {self.sampler.rank} at batch_counter={batch_counter} and n_batches_per_gpu={self.sampler.n_batches_per_gpu}")
            #     break
        if len(batch) > 0:
            yield batch
            batch_counter += 1
        if self.sampler.n_samples_per_batch is not None:
            # NOTE: Only do this check if we know aprior what the `self.sampler.n_samples_per_batch` will be (i.e. we've already run __len__ on this ApproxBatchSampler)
            # Otherwise, `self.sampler.n_batches_per_gpu` won't be accurate (will be the # of examples rather than the # of batches) b/c we haven't updated it 
            # with `self.sampler.n_samples_per_batch` in the __init__ of SortishSampler yet
            assert batch_counter == self.sampler.n_batches_per_gpu, f"ERROR: batch_counter ({batch_counter}) != self.sampler.n_batches_per_gpu ({self.sampler.n_batches_per_gpu})"

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
        29,             # 14
        30,             # 15
    ]

    # Multi-GPU distributed setup
    n_gpus: int = 2
    ## First, determine total # of batches in epoch
    sampler = SortishSampler(sequence_lengths, 10, False, False, None, n_replicas=1, rank=0)
    assert len(sampler) == 16
    assert len(sampler.bucketed_data) == 2
    assert sampler.n_samples_per_gpu == [16]
    assert [ x for x in sampler ] ==[15, 14, 13, 12, 11, 9, 10, 8, 7, 6, 5, 4, 3, 2, 1, 0], f"sampler={[ x for x in sampler ]}"
    approx = ApproxBatchSampler(sequence_lengths, sampler, 30, 30, 99999999, 1, True)
    assert len(approx) == 9
    assert approx.n_samples_per_batch == [1, 1, 1, 1, 2, 2, 2, 4, 2]
    assert [ x for x in approx ] == [[15], [14], [13], [12], [11, 9], [10, 8], [7, 6], [5, 4, 3, 2], [1, 0]]
    n_samples_per_batch = approx.n_samples_per_batch
    ## Second, split across GPUs
    sampler0 = SortishSampler(sequence_lengths, 10, False, False, None, n_samples_per_batch=n_samples_per_batch, n_replicas=n_gpus, rank=0)
    assert sampler0.n_samples_per_gpu == [4, 10], f"sampler0.n_samples_per_gpu={sampler0.n_samples_per_gpu}"
    sampler1 = SortishSampler(sequence_lengths, 10, False, False, None, n_samples_per_batch=n_samples_per_batch, n_replicas=n_gpus, rank=1)
    assert sampler1.n_samples_per_gpu == [4, 10], f"sampler1.n_samples_per_gpu={sampler1.n_samples_per_gpu}"
    assert all(np.array(sampler0.n_samples_per_batch) == np.array(sampler1.n_samples_per_batch)), f"ERROR: sampler0.n_samples_per_batch ({sampler0.n_samples_per_batch}) != sampler1.n_samples_per_batch ({sampler1.n_samples_per_batch})"
    approx0 = ApproxBatchSampler(sequence_lengths, sampler0, 30, 30, 99999999, 1, True)
    assert len(approx0) == 4
    assert approx0.sampler.n_batches_per_gpu == 4
    assert approx0.n_samples_per_batch == [1, 1, 1, 1]
    assert [ x for x in approx0 ] == [[15], [14], [13], [12]]
    approx1 = ApproxBatchSampler(sequence_lengths, sampler1, 30, 30, 99999999, 1, True)
    assert len(approx1) == 4
    assert approx1.sampler.n_batches_per_gpu == 4
    assert approx1.n_samples_per_batch == [2, 2, 2, 4]
    assert [ x for x in approx1 ] == [[11, 9], [10, 8], [7, 6], [5, 4, 3, 2]]
    
    # Confirm buckets of size 10 are sorted properly
    sequence_lengths = [ # NOTE: Sort descending, so we batch in reverse order (i.e. starting from bottom)
        4, 4,           # 0,1
        5, 5, 6, 7,     # 2,3,4,5
        10, 13,         # 6,7
        14, 14,         # 8,9
        14, 15,         # 10,11
        20,             # 12
        22,             # 13 
        29,             # 14
    ]
    sampler = SortishSampler(sequence_lengths,
                             10,
                            False,
                            False,
                            None,
                            n_replicas=1)
    assert all(sampler.bucketed_data[0] == [ 14, 13, 12, 11, 9, 8, 10, 7, 6, 5, ])
    assert all(sampler.bucketed_data[1] == [ 4, 2, 3, 0, 1 ])
    print("Buckets:", sampler.bucketed_data)
    
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
