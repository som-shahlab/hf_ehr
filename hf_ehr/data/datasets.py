import os
import numpy as np
import femr.datasets
from typing import Dict, List, Tuple
from torch.utils.data import Dataset
from hf_ehr.config import Event, SPLIT_TRAIN_CUTOFF, SPLIT_VAL_CUTOFF, SPLIT_SEED
from hf_ehr.data.tokenization import DescTokenizer


class BaseDataset(Dataset):
    pass


class MEDSDataset(BaseDataset):
    # TODO

    def __init__(
        self,
        path_to_meds_extract: str,
        split: str = "train",
        is_debug: bool = False,
        seed: int = 1,
    ):
        assert split in [
            "train",
            "val",
            "test",
        ], f"{split} not in ['train', 'val', 'test']"
        self.path_to_meds_extract: str = path_to_meds_extract
        self.split: str = split
        self.is_debug: bool = is_debug
        self.seed: int = seed

        # Set metadata -- used for tokenizer versioning later
        # ! CAUTION: Essential that this contains all args/kwargs; otherwise get_seq_length_per_patient() in tokenizer breaks!
        self.metadata = {
            "cls": "SparkDataset",
            "split": split,
            "is_debug": is_debug,
            "seed": seed,
        }

        # Pre-calculate canonical splits based on patient ids
        # TODO -- rewrite
        self.train_pids: np.ndarray = []
        self.val_pids: np.ndarray = []
        self.test_pids: np.ndarray = []

        # Confirm disjoint train/val/test
        assert np.intersect1d(self.train_pids, self.val_pids).shape[0] == 0
        assert np.intersect1d(self.train_pids, self.test_pids).shape[0] == 0
        assert np.intersect1d(self.val_pids, self.test_pids).shape[0] == 0

        # If debug, then shrink to 1k patients
        if is_debug:
            self.train_pids = self.train_pids[:1000]
            self.val_pids = self.val_pids[:1000]
            self.test_pids = self.test_pids[:1000]

    def get_n_patients(self) -> int:
        return len(self.get_pids())

    def get_pids(self) -> np.ndarray:
        """Return patient ids for this split"""
        if self.split == "train":
            pids = self.train_pids
        elif self.split == "val":
            pids = self.val_pids
        elif self.split == "test":
            pids = self.test_pids
        else:
            raise ValueError(f"Invalid split: {self.split}")
        return pids

    def __len__(self) -> int:
        return len(self.get_pids())

    def __getitem__(self, idx: int) -> Tuple[int, List[Event]]:
        """Return all event codes for this patient at `idx` in `self.split`."""
        # with meds_reader.PatientDatabase(self.path_to_meds_extract, num_threads=32) as database:
        # TODO
        pass


class SparkDataset(BaseDataset):
    """
    This is an implementation of the FEMRDataset class that reads data from a sql table using a spark context
    """

    from pyspark.context import SparkContext

    def __init__(
        self,
        spark: SparkContext,
        data_table_name: str,  # table in current spark context with columns [pat_id (int), timestamp, code (string)]
        split_table_name: str,  # table in current spark context with columns [pat_id (int), split (string in {train, test, val})]
        split: str = "train",
        is_debug: bool = False,
        seed: int = 1,
    ):
        assert split in [
            "train",
            "val",
            "test",
        ], f"{split} not in ['train', 'val', 'test']"
        self.split: str = split
        self.is_debug: bool = is_debug
        self.seed: int = seed

        # Set metadata -- used for tokenizer versioning later
        # ! CAUTION: Essential that this contains all args/kwargs; otherwise get_seq_length_per_patient() in tokenizer breaks!
        self.metadata = {
            "cls": "SparkDataset",
            "split": split,
            "is_debug": is_debug,
            "seed": seed,
        }

        self.code_df = spark.sql(f"SELECT * FROM {data_table_name}")

        split_df = spark.sql(f"SELECT * FROM {split_table_name}").toPandas()

        train_pids = split_df.loc[split_df['split'] == 'train', 'pat_id'].values
        test_pids = split_df.loc[split_df['split'] == 'test', 'pat_id'].values
        val_pids = split_df.loc[split_df['split'] == 'val', 'pat_id'].values

        # create patient id lists for each split
        self.train_pids: np.ndarray = np.array(train_pids)
        self.val_pids: np.ndarray = np.array(val_pids)
        self.test_pids: np.ndarray = np.array(test_pids)


        # Confirm disjoint train/val/test
        assert np.intersect1d(self.train_pids, self.val_pids).shape[0] == 0
        assert np.intersect1d(self.train_pids, self.test_pids).shape[0] == 0
        assert np.intersect1d(self.val_pids, self.test_pids).shape[0] == 0

        # If debug, then shrink to 1k patients
        if is_debug:
            self.train_pids = self.train_pids[:1000]
            self.val_pids = self.val_pids[:1000]
            self.test_pids = self.test_pids[:1000]

    def get_n_patients(self) -> int:
        return len(self.get_pids())

    def get_pids(self) -> np.ndarray:
        """Return patient ids for this split"""
        if self.split == "train":
            pids = self.train_pids
        elif self.split == "val":
            pids = self.val_pids
        elif self.split == "test":
            pids = self.test_pids
        else:
            raise ValueError(f"Invalid split: {self.split}")
        return pids

    def __len__(self) -> int:
        return len(self.get_pids())

    def __getitem__(self, idx: int) -> Tuple[int, List[Event]]:
        """
        Return all event codes for this patient at `idx` in `self.split`.
        """

        from pyspark.sql.functions import col

        # get the patient id
        pat_id = self.get_pids()[idx]
        print(f"getting events for patient with id={pat_id}")

        df = (
            self.code_df.select("code")
            .filter(col("pat_id") == pat_id)
            .orderBy(col("timestamp"))
        )

        code_list = df.toPandas()["code"].to_list()

        return (pat_id, code_list)


class FEMRDataset(BaseDataset):
    """Dataset that returns patients in a FEMR extract.
    dataset[idx] = a specific patient, so you can only retrieve ONE sample per patient.
    Note: Takes 1.5 hrs to loop through all event.code of all 3769353 patients in STARR-OMOP-deid-lite.
    """

    def __init__(
        self,
        path_to_femr_extract: str,
        split: str = "train",
        is_debug: bool = False,
        seed: int = 1,
    ):
        assert os.path.exists(
            path_to_femr_extract
        ), f"{path_to_femr_extract} is not a valid path"
        assert split in [
            "train",
            "val",
            "test",
        ], f"{split} not in ['train', 'val', 'test']"
        self.path_to_femr_extract: str = path_to_femr_extract
        self.femr_db = femr.datasets.PatientDatabase(path_to_femr_extract)
        self.split: str = split
        self.is_debug: bool = is_debug
        self.seed: int = seed

        # Set metadata -- used for tokenizer versioning later
        # ! CAUTION: Essential that this contains all args/kwargs; otherwise get_seq_length_per_patient() in tokenizer breaks!
        self.metadata = {
            "cls": "FEMRDataset",
            "path_to_femr_extract": path_to_femr_extract,
            "split": split,
            "is_debug": is_debug,
            "seed": seed,
        }

        # Pre-calculate canonical splits based on patient ids
        all_pids: np.ndarray = np.array([pid for pid in self.femr_db])
        hashed_pids: np.ndarray = np.array(
            [self.femr_db.compute_split(SPLIT_SEED, pid) for pid in all_pids]
        )
        self.train_pids: np.ndarray = all_pids[
            np.where(hashed_pids < SPLIT_TRAIN_CUTOFF)[0]
        ]
        self.val_pids: np.ndarray = all_pids[
            np.where(
                (SPLIT_TRAIN_CUTOFF <= hashed_pids) & (hashed_pids < SPLIT_VAL_CUTOFF)
            )[0]
        ]
        self.test_pids: np.ndarray = all_pids[
            np.where(hashed_pids >= SPLIT_VAL_CUTOFF)[0]
        ]

        # Confirm disjoint train/val/test
        assert np.intersect1d(self.train_pids, self.val_pids).shape[0] == 0
        assert np.intersect1d(self.train_pids, self.test_pids).shape[0] == 0
        assert np.intersect1d(self.val_pids, self.test_pids).shape[0] == 0

        # If debug, then shrink to 1k patients
        if is_debug:
            self.train_pids = self.train_pids[:1000]
            self.val_pids = self.val_pids[:1000]
            self.test_pids = self.test_pids[:1000]

    def get_n_patients(self) -> int:
        return len(self.get_pids())

    def get_pids(self) -> np.ndarray:
        """Return patient ids for this split"""
        if self.split == "train":
            pids = self.train_pids
        elif self.split == "val":
            pids = self.val_pids
        elif self.split == "test":
            pids = self.test_pids
        else:
            raise ValueError(f"Invalid split: {self.split}")
        return pids

    def __len__(self) -> int:
        return len(self.get_pids())

    def __getitem__(self, idx: int) -> Tuple[int, List[Event]]:
        """Return all event codes for this patient at `idx` in `self.split`."""
        pids: np.ndarray = self.get_pids()
        pid: int = pids[idx]

        # For negative `idx`, we need to unwrap `pid`
        if len(pid.shape) > 0:
            pid = pid[0]

        # Get data for each clinical event in patient timeline
        events: List[Event] = [
            Event(
                code=e.code,
                value=e.value,
                unit=e.unit,
                start=e.start,
                end=e.end,
                omop_table=e.omop_table,
            )
            for e in self.femr_db[pid].events
        ]
        # print("Time to fetch events: ", time.time() - start)
        return (pid, events)


class AllTokensFEMRDataset(FEMRDataset):
    """
    Wrapper around FEMRDataset that returns all tokens in the dataset.
    dataset[idx] = a specific sequence of tokens, so you can retrieve MULTIPLE samples per patient.
    Requires you to know (a) the tokenizer up front and (b) max_length per example, so that it knows how much to chunk each patient by.
    """

    def __init__(self, tokenizer, max_length: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(tokenizer, DescTokenizer):
            raise ValueError("DescTokenizer not supported for AllTokensFEMRDataset")
        self.tokenizer = tokenizer
        self.metadata = {
            **self.metadata,
            "tokenizer_metadata": tokenizer.metadata,
            "cls": "AllTokensFEMRDataset",
        }

        # ! Keep out of `metadata` b/c we want to treat all versions of the AllTokensDataset the same regardless of their `max_length`
        self.max_length: int = (
            max_length  # data.dataloader.max_length -- max length of a single sequence
        )

        # Number of tokens per patient timeline
        self.seq_length_per_patient: List[int] = tokenizer.get_seq_length_per_patient(
            self, n_procs=5
        )
        # Number of unique examples that will be extracted per patient by truncating their timeline to `max_length` tokens
        self.n_examples_per_patient: List[int] = [
            int(np.ceil(seq_length / max_length))
            for seq_length in self.seq_length_per_patient
        ]

        # Map [idx] => (p_idx, start idx in p_idx's timeline, end idx in p_idx's timeline)
        self.idx_to_pidx_start_end: List[Tuple[int, int, int]] = [
            (
                p_idx,
                i * max_length,
                min((i + 1) * max_length, self.seq_length_per_patient[p_idx]),
            )
            for p_idx, n_examples in enumerate(self.n_examples_per_patient)
            for i in range(n_examples)
        ]
        assert len(self.idx_to_pidx_start_end) == sum(
            self.n_examples_per_patient
        ), f"{len(self.idx_to_pidx_start_end)} != {sum(self.n_examples_per_patient)}"
        # Number of tokens per example in this dataset
        self.idx_to_seq_length: List[int] = [
            end - start for (p_idx, start, end) in self.idx_to_pidx_start_end
        ]

        self.cache: Dict[int, Tuple[int, int, List[Event]]] = (
            {}
        )  # Cache for last 1000 patients' timelines; [key] = p_idx, [value] = Tuple[idx, pid, events]

    def __len__(self) -> int:
        return len(self.idx_to_pidx_start_end)

    def __getitem__(self, idx: int) -> Tuple[int, List[Event]]:
        """
        Return all event codes for this example at `idx` in `self.split`.
        Maps this `idx` to the proper subsequence of events in this patient's timeline.
            Returns `start_token_idx` and `end_token_idx` to indicate the start and end of
            the subsequence within this patient's timeline that corresponds to `idx` in our dataset

        ! NOTE: This relies on the assumption that there is a one-to-one map between Event => Token;
            otherwise the indexing will be incorrect
        """
        (p_idx, start_token_idx, end_token_idx) = self.idx_to_pidx_start_end[idx]

        # Cache hit
        if p_idx in self.cache:
            pid: int = self.cache[p_idx][1]
            tokenizable_events: List[Event] = self.cache[p_idx][2][
                start_token_idx:end_token_idx
            ]
            return (pid, tokenizable_events)

        # Cache miss
        (pid, events) = super().__getitem__(p_idx)  # Fetch all events for this patient
        # Filter out events that don't have a corresponding token, then return the subsequence
        tokenizable_events: List[Event] = []
        idx: int = 0
        while idx < len(events):
            if self.tokenizer.convert_event_to_token(events[idx]) is not None:
                tokenizable_events.append(events[idx])
            idx += 1

        # Update cache
        self.cache[p_idx] = (idx, pid, tokenizable_events)
        if len(self.cache) > 1_000:
            # Find minimal `idx` in cache, then delete it
            min_p_idx: int = min(self.cache, key=lambda x: self.cache[x][0])
            del self.cache[min_p_idx]

        return (pid, tokenizable_events[start_token_idx:end_token_idx])


if __name__ == "__main__":
    from hf_ehr.data.tokenization import CLMBRTokenizer, DescTokenizer
    from hf_ehr.config import (
        PATH_TO_FEMR_EXTRACT_v8,
        PATH_TO_FEMR_EXTRACT_MIMIC4,
        PATH_TO_TOKENIZER_CLMBR_v8_CONFIG,
        PATH_TO_TOKENIZER_DESC_v8_CONFIG,
        PATH_TO_TOKENIZER_COOKBOOK_v8_CONFIG,
    )
    import time

    # Tokenizer
    # tokenizer = CLMBRTokenizer(PATH_TO_TOKENIZER_CLMBR_v8_CONFIG)

    # # AllTokensFEMRDataset
    # train_dataset = AllTokensFEMRDataset(tokenizer, max_length=1024, path_to_femr_extract=PATH_TO_FEMR_EXTRACT_v8, split='train', is_debug=True)
    # val_dataset = AllTokensFEMRDataset(tokenizer, max_length=1024, path_to_femr_extract=PATH_TO_FEMR_EXTRACT_v8, split='val', is_debug=True)
    # test_dataset = AllTokensFEMRDataset(tokenizer, max_length=1024, path_to_femr_extract=PATH_TO_FEMR_EXTRACT_v8, split='test', is_debug=True)
    # assert len(train_dataset) == 1081
    # assert len(train_dataset.get_pids()) == 1000
    # assert len(train_dataset[-3][1]) == 2326 # raw FEMR events
    # assert train_dataset[-3][2] == 1024 # start token idx
    # assert train_dataset[-3][3] == 1858 # end token idx
    # assert train_dataset.idx_to_pidx_start_end[-3] == (997, 1024, 1858)
    # assert train_dataset.idx_to_seq_length[-3] == 1858 - 1024
    # assert train_dataset.seq_length_per_patient[997] == 1858
    # print('train', len(train_dataset))
    # print('val', len(val_dataset))
    # print('test', len(test_dataset))
    # print(train_dataset[0])
    # print(train_dataset[-1])

    # FEMRDataset
    # TODO - sanity check that `PATH_TO_FEMR_EXTRACT_MIMIC4` works
    train_dataset = FEMRDataset(PATH_TO_FEMR_EXTRACT_MIMIC4, split="train")
    val_dataset = FEMRDataset(PATH_TO_FEMR_EXTRACT_MIMIC4, split="val")
    test_dataset = FEMRDataset(PATH_TO_FEMR_EXTRACT_MIMIC4, split="test")
    breakpoint()
    t1 = time.time()
    event_count = 0
    for pid in tqdm(train_dataset.get_pids()[:100000]):
        for e in train_dataset.femr_db[pid].events:
            event_count += 1
            train_dataset.femr_db.get_ontology().get_text_description(e.code)
    t2 = time.time()
    print("Time to loop through all events in train_dataset: ", t2 - t1)
    # Print average time per event
    print("Average time per patient: ", (t2 - t1) / 100000)
    print("Average time per event: ", (t2 - t1) / event_count)
