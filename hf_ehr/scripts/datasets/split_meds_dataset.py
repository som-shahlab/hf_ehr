"""
Purpose:
    Split the MEDS dataset into train, tuning, and held-out sets.

Usage:
    python3 split_meds_dataset.py --path_to_meds_reader $PATH_TO_MEDS_READER --train_split_size 0.8 --val_split_size 0.1
"""
import meds_reader
import polars as pl
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path_to_meds_reader", type=str, required=True)
parser.add_argument("--train_split_size", type=float, default=0.8)
parser.add_argument("--val_split_size", type=float, default=0.1)
args = parser.parse_args()

assert args.train_split_size + args.val_split_size <= 1.0, "Train and val split sizes must sum to less than 1.0"
assert args.train_split_size > 0.0 and args.val_split_size > 0.0, "Train and val split sizes must be greater than 0.0"

# Load the MEDS dataset
database = meds_reader.SubjectDatabase(args.path_to_meds_reader)
subject_ids = list(database)
n_patients = len(subject_ids)

# Split the dataset into train, val ("tuning"), and test ("held_out") sets
splits = [
    ('train' if idx < args.train_split_size * n_patients else 'tuning' if idx < (args.train_split_size + args.val_split_size) * n_patients else 'held_out', subject_ids[idx])
    for idx in range(len(subject_ids))
]
df = pl.DataFrame(splits, schema=["split", "subject_id"])
print("# of TOTAL patients:", n_patients)
print("# of train patients:", df.filter(pl.col("split") == "train").shape[0])
print("# of val patients:", df.filter(pl.col("split") == "tuning").shape[0])
print("# of test patients:", df.filter(pl.col("split") == "held_out").shape[0])

# Write the splits to a parquet file
df.write_parquet(os.path.join(args.path_to_meds_reader, 'metadata', 'subject_splits.parquet'))