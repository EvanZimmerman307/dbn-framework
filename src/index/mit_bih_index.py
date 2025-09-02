# src/index/wfdb_index.py

"""
Given a dataset in wfdb format. Split the dataset into train/validation/test sets.
This step varies a lot on the raw dataset iteself. For now, don't worry about generalizing.
"""

import pandas as pd
import yaml
from pathlib import Path
import os
from collections import defaultdict
import logging
import re
import numpy as np
import math
import pyarrow as pa
import pyarrow.parquet as pq


def mit_bih_index_main(config_path: str):
    # Configure basic logging (only runs once)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Get a logger instance
    logger = logging.getLogger(__name__)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    raw_dir = Path(cfg["raw_dir"])
    out_dir = Path(cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)   

    # Group files by common stem
    file_stem_map = defaultdict(list)
    record_id_pattern_str = cfg["record_id_pattern"]
    record_id_regex = re.compile(record_id_pattern_str)
    for filename in os.listdir(raw_dir):
        root, extension = os.path.splitext(filename)
        is_record = re.search(record_id_regex, filename)
        if is_record:
            file_stem_map[root].append(filename)
    logger.info(f"file_stem_map: {file_stem_map}")
    num_records = len(file_stem_map)
    logger.info(f"num_records: {num_records}")

    # 201 and 202 need to be combined becuase they are the same patient
    files_202 = file_stem_map.pop("202")
    file_stem_map["201"] += files_202

    # seeded shuffle for random, deterministic, train/val/test split
    record_ids = list(file_stem_map.keys())
    record_ids.sort()
    record_ids = np.array(record_ids)
    seed = int(cfg["seed"])
    # Initialize a PCG64 BitGenerator
    bitgen = np.random.PCG64(seed=seed)
    # Create a Generator
    rng = np.random.Generator(bitgen)
    # shuffle in-place
    rng.shuffle(record_ids)
    logger.info(f"shuffled records ids: {record_ids}")

    train_split, val_split, test_split = cfg["train_split"], cfg["val_split"], cfg["test_split"]

    train_cutoff = math.ceil(num_records * train_split)
    val_cutoff = train_cutoff + math.ceil(num_records * val_split)
    # test set is the remaining records
    logger.info(f"train_cutoff: {train_cutoff}, val_cutoff: {val_cutoff}")

    split_map = {}
    split_map["train"], split_map["val"], split_map["test"] = record_ids[:train_cutoff].astype(int), record_ids[train_cutoff:val_cutoff].astype(int), record_ids[val_cutoff:].astype(int)
    logger.info(f"train records: {split_map["train"]}, {len(split_map["train"])} records")
    logger.info(f"val records: {split_map["val"]}, {len(split_map["val"])} records")
    logger.info(f"test records: {split_map["test"]}, {len(split_map["test"])} records")
    train_checksum, val_checksum, test_checksum = np.sum(split_map["train"]), np.sum(split_map["val"]), np.sum(split_map["test"])
    logger.info(f"train_checksum: {train_checksum}, val_checksum: {val_checksum}, test_checksum: {test_checksum}")

    # Create split artifacts
    # Combine the lists into a dictionary where the values are lists of lists
    data = {
        'split_column': [split_map["train"], split_map["val"], split_map["test"]]
    }
    # Create a pandas DataFrame
    df = pd.DataFrame(data)
    # Write the DataFrame to a Parquet file using PyArrow
    table = pa.Table.from_pandas(df)
    pq.write_table(table, f'{out_dir}/mit_bih_index.parquet')
    # --- Verification ---
    # Read the Parquet file back to confirm the nested structure is preserved
    read_df = pd.read_parquet(f'{out_dir}/mit_bih_index.parquet')
    print(read_df)
    print(read_df.dtypes)

    # write out check sums for future validation
    checksum_dict = {
        "train_checksum": int(train_checksum),
        "val_checksum": int(val_checksum),
        "test_checksum": int(test_checksum)
    }
    with open(out_dir / "mit_bih_index_checksums.yaml", "w") as f:
        yaml.dump(checksum_dict, f)


if __name__ == "__main__":
    config_path = "../../configs/index/mit_bih_index.yaml"
    mit_bih_index_main(config_path)

    