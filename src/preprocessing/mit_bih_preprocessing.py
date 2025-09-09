import logging
from pathlib import Path
import yaml
import pandas as pd
import wfdb
from record import Record
from preprocessing_base import STEP_REGISTRY, PreprocessingStep
import preproccesing_steps
import pyarrow as pa
import pyarrow.parquet as pq
import time

"""
Summary of the Runtime Flow
Python starts your script.
Imports happen:
preproccesing_steps.py is imported, registering all steps.
STEP_REGISTRY is now populated.
Your main function loads the pipeline config.
For each record, run_pipeline looks up and applies each step in order.
No need to manually instantiate or call each step—just ensure the module is imported.
"""

def run_pipeline(record: Record, pipeline: list[dict], logger) -> Record:
    """pipeline defiend in preprocessing config"""
    out = record
    for spec in pipeline:
        op = spec["op"]
        params = spec.get("params", {})
        step_cls = STEP_REGISTRY[op]
        out = step_cls(params)(out)
    return out

def mit_bih_preprocessing_main(config_path: str):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    raw_dir = Path(cfg["raw_dir"])
    index_parquet = Path(cfg["index_parquet"])
    pipeline = cfg["pipeline"]
    annotation_extension = cfg["annotation_extension"]
    index_row_map = cfg["index_row_map"]
    out_dir = Path(cfg["out_dir"])
    for split_type in index_row_map.values():
        preprocessed_dir = out_dir / split_type
        preprocessed_dir.mkdir(parents=True, exist_ok=True)   

    record_df = pd.read_parquet(index_parquet) # row 0 is train, row 1 is val, row 2 is test

    # TODO: parallelize because processing records is an independent operation
    for index, row in record_df.iterrows():
        split_type = index_row_map[index]
        write_dir = out_dir / split_type
        for record_id in row['split_column']:
            recording = wfdb.rdrecord(f'{raw_dir}/{record_id}')
            annotation = wfdb.rdann(f'{raw_dir}/{record_id}', annotation_extension)
            record = Record.from_wfdb(recording, annotation, record_id)
            record = run_pipeline(record, pipeline, logger)

            # write the preprocessed intermediate data to parquet
            record_windows = pa.Table.from_pandas(record.window_table)
            pq.write_table(record_windows, f'{write_dir}/{record_id}.parquet')

            logger.info(f"Wrote preprocessed parquet for record {record_id}")

        
if __name__ == "__main__":
    config_path = "../../configs/preprocessing/mit_bih_preprocessing.yaml"
    start_time = time.perf_counter()
    mit_bih_preprocessing_main(config_path)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Function executed in {elapsed_time:.6f} seconds")



           