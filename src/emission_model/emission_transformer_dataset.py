import torch
from torch.utils.data import Dataset
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from dataclasses import dataclass
import time
import logging

"""
Training dataset class that loads preprocessed parquet files from data/directory.
Extract labeled windows: each sample should be a tuple of (snippet_array, label) where label indicates heartbeat superclass
Split into train/validation/test sets
"""
@dataclass
class DatasetConfig:
    data_dir: Path
    fs: int
    window_before: int 
    window_after: int
    superclass2label: dict[str, int]
    non_channel_columns: set
    window_len: int
    out_dir: Path

class EmissionTransformerDataset(Dataset):
    dataset_config: DatasetConfig = None
    missing_annotation_row_count = 0

    def __init__(self, data_path: str):
        """
        data_dir (string): file with preprocessed data
        """
        self.data_path = data_path
        self.loaded_tensors = torch.load(self.data_path)
        self.snippets = self.loaded_tensors["snippets"]
        self.labels = self.loaded_tensors["labels"]
        
    
    def __len__(self):
        """
        Number of samples in the dataset. This should be the number of candidate points.
        """
        return len(self.loaded_tensors["snippets"])

    def __getitem__(self, idx):
        """
        Retrieve a single sample and its label given an index.
        It typically involves loading the data (e.g., an image, text), 
        applying any specified transformations, and returning them as PyTorch tensors.
        """
        return self.snippets[idx].float(), self.labels[idx]


    @staticmethod
    def prepare_parquet(config_path: str):
        """
        Given a directory containing parquet files for preprocessed individual records, 
        generate a torch file containing training dataset samples that will be used by the dataset class.
        Save as a .pt file with a dictionary like {'snippets': torch.tensor(snippets), 'labels': torch.tensor(labels)}.
        Collect `snippets` as a list of numpy arrays (each snippet is a (C, T) array)
        Collect `labels` as a list of integers (0,1,2,3,4)
        """
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        dataset_config = DatasetConfig(
            data_dir = Path(cfg["data_dir"]),
            fs = int(cfg["fs"]),
            window_before = int(cfg["window_before"]), 
            window_after = int(cfg["window_after"]),
            superclass2label = cfg["superclass2label"],
            non_channel_columns= set(cfg["non_channel_columns"]),
            window_len = cfg["window_len"],
            out_dir = Path(cfg["out_dir"])
        )
        EmissionTransformerDataset.dataset_config = dataset_config

        for data_split in dataset_config.data_dir.iterdir():
            snippets, labels = [], []
            if data_split.is_dir():
                split_name = data_split.name
                logger.info(f"Starting to prepare emission data for {split_name}")
                emission_pt_path = dataset_config.out_dir / split_name
                emission_pt_path.mkdir(parents=True, exist_ok=True)
                pt_file_path = emission_pt_path / "data.pt"

                for intermediate_file in data_split.iterdir():
                    if intermediate_file.suffix == '.parquet':
                        intermediate_data = pd.read_parquet(intermediate_file)
            
                        for index, row in intermediate_data.iterrows():
                            snippets, labels = EmissionTransformerDataset._process_intermediate_row(row, snippets, labels, logger)
                
                # Save after processing all files in the split
                logger.info(f"Finished preparing emission data for {split_name}")
                snippets_tensor = torch.stack([torch.from_numpy(s) for s in snippets])
                labels_tensor = torch.tensor(labels)
                torch.save({'snippets': snippets_tensor, 'labels': labels_tensor}, pt_file_path)
        
        logger.info(f"missing annotation row count: {EmissionTransformerDataset.missing_annotation_row_count}")
        

    @staticmethod 
    def _process_intermediate_row(row, all_snippets, all_labels, logger):
        """
        Return current snippets and current labels with added ecg snippets and 
        heartbeat labels (one-hot) for the input row
        """
        row_snippets, row_labels = [], []
        candidate_indices = row["candidates"]
        annotation_ind = row["annotation_ind"]
        if len(annotation_ind) <= 0:
            # annotation_ind being less than 0 we will assume is an error in the training data (only see this in training data)
            EmissionTransformerDataset.missing_annotation_row_count += 1
            return all_snippets, all_labels

        annotation_symbols = row["annotation_symbols"]  # Assuming corrected column name
        superclass2label = EmissionTransformerDataset.dataset_config.superclass2label
        tolerance = 27  # Samples = 75 ms, adjust as needed
        ms_per_sample = 1000 / EmissionTransformerDataset.dataset_config.fs
        samples_before = int(EmissionTransformerDataset.dataset_config.window_before / ms_per_sample)
        samples_after = int(EmissionTransformerDataset.dataset_config.window_after / ms_per_sample)

        column_names = set(row.index)
        ecg_columns = column_names.difference(EmissionTransformerDataset.dataset_config.non_channel_columns)

        for candidate_i in candidate_indices:
            # for all annotation ind, calculate abs value between candidate_i and annotation_ind
            # min gives the closest annotation -> closest_idx
            closest_idx = min(range(len(annotation_ind)), key=lambda j: abs(annotation_ind[j] - candidate_i))
            dist = abs(annotation_ind[closest_idx] - candidate_i)
            if dist <= tolerance: # candidate is within 75 ms of a true beat
                symbol = annotation_symbols[closest_idx]
                label = superclass2label[symbol]
            else:
                label = 5  # No beat class
            row_labels.append(label)  # Integer label for multi-class

            # Extract snippet
            rel_candidate_i = candidate_i - row["start"]
            snippet_vec = []
            for lead in ecg_columns:
                signal = row[lead]
                if len(signal) != 3600:
                    logger.info(f"{lead} is not 3600 length, actual length is {len(signal)} rel_cand_i is {rel_candidate_i}")
                snippet = signal[rel_candidate_i - samples_before:rel_candidate_i + samples_after]
                if len(snippet) != 288:
                    logger.info(f"Row start: {row["start"]}, Row end: {row["end"]}, candidate_i: {candidate_i}")
                snippet_vec.append(snippet)
            row_snippets.append(np.stack(snippet_vec))
        
        all_snippets += row_snippets
        all_labels += row_labels
        return all_snippets, all_labels


if __name__ == "__main__":
    config = "../../configs/emission/mit_bih_emission_transformer_dataset.yaml"
    start_time = time.perf_counter()
    EmissionTransformerDataset.prepare_parquet(config)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Function executed in {elapsed_time:.6f} seconds")


"""
For some reason len(annotation_ind) == 0 for quite a few windows in the train set.
grok-code says that my code is right and this is just a byproduct of the raw data missing annotations.
This is a bad issue with the raw data.
For the sake of training the transformer it shouldn't be huge deal - we will just skip training on those candidates 
as opposed to assigning them no beat. We know that XQRS is pretty good at beat detection so the odds that XQRS detects a window with beats
but there are actually no beats at all in the window are really low.
"""










        






