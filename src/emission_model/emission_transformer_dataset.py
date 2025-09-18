import torch
from torch.utils.data import Dataset
from pathlib import Path
import yaml
import pandas as pd
import numpy as np

"""
Training dataset class that loads preprocessed parquet files from data/ directory.
Extract labeled windows: each sample should be a tuple of (snippet_array, label) where label indicates presence/absence of target event
Handle class imbalance (likely many negative examples)
Split into train/validation sets
"""

class EmissionTransformerDataset(Dataset):
    def __init__(self, data_dir: str):
        """
        data_dir (string): directory with preprocessed data
        """
        self.data_dir = data_dir
    
    def __len__(self):
        """
        Number of samples in the dataset. This should be the number of candidate points.
        """
        pass

    def __getitem__(self, idx):
        """
        Retrieve a single sample and its label given an index.
        It typically involves loading the data (e.g., an image, text), 
        applying any specified transformations, and returning them as PyTorch tensors.
        """
    

    @staticmethod
    def prepare_parquet(config_path: str):
        """
        Given a directory containing parquet files for preprocessed individual records, 
        generate a torch file containing training dataset samples that will be used by the dataset class.
        Save as a .pt file with a dictionary like {'snippets': torch.tensor(snippets), 'labels': torch.tensor(labels)}.
        Collect `snippets` as a list of numpy arrays (each snippet is a (C, T) array)
        Collect `labels` as a list of integers (0 or 1)
        """
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        data_dir = Path(cfg["data_dir"])

        snippets, labels = np.array(), np.array()

        for intermediate_file in data_dir.path:
            intermediate_data = pd.read_parquet(intermediate_file)
            
            for index, row in intermediate_data.iterrows():
                snippets, labels = process_intermediate_row(row, snippets, labels)
        

    @staticmethod 
    def _process_intermediate_row(row, snippets, labels):
        #


        #`torch.stack(snippets)` and `torch.tensor(labels)`
        #Save the dict: `torch.save({'snippets': snippets_tensor, 'labels': labels_tensor}, 'training_data.pt')`






