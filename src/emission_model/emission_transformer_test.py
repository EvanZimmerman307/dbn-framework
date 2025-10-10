import logging
import yaml
from pathlib import Path
import torch
import torch.nn as nn
from emission_transformer_model import EmissionTransformer
from emission_transformer_dataset import EmissionTransformerDataset
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import time

def test_emission_transformer(config_path: str):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Testing!")
    

    with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
    
    model_path = Path(cfg["model_path"])
    test_data_path = Path(cfg["test_data_path"])
    batch_size = cfg["batch_size"]
    c_in = cfg["c_in"]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print("Using GPU!")

    model = EmissionTransformer(
        c_in = c_in
    ).to(device)

    state_dict = torch.load(model_path, map_location='cpu') 
    model.load_state_dict(state_dict)
    model.eval()

    emission_transformer_test_dataset = EmissionTransformerDataset(test_data_path)
    test_dataloader = DataLoader(emission_transformer_test_dataset, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()

    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for snippet, label in test_dataloader:
            snippet, label = snippet.to(device), label.to(device)
            logits, _ = model(snippet)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
    
    # Compute metrics:
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    conf_matrix = confusion_matrix(all_labels, all_preds)

    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Macro Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    logging.info("Confusion Matrix:")
    logging.info(conf_matrix)

if __name__ == "__main__":
    config_path = "../../configs/emission/mit_bih_emission_transformer_test.yaml"
    start_time = time.perf_counter()
    test_emission_transformer(config_path)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Testing executed in {elapsed_time:.6f} seconds")