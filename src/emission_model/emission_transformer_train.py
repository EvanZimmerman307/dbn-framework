import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from emission_transformer_model import EmissionTransformer
from emission_transformer_dataset import EmissionTransformerDataset
from dataclasses import dataclass
import yaml
from pathlib import Path
import time


def train_emission_transformer(config_path):
    print("Starting Training!")

    with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

    c_in = cfg["c_in"]
    train_data_path = Path(cfg["train_data_path"])
    val_data_path = Path(cfg["val_data_path"])
    batch_size = cfg["batch_size"]
    num_epochs = cfg["num_epochs"]
    lr = cfg["lr"]
    save_model_path = cfg["save_model_path"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print("Using GPU!")

    model = EmissionTransformer(
        c_in = c_in
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    emission_transformer_train_dataset = EmissionTransformerDataset(train_data_path)
    emission_transformer_val_dataset = EmissionTransformerDataset(val_data_path)
    train_dataloader = DataLoader(emission_transformer_train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(emission_transformer_val_dataset, batch_size=batch_size)
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for i, (snippet, label) in enumerate(train_dataloader):
            # inner loop (batches)
            snippet, label = snippet.to(device), label.to(device)
            optimizer.zero_grad()
            logits, _ = model(snippet)
            loss = criterion(logits, label)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss / len(train_dataloader):.4f}")

        #  save the model after eval if we get better results.
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (snippet, label) in enumerate(val_dataloader):
                # inner loop (batches)
                snippet, label = snippet.to(device), label.to(device)
                logits, _ = model(snippet)
                loss = criterion(logits, label)

                val_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss / len(val_dataloader):.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("NEW BEST VAL LOSS ACHIEVED! saving new best model")
            torch.save(model.state_dict(), save_model_path)

if __name__ == "__main__":
    config_path = "../../configs/emission/mit_bih_emission_transformer_train.yaml"
    start_time = time.perf_counter()
    train_emission_transformer(config_path)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Training executed in {elapsed_time:.6f} seconds")

             








