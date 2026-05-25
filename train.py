import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from load_dataset import OSAData
from src.apnea_engine import ApneaCNNLSTM, extract_windows

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

with open("config.json") as f:
    cfg = json.load(f)


def main():
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Install torch to train the model.")
        sys.exit(1)

    dataset = OSAData(cfg["data_path_train"])
    patients = dataset.load_data(cfg["train_patients"])

    if not patients:
        print("No training patients found. Check data_path_train and train_patients in config.json.")
        sys.exit(1)

    X_all, y_all = [], []
    for p in patients:
        ts = np.array(p["timestamps"])
        vals = np.array(p["values"])
        X, y = extract_windows(ts.tolist(), vals.tolist(), p["events"], cfg)
        if len(X) > 0:
            X_all.append(X)
            y_all.append(y)
        print(f"Patient {p['patient_id']}: {len(X)} windows")

    if not X_all:
        print("No windows extracted.")
        sys.exit(1)

    X = np.concatenate(X_all)
    y = np.concatenate(y_all)

    class_counts = np.bincount(y)
    weights = 1.0 / class_counts[y]
    sampler = WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.float32), len(weights)
    )

    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    loader = DataLoader(ds, batch_size=cfg["batch_size"], sampler=sampler)

    model = ApneaCNNLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{cfg['epochs']} loss={total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), cfg["model_path"])
    print(f"Model saved to {cfg['model_path']}")


if __name__ == "__main__":
    main()
