import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from load_dataset import OSAData

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

with open("config.json") as f:
    cfg = json.load(f)

WINDOW_SIZE = cfg["window_size_seconds"]
MARGIN = cfg["event_margin_seconds"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_event(t_start, t_end, events, margin=30):
    for ev in events:
        ev_start = ev["start"]
        ev_end = ev["start"] + ev["duration"]
        if t_start < ev_end + margin and t_end > ev_start - margin:
            return True
    return False


def extract_windows(timestamps, values, events):
    timestamps = np.array(timestamps)
    values = np.array(values)

    signals, labels = [], []

    for ev in events:
        t_start = ev["start"]
        mask = (timestamps >= t_start) & (timestamps <= t_start + WINDOW_SIZE - 1)
        window = values[mask]
        if len(window) == WINDOW_SIZE:
            signals.append(window)
            labels.append(1)

    t = timestamps[0] + 30
    while t + WINDOW_SIZE < timestamps[-1]:
        if not find_event(t, t + WINDOW_SIZE, events):
            mask = (timestamps >= t) & (timestamps < t + WINDOW_SIZE)
            window = values[mask]
            if len(window) == WINDOW_SIZE:
                signals.append(window)
                labels.append(0)
        t += WINDOW_SIZE

    return np.array(signals, dtype=np.float32), np.array(labels, dtype=np.int64)


class CNNLSTMClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=11, padding=5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        self.lstm = nn.LSTM(
            input_size=128, hidden_size=64, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.4
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        x = out[:, -1, :]
        return self.fc(x)


def main():
    if not TORCH_AVAILABLE:
        print("PyTorch not available.")
        sys.exit(1)

    print(f"Device: {DEVICE}")

    dataset = OSAData("../Data")
    patients = dataset.load_data(cfg["train_patients"])

    if not patients:
        print("No training patients found.")
        sys.exit(1)

    X_all, y_all = [], []
    for p in patients:
        X, y = extract_windows(p["timestamps"], p["values"], p["events"])
        if len(X) > 0:
            X_all.append(X)
            y_all.append(y)
        print(f"Patient {p['patient_id']}: {len(X)} windows, apnea={y.sum()}, normal={(y==0).sum()}")

    X = np.concatenate(X_all)
    y = np.concatenate(y_all)

    # normalizacja per-okno (z-score) - lepsza generalizacja na nowych pacjentach
    mean = X.mean(axis=1, keepdims=True)
    std  = X.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    X = (X - mean) / std

    np.save("norm_params.npy", np.array([0.0, 1.0]))  # per-window, brak globalnych parametrow
    print(f"\nTotal windows: {len(X)}, apnea={y.sum()}, normal={(y==0).sum()}")

    X_tensor = torch.from_numpy(X).unsqueeze(-1)
    y_tensor = torch.from_numpy(y).long()

    counts = np.bincount(y)
    weights = 1.0 / counts[y]
    sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.float32), len(weights))

    ds = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(ds, batch_size=cfg["batch_size"], sampler=sampler)

    model = CNNLSTMClassifier().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=2)
    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")
    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.float().to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (out.argmax(1) == yb).sum().item()
            total += yb.size(0)
        scheduler.step()
        avg_loss = total_loss / len(loader)
        acc = correct / total
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), cfg["model_path"])
        print(f"Epoch {epoch+1}/{cfg['epochs']} loss={avg_loss:.4f} acc={acc:.4f} lr={scheduler.get_last_lr()[0]:.6f}")

    print(f"\nModel saved to {cfg['model_path']}")
    print(f"Norm params saved to norm_params.npy  (mean={X_mean:.4f}, std={X_std:.4f})")


if __name__ == "__main__":
    main()
