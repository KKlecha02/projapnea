import json
import numpy as np
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src import apnea


def load_config():
    p = Path(__file__).parent.parent / "config.json"
    with open(p) as f:
        return json.load(f)


if TORCH_AVAILABLE:
    class ApneaCNNLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 32, kernel_size=11, padding=5)
            self.bn1 = nn.BatchNorm1d(32)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=7, padding=3)
            self.bn2 = nn.BatchNorm1d(64)
            self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
            self.dropout = nn.Dropout(0.3)
            self.lstm = nn.LSTM(128, 64, num_layers=2, bidirectional=True, batch_first=True)
            self.fc1 = nn.Linear(128, 64)
            self.fc2 = nn.Linear(64, 2)

        def forward(self, x):
            x = x.permute(0, 2, 1)
            x = torch.max_pool1d(torch.relu(self.bn1(self.conv1(x))), 2)
            x = torch.max_pool1d(torch.relu(self.bn2(self.conv2(x))), 2)
            x = self.dropout(torch.relu(self.conv3(x)))
            x = x.permute(0, 2, 1)
            out, _ = self.lstm(x)
            x = out[:, -1, :]
            x = self.dropout(torch.relu(self.fc1(x)))
            return torch.softmax(self.fc2(x), dim=1)


def zscore_normalize(window):
    std = window.std()
    if std == 0:
        return np.zeros_like(window)
    return (window - window.mean()) / std


def extract_windows(timestamps, values, events, cfg):
    ws = cfg["window_size_seconds"]
    margin = cfg["event_margin_seconds"]
    values = np.array(values)
    timestamps = np.array(timestamps)

    event_times = [(e["start"], e["start"] + e["duration"]) for e in events]

    X, y = [], []

    for ev in events:
        center = ev["start"] + ev["duration"] / 2
        start_t = center - ws / 2
        end_t = center + ws / 2
        mask = (timestamps >= start_t) & (timestamps < end_t)
        window_vals = values[mask]
        if len(window_vals) == ws:
            w = zscore_normalize(window_vals.reshape(-1, 1).astype(np.float32))
            X.append(w)
            y.append(1)

    t0 = timestamps[0]
    t1 = timestamps[-1]
    t = t0
    while t + ws <= t1:
        in_margin = any(
            abs(t + ws / 2 - (es + ed / 2)) <= margin
            for es, ed in [(e["start"], e["duration"]) for e in events]
        )
        if not in_margin:
            mask = (timestamps >= t) & (timestamps < t + ws)
            window_vals = values[mask]
            if len(window_vals) == ws:
                w = zscore_normalize(window_vals.reshape(-1, 1).astype(np.float32))
                X.append(w)
                y.append(0)
        t += ws

    if not X:
        return np.empty((0, ws, 1), dtype=np.float32), np.empty((0,), dtype=np.int64)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def analyze(spo2_data, record_duration_hours):
    result = apnea.analyze(spo2_data, record_duration_hours)
    result["spo2_data"] = spo2_data
    return result
