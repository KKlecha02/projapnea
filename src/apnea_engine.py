import json
import numpy as np
from pathlib import Path

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


WINDOW = 60
STRIDE = 10
THRESHOLD = 0.25


def load_config():
    p = Path(__file__).parent.parent / "config.json"
    with open(p) as f:
        return json.load(f)


def _get_session():
    model_path = Path(__file__).parent.parent / "model.onnx"
    if not model_path.exists():
        return None
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def _predict_windows(session, windows):
    X = np.array(windows, dtype=np.float32)
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    X = (X - mean) / std
    X = X[:, :, np.newaxis]
    logits = session.run(["output"], {"input": X})[0]
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    return probs[:, 1]


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def _classify_severity(ahi, cfg):
    t = cfg["ahi_thresholds"]
    if ahi < t["mild"]:
        return "brak"
    if ahi < t["moderate"]:
        return "lagodna"
    if ahi < t["severe"]:
        return "umiarkowana"
    return "ciezka"


def _compute_score(ahi):
    return max(0, round(100 - (ahi / 30) * 100))


def analyze(spo2_data, record_duration_hours):
    cfg = load_config()

    timestamps = np.array([d["timestamp"] for d in spo2_data])
    values = np.array([d["value"] for d in spo2_data], dtype=np.float32)

    session = _get_session()

    if not ONNX_AVAILABLE or session is None:
        from src import apnea as rule_apnea
        result = rule_apnea.analyze(spo2_data, record_duration_hours)
        result["spo2_data"] = spo2_data
        return result

    windows, w_starts = [], []
    t = timestamps[0]
    while t + WINDOW <= timestamps[-1]:
        mask = (timestamps >= t) & (timestamps < t + WINDOW)
        w = values[mask]
        if len(w) == WINDOW:
            windows.append(w)
            w_starts.append(t)
        t += STRIDE

    if not windows:
        from src import apnea as rule_apnea
        result = rule_apnea.analyze(spo2_data, record_duration_hours)
        result["spo2_data"] = spo2_data
        return result

    probs = _predict_windows(session, windows)
    preds = (probs >= THRESHOLD).astype(int)

    # buduj epizody z bloków sasiadujacych okien
    episodes = []
    in_ep = False
    ep_start = None
    for wt, pred in zip(w_starts, preds):
        if pred == 1 and not in_ep:
            ep_start = wt
            in_ep = True
        elif pred == 0 and in_ep:
            episodes.append({"start_time": float(ep_start), "end_time": float(wt),
                             "duration": round(float(wt - ep_start), 1)})
            in_ep = False
    if in_ep:
        ep_end = w_starts[-1] + WINDOW
        episodes.append({"start_time": float(ep_start), "end_time": float(ep_end),
                        "duration": round(float(ep_end - ep_start), 1)})

    # AHI na podstawie przejsc 0->1
    transitions = sum(1 for i in range(1, len(preds)) if preds[i] == 1 and preds[i-1] == 0)
    if preds[0] == 1:
        transitions += 1

    ahi = round(transitions / record_duration_hours, 1)
    severity = _classify_severity(ahi, cfg)
    score = _compute_score(ahi)

    return {
        "apnea_events": transitions,
        "ahi": ahi,
        "severity": severity,
        "score": score,
        "_episodes": episodes,
        "spo2_data": spo2_data,
    }


if TORCH_AVAILABLE:
    class ApneaCNNLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 32, kernel_size=11, padding=5)
            self.bn1 = nn.BatchNorm1d(32)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=7, padding=3)
            self.bn2 = nn.BatchNorm1d(64)
            self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
            self.dropout = nn.Dropout(0.4)
            self.lstm = nn.LSTM(128, 64, num_layers=2, bidirectional=True, batch_first=True, dropout=0.4)
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


def extract_windows(timestamps, values, events, cfg):
    ws = cfg["window_size_seconds"]
    margin = cfg["event_margin_seconds"]
    values = np.array(values)
    timestamps = np.array(timestamps)

    X, y = [], []

    for ev in events:
        t_start = ev["start"]
        mask = (timestamps >= t_start) & (timestamps <= t_start + ws - 1)
        window_vals = values[mask]
        if len(window_vals) == ws:
            mean = window_vals.mean(); std = window_vals.std()
            if std == 0: std = 1.0
            w = ((window_vals - mean) / std).reshape(-1, 1).astype(np.float32)
            X.append(w); y.append(1)

    t0 = timestamps[0]; t1 = timestamps[-1]
    t = t0
    while t + ws <= t1:
        in_margin = any(
            abs(t + ws / 2 - (e["start"] + e["duration"] / 2)) <= margin
            for e in events
        )
        if not in_margin:
            mask = (timestamps >= t) & (timestamps < t + ws)
            window_vals = values[mask]
            if len(window_vals) == ws:
                mean = window_vals.mean(); std = window_vals.std()
                if std == 0: std = 1.0
                w = ((window_vals - mean) / std).reshape(-1, 1).astype(np.float32)
                X.append(w); y.append(0)
        t += ws

    if not X:
        return np.empty((0, ws, 1), dtype=np.float32), np.empty((0,), dtype=np.int64)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)
