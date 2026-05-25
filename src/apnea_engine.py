import json
import numpy as np
from pathlib import Path

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


WINDOW = 60
STRIDE = 10
THRESHOLD = 0.25


def load_config():
    p = Path(__file__).parent.parent / "config.json"
    with open(p) as f:
        return json.load(f)


def _get_session():
    if not ONNX_AVAILABLE:
        raise RuntimeError("onnxruntime nie jest zainstalowane")
    model_path = Path(__file__).parent.parent / "model.onnx"
    if not model_path.exists():
        raise RuntimeError(f"Brak pliku modelu: {model_path}")
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
    session = _get_session()

    timestamps = np.array([d["timestamp"] for d in spo2_data])
    values = np.array([d["value"] for d in spo2_data], dtype=np.float32)

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
        raise RuntimeError("Za mało danych do analizy (min. 60 sekund)")

    probs = _predict_windows(session, windows)
    preds = (probs >= THRESHOLD).astype(int)

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
