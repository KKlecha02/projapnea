import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).parent))

from load_dataset import OSAData
from src import apnea

with open("config.json") as f:
    _cfg = json.load(f)


def intervals_overlap(a_start, a_end, b_start, b_end, min_overlap=10):
    overlap = min(a_end, b_end) - max(a_start, b_start)
    return overlap >= min_overlap


def make_window_labels(timestamps, events, episodes, window_size=60):
    if not timestamps:
        return [], []
    t_end = timestamps[-1]
    y_true, y_pred = [], []
    t = 0.0
    while t + window_size <= t_end:
        wend = t + window_size
        true_hit = any(
            intervals_overlap(t, wend, ev["start"], ev["start"] + ev["duration"])
            for ev in events
        )
        pred_hit = any(
            intervals_overlap(t, wend, ep["start_time"], ep["end_time"])
            for ep in episodes
        )
        y_true.append(int(true_hit))
        y_pred.append(int(pred_hit))
        t += window_size
    return y_true, y_pred


def run_validation():
    dataset = OSAData(_cfg["data_path_val"])
    patients = dataset.load_data(_cfg["val_patients"])

    if not patients:
        print("No validation patients found. Check data_path_val and val_patients in config.json.")
        sys.exit(1)

    all_true, all_pred = [], []

    for p in patients:
        spo2_input = [
            {"timestamp": t, "value": v}
            for t, v in zip(p["timestamps"], p["values"])
        ]
        duration_hours = p["timestamps"][-1] / 3600 if p["timestamps"] else 1.0
        result = apnea.analyze(spo2_input, duration_hours)
        y_true, y_pred = make_window_labels(
            p["timestamps"], p["events"], result["_episodes"]
        )
        all_true.extend(y_true)
        all_pred.extend(y_pred)
        print(f"Patient {p['patient_id']}: events={result['apnea_events']} AHI={result['ahi']}")

    f1 = f1_score(all_true, all_pred, zero_division=0)
    cm = confusion_matrix(all_true, all_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0][0], 0, 0, 0)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    meets = bool(f1 >= 0.85)

    print(f"\nF1={f1:.3f}  sensitivity={sensitivity:.3f}  specificity={specificity:.3f}")

    if not meets:
        print("WARNING: F1 below 0.85 requirement.")
        print("Suggestions: lower min_spo2_drop_percent or min_duration_seconds in config.json")

    Path("docs").mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Normal", "Pred Apnea"])
    ax.set_yticklabels(["True Normal", "True Apnea"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("docs/confusion_matrix.png", dpi=150)
    plt.close()

    report = {
        "f1_score": round(f1, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "meets_requirement": meets,
    }
    with open("docs/validation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("Saved docs/confusion_matrix.png and docs/validation_report.json")
    return report


if __name__ == "__main__":
    run_validation()
