import json
from pathlib import Path
from scipy.signal import medfilt


def load_config():
    p = Path(__file__).parent.parent / "config.json"
    with open(p) as f:
        return json.load(f)


def apply_median_filter(values, points):
    if points % 2 == 0:
        points += 1
    return list(medfilt(values, kernel_size=points))


def replace_zeros(values):
    arr = list(values)
    non_zero = [v for v in arr if v != 0]
    if not non_zero:
        return arr
    mean_val = sum(non_zero) / len(non_zero)
    return [mean_val if v == 0 else v for v in arr]


def compute_baseline(values, timestamps, idx, window_secs):
    t = timestamps[idx]
    window = [values[j] for j in range(idx) if t - timestamps[j] <= window_secs]
    if len(window) < 5:
        fallback = values[:min(60, len(values))]
        return sum(fallback) / len(fallback)
    return sum(window) / len(window)


def detect_episodes(spo2_data, cfg):
    timestamps = [d["timestamp"] for d in spo2_data]
    values = replace_zeros([d["value"] for d in spo2_data])
    filtered = apply_median_filter(values, cfg["median_filter_points"])

    drop_thr = cfg["min_spo2_drop_percent"]
    abs_thr = cfg["min_spo2_absolute"]
    min_dur = cfg["min_duration_seconds"]
    baseline_win = cfg["baseline_window_seconds"]

    episodes = []
    in_episode = False
    ep_start_idx = None
    ep_min = None
    ep_baseline = None
    ep_trigger = None

    for i, val in enumerate(filtered):
        baseline = compute_baseline(filtered, timestamps, i, baseline_win)
        drop = baseline - val
        is_apneic = drop >= drop_thr or val < abs_thr

        if not in_episode and is_apneic:
            in_episode = True
            ep_start_idx = i
            ep_min = val
            ep_baseline = baseline
            ep_trigger = "drop" if drop >= drop_thr else "absolute"
        elif in_episode and is_apneic:
            ep_min = min(ep_min, val)
        elif in_episode and not is_apneic:
            duration = timestamps[i] - timestamps[ep_start_idx]
            if duration >= min_dur:
                episodes.append({
                    "start_time": timestamps[ep_start_idx],
                    "end_time": timestamps[i],
                    "duration": round(duration, 1),
                    "min_spo2": round(ep_min, 1),
                    "baseline_spo2": round(ep_baseline, 1),
                    "drop_pct": round(ep_baseline - ep_min, 1),
                    "trigger": ep_trigger,
                })
            in_episode = False

    if in_episode:
        i = len(filtered) - 1
        duration = timestamps[i] - timestamps[ep_start_idx]
        if duration >= min_dur:
            episodes.append({
                "start_time": timestamps[ep_start_idx],
                "end_time": timestamps[i],
                "duration": round(duration, 1),
                "min_spo2": round(ep_min, 1),
                "baseline_spo2": round(ep_baseline, 1),
                "drop_pct": round(ep_baseline - ep_min, 1),
                "trigger": ep_trigger,
            })

    return merge_episodes(episodes, cfg["merge_gap_seconds"])


def merge_episodes(episodes, gap_secs):
    if len(episodes) <= 1:
        return episodes
    merged = [episodes[0].copy()]
    for ep in episodes[1:]:
        prev = merged[-1]
        if ep["start_time"] - prev["end_time"] <= gap_secs:
            prev["end_time"] = ep["end_time"]
            prev["duration"] = round(prev["end_time"] - prev["start_time"], 1)
            prev["min_spo2"] = min(prev["min_spo2"], ep["min_spo2"])
        else:
            merged.append(ep.copy())
    return merged


def compute_ahi(n_events, hours):
    return float(round(n_events / hours, 1))


def classify_severity(ahi, cfg):
    t = cfg["ahi_thresholds"]
    if ahi < t["mild"]:
        return "brak"
    if ahi < t["moderate"]:
        return "lagodna"
    if ahi < t["severe"]:
        return "umiarkowana"
    return "ciezka"


def compute_score(ahi):
    return max(0, round(100 - (ahi / 30) * 100))


def analyze(spo2_data, record_duration_hours):
    if len(spo2_data) == 0:
        raise ValueError("spo2_data must not be empty")
    if len(spo2_data) < 2:
        raise ValueError("spo2_data must contain at least 2 samples")
    if record_duration_hours <= 0:
        raise ValueError("record_duration_hours must be positive")

    cfg = load_config()
    episodes = detect_episodes(spo2_data, cfg)
    n = len(episodes)
    ahi = compute_ahi(n, record_duration_hours)
    return {
        "apnea_events": n,
        "ahi": ahi,
        "severity": classify_severity(ahi, cfg),
        "score": compute_score(ahi),
        "_episodes": episodes,
    }
