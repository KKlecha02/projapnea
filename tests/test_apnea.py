import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.apnea import analyze


def flat_signal(value, n_seconds):
    return [{"timestamp": float(i), "value": value} for i in range(n_seconds)]


def signal_with_drops(n_seconds, drop_centers, drop_value, drop_duration):
    values = [97.0] * n_seconds
    for center in drop_centers:
        half = drop_duration // 2
        for j in range(max(0, center - half), min(n_seconds, center + half + 1)):
            values[j] = drop_value
    return [{"timestamp": float(i), "value": values[i]} for i in range(n_seconds)]


def test_no_events():
    data = flat_signal(97.0, 3600)
    result = analyze(data, 1.0)
    assert result["apnea_events"] == 0
    assert result["ahi"] == 0.0
    assert result["severity"] == "brak"


def test_three_events():
    data = signal_with_drops(3600, [600, 1800, 3000], 91.0, 15)
    result = analyze(data, 1.0)
    assert result["apnea_events"] == 3


def test_median_removes_noise():
    values = [97.0] * 3600
    values[900] = 80.0
    data = [{"timestamp": float(i), "value": values[i]} for i in range(3600)]
    result = analyze(data, 1.0)
    assert result["apnea_events"] == 0


def test_empty_raises():
    with pytest.raises(ValueError):
        analyze([], 1.0)


def test_severity_lagodna():
    data = signal_with_drops(7200, [i * 450 + 300 for i in range(15)], 91.0, 15)
    result = analyze(data, 2.0)
    assert result["severity"] == "lagodna"


def test_severity_umiarkowana():
    data = signal_with_drops(7200, [i * 175 + 150 for i in range(40)], 91.0, 15)
    result = analyze(data, 2.0)
    assert result["severity"] == "umiarkowana"
