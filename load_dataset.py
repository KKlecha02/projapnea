import json
import os
from glob import glob

import numpy as np
import pandas as pd


class OSAData:
    def __init__(self, data_path):
        self.data_path = data_path
        self.path_list = sorted(glob(os.path.join(data_path, "*")))

    @staticmethod
    def str2seconds(time_str):
        parts = time_str.split(":")
        h, m = int(parts[0]), int(parts[1])
        s_part = parts[2]
        if "." in s_part:
            s, ms = s_part.split(".")
            s, ms = int(s), float("0." + ms)
        else:
            s, ms = int(s_part), 0.0
        if h < 12:
            h += 24
        return h * 3600 + m * 60 + s + ms

    @staticmethod
    def extract_event_annotations(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def extract_spo2(spo_path, record_start):
        ts_col = pd.read_csv(spo_path, usecols=[1]).iloc[:, 0]
        timestamps = ts_col.apply(OSAData.str2seconds) - record_start

        val_col = pd.read_csv(spo_path, usecols=[2]).iloc[:, 0]
        values = [0.0 if str(x) == "-" else float(x) for x in val_col]

        arr = np.array(values)
        non_zero = arr[arr != 0]
        if len(non_zero) > 0:
            arr[arr == 0] = non_zero.mean()

        return timestamps.tolist(), arr.tolist()

    def load_data(self, patient_ids=None):
        patients = []
        for path in self.path_list:
            if not os.path.isdir(path):
                continue
            pid = os.path.basename(path)
            if patient_ids is not None and pid not in patient_ids:
                continue

            files = glob(os.path.join(path, "*"))
            ann_file = next((f for f in files if "annotation" in f), None)
            spo2_file = next((f for f in files if "SpO2" in f), None)

            if not ann_file or not spo2_file:
                continue

            annotation = self.extract_event_annotations(ann_file)
            record_start = annotation["record_start"]
            timestamps, values = self.extract_spo2(spo2_file, record_start)

            events = []
            for ev in annotation.get("events", []):
                events.append({
                    "event_type": ev["event_type"],
                    "start": ev["evnet_start"] - record_start,
                    "duration": ev["event_duration"],
                })

            patients.append({
                "patient_id": pid,
                "timestamps": timestamps,
                "values": values,
                "events": events,
                "record_start": record_start,
            })

        return patients
