import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template, request
import jsonschema

from src import apnea_engine

app = Flask(__name__, template_folder="../templates", static_folder="../static")

_cfg_path = Path(__file__).parent.parent / "config.json"
with open(_cfg_path) as _f:
    _cfg = json.load(_f)

INPUT_SCHEMA = {
    "type": "object",
    "required": ["record_duration_hours", "spo2"],
    "additionalProperties": True,
    "properties": {
        "record_duration_hours": {"type": "number", "exclusiveMinimum": 0},
        "spo2": {
            "type": "array",
            "minItems": 2,
            "items": {
                "type": "object",
                "required": ["timestamp", "value"],
                "properties": {
                    "timestamp": {"type": "number"},
                    "value": {"type": "number", "minimum": 0, "maximum": 100},
                },
            },
        },
    },
}


def load_history():
    results_dir = Path(_cfg["results_dir"])
    if not results_dir.exists():
        return []
    files = sorted(results_dir.glob("*.json"), reverse=True)
    entries = []
    for f in files[: _cfg["history_max_entries"]]:
        try:
            with open(f) as fh:
                entries.append(json.load(fh))
        except Exception:
            pass
    return entries


def save_result(result):
    results_dir = Path(_cfg["results_dir"])
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    saved = {
        "timestamp": ts,
        "apnea_events": result["apnea_events"],
        "ahi": result["ahi"],
        "severity": result["severity"],
        "score": result["score"],
    }
    with open(results_dir / f"{ts}.json", "w") as f:
        json.dump(saved, f, indent=2)
    return saved


def parse_payload(req):
    ct = req.content_type or ""
    if "multipart" in ct:
        file = req.files.get("file")
        if not file:
            keys = list(req.files.keys())
            return None, f"No file provided (received fields: {keys})"
        try:
            return json.load(file), None
        except json.JSONDecodeError as e:
            return None, f"Invalid JSON: {e}"
    if not ct:
        return None, f"Missing Content-Type header"
    data = req.get_json(force=True, silent=True)
    if data is None:
        return None, f"Invalid JSON or unsupported Content-Type: {ct}"
    return data, None


@app.route("/", methods=["GET"])
def index():
    history = load_history()
    return render_template("index.html", history=history)


@app.route("/analyze", methods=["POST"])
def analyze():
    payload, err = parse_payload(request)
    if err:
        if "application/json" in request.headers.get("Accept", ""):
            return jsonify({"error": err}), 400
        return render_template("index.html", history=load_history(), error=err), 400

    try:
        jsonschema.validate(payload, INPUT_SCHEMA)
    except jsonschema.ValidationError as e:
        msg = e.message
        if "application/json" in request.headers.get("Accept", ""):
            return jsonify({"error": msg}), 400
        return render_template("index.html", history=load_history(), error=msg), 400

    timestamps = [d["timestamp"] for d in payload["spo2"]]
    if timestamps != sorted(timestamps):
        msg = "spo2 timestamps must be monotonically increasing"
        if "application/json" in request.headers.get("Accept", ""):
            return jsonify({"error": msg}), 400
        return render_template("index.html", history=load_history(), error=msg), 400

    result_box = {}
    error_box = {}

    def run():
        try:
            result_box["data"] = apnea_engine.analyze(
                payload["spo2"], payload["record_duration_hours"]
            )
        except Exception as e:
            error_box["msg"] = str(e)

    t = threading.Thread(target=run)
    t.start()
    t.join(timeout=_cfg["api_timeout_seconds"])

    if t.is_alive():
        if "application/json" in request.headers.get("Accept", ""):
            return jsonify({"error": "Analysis timed out"}), 408
        return render_template("index.html", history=load_history(), error="Analysis timed out"), 408

    if "msg" in error_box:
        msg = error_box["msg"]
        if "application/json" in request.headers.get("Accept", ""):
            return jsonify({"error": msg}), 400
        return render_template("index.html", history=load_history(), error=msg), 400

    result = result_box["data"]
    save_result(result)

    if "application/json" in request.headers.get("Accept", ""):
        return jsonify({
            "apnea_events": result["apnea_events"],
            "ahi": result["ahi"],
            "severity": result["severity"],
            "score": result["score"],
        })

    spo2_data = result["spo2_data"]
    step = max(1, len(spo2_data) // 1800)
    display_spo2 = [
        {"x": d["timestamp"], "y": d["value"]}
        for d in spo2_data[::step]
    ]
    episodes = result["_episodes"]

    return render_template(
        "result.html",
        result=result,
        spo2_json=json.dumps(display_spo2),
        events_json=json.dumps(episodes),
    )


@app.route("/history", methods=["GET"])
def history():
    return jsonify(load_history())
