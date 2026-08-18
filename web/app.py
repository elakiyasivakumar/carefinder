"""Localhost web UI for the triage pipeline.

Deliberately thin: it validates input, calls run_triage, and hands the dict to
the page. All clinical logic lives in the graph — this layer must never make a
care decision of its own.
"""
import logging
import os
import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from triage_cli import ZIP_CODE_RE, build_deps, run_triage  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", static_url_path="/static")

# Built once: MedGemma and Flash clients are lazy, so this costs nothing until
# a request actually arrives.
_deps = None


def deps():
    global _deps
    if _deps is None:
        _deps = build_deps()
    return _deps


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/triage", methods=["POST"])
def triage():
    data = request.get_json(silent=True) or {}

    age = data.get("age")
    gender = data.get("gender", "Other")
    symptoms = (data.get("symptoms") or "").strip()
    zip_code = (data.get("zip_code") or "").strip()

    try:
        age = int(age)
    except (TypeError, ValueError):
        return jsonify({"error": "Please enter a valid age."}), 400
    if not 0 <= age <= 120:
        return jsonify({"error": "Please enter an age between 0 and 120."}), 400
    if not symptoms:
        return jsonify({"error": "Please describe your symptoms."}), 400
    if not ZIP_CODE_RE.match(zip_code):
        return jsonify({"error": "Please enter a valid 5-digit US zip code."}), 400

    try:
        result = run_triage(age, gender, symptoms, zip_code, deps=deps())
    except Exception as e:
        logger.exception("triage failed")
        return jsonify({"error": "Assessment unavailable. Please try again."}), 500

    return jsonify(result)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    print(f"\n  med-triage running at http://localhost:{port}\n")
    app.run(host="127.0.0.1", port=port, debug=False)
