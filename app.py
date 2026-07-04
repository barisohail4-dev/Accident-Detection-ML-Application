"""Flask web application for uploading images and predicting accident risk."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

from src.config import OUTPUTS_DIR, UPLOADS_DIR, DATA_DIR
from src.predict import predict_image
from src.utils import load_image_for_model
import json
import threading
import subprocess
import sys

app = Flask(__name__, static_folder=str(OUTPUTS_DIR), static_url_path="/outputs")
app.config["UPLOAD_FOLDER"] = str(UPLOADS_DIR)
os.makedirs(UPLOADS_DIR, exist_ok=True)
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret")


@app.route("/", methods=["GET", "POST"])
def index() -> str:
    """Render the upload page and show predictions for uploaded images."""
    prediction = None
    confidence = None
    image_name = None

    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", prediction=prediction, confidence=confidence, image_name=image_name)

        uploaded_file = request.files["file"]
        if uploaded_file.filename == "":
            return render_template("index.html", prediction=prediction, confidence=confidence, image_name=image_name)

        filename = secure_filename(uploaded_file.filename)
        image_path = UPLOADS_DIR / filename
        uploaded_file.save(image_path)
        image_name = filename

        prediction, confidence = predict_image(image_path)

    # Load metrics and training plot if available
    metrics = None
    metrics_path = OUTPUTS_DIR / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            metrics = None

    # Count dataset instances per class
    counts = {}
    try:
        for split in ["train", "val", "test"]:
            split_dir = DATA_DIR / split
            if split_dir.exists():
                for class_dir in split_dir.iterdir():
                    if class_dir.is_dir():
                        counts.setdefault(split, {})[class_dir.name] = sum(1 for _ in class_dir.iterdir() if _.is_file())
    except Exception:
        counts = {}

    training_plot = None
    if (OUTPUTS_DIR / "training_history.png").exists():
        training_plot = "/outputs/training_history.png"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_name=image_name,
        metrics=metrics,
        counts=counts,
        training_plot=training_plot,
    )


def _run_retrain():
    """Run the training script in a background thread."""
    try:
        python = sys.executable
        subprocess.run([python, "src/train.py"], check=True)
    except Exception:
        pass


@app.route("/retrain", methods=["POST"])
def retrain():
    """Trigger retraining in background and redirect to index."""
    thread = threading.Thread(target=_run_retrain, daemon=True)
    thread.start()
    flash("Retraining started in background. Check logs in the server terminal.")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
