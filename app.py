"""
ReadFit - Flask API. THE CONTRACT FILE.
OWNER: Moyinoluwa Ajibola (Track 1) / Adelabu Emmanuel (Track 6)

Every other track builds against the endpoints in this file. It is deliberately
written so it STARTS AND SERVES even when models/ is empty - that is what lets the
GCP track deploy at hour zero, before the model exists.

Endpoints:
  GET  /             HTML form
  POST /analyze      HTML result page
  POST /predict      JSON  <- outcome 6, the data service
  GET  /predict      JSON via query string (?text=...&lesson=34) - for curl demos
  GET  /metrics      JSON aggregate stats  <- dashboard reads this
  GET  /history      JSON recent predictions <- dashboard reads this
  GET  /benchmark    JSON model comparison  <- dashboard reads this
  GET  /lessons      JSON available phonics lessons
  GET  /health       JSON liveness         <- hour-zero deploy gate
"""
import json
import os
import sqlite3
import sys
import time

from flask import Flask, jsonify, request, render_template

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
import joblib
from features import linguistic_features, band_from_score, flesch_kincaid_grade
from verifier import check, available_lessons

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(ROOT, "models")
DB = os.path.join(ROOT, "data", "predictions.db")

app = Flask(__name__)

MODEL = None
VEC = None
METRICS = {}
MODEL_VERSION = "not-loaded"


def load_model():
    """Never raises. A missing model degrades the service, it does not kill it."""
    global MODEL, VEC, METRICS, MODEL_VERSION
    try:
        VEC = joblib.load(os.path.join(MODELS, "vectorizer.joblib"))
        MODEL = joblib.load(os.path.join(MODELS, "readability_model.joblib"))
        with open(os.path.join(MODELS, "metrics.json")) as f:
            METRICS = json.load(f)
        MODEL_VERSION = METRICS.get("trained_at", "unknown")
        print(f"[readfit] model loaded, version {MODEL_VERSION}")
    except Exception as e:
        print(f"[readfit] NO MODEL YET ({e}) - serving in degraded mode")
        MODEL, VEC, METRICS, MODEL_VERSION = None, None, {}, "not-loaded"


def init_db():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    c = sqlite3.connect(DB)
    c.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL, word_count INTEGER, lesson INTEGER,
        readability REAL, band TEXT, decodability REAL, fk_grade REAL,
        snippet TEXT)""")
    c.commit()
    c.close()


def log_prediction(row):
    try:
        c = sqlite3.connect(DB)
        c.execute("""INSERT INTO predictions
            (ts, word_count, lesson, readability, band, decodability, fk_grade, snippet)
            VALUES (?,?,?,?,?,?,?,?)""",
                  (time.time(), row["word_count"], row["lesson"], row["readability"],
                   row["band"], row["decodability"], row["fk_grade"], row["snippet"]))
        c.commit()
        c.close()
    except Exception as e:
        print(f"[readfit] log failed: {e}")


def analyse(text, lesson):
    """The one function that does the work. Both HTML and JSON paths call it."""
    from scipy.sparse import hstack, csr_matrix

    decod = check(text, lesson)
    fk = round(float(flesch_kincaid_grade(text)), 2)

    if MODEL is None or VEC is None:
        readability, band = None, "model not loaded"
    else:
        L = VEC["scaler"].transform(linguistic_features(text).reshape(1, -1))
        T = VEC["tfidf"].transform([text])
        readability = round(float(MODEL.predict(hstack([csr_matrix(L), T]).tocsr())[0]), 4)
        band = band_from_score(readability, METRICS.get("band_cuts", [-1, -0.5, 0, 0.5]))

    out = {
        "readability": readability,
        "band": band,
        "fk_grade": fk,
        "model_version": MODEL_VERSION,
        "word_count": decod["total_words"],
        "lesson": decod["lesson"],
        "lesson_label": decod["lesson_label"],
        "decodability": decod["decodability"],
        "illegal_words": decod["illegal_words"],
        "top_violations": decod["top_violations"],
        "snippet": text[:80],
    }
    log_prediction(out)
    return out


@app.route("/")
def home():
    return render_template("index.html", lessons=available_lessons(),
                           version=MODEL_VERSION)


@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    text = request.form.get("text") or request.args.get("text", "")
    lesson = request.form.get("lesson") or request.args.get("lesson", 34)
    if not text.strip():
        return render_template("index.html", lessons=available_lessons(),
                               version=MODEL_VERSION, error="Paste some text first.")
    r = analyse(text, lesson)
    return render_template("result.html", r=r, text=text)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        data = request.get_json(silent=True) or request.form
    else:
        data = request.args
    text = (data.get("text") or "").strip()
    lesson = data.get("lesson", 34)
    if not text:
        return jsonify({"code": 400, "message": "field 'text' is required"}), 400
    r = analyse(text, lesson)
    return jsonify({"code": 200, "message": "ok", **r})


@app.route("/benchmark")
def benchmark():
    if not METRICS:
        return jsonify({"code": 503, "message": "model not trained yet"}), 503
    return jsonify({"code": 200, "benchmark": METRICS.get("benchmark", []),
                    "winner": METRICS.get("winner", {}),
                    "tuning_curve": METRICS.get("tuning_curve", []),
                    "feature_importance": METRICS.get("feature_importance", []),
                    "dataset": METRICS.get("dataset"),
                    "n_train": METRICS.get("n_train"), "n_test": METRICS.get("n_test")})


@app.route("/history")
def history():
    n = int(request.args.get("n", 100))
    try:
        c = sqlite3.connect(DB)
        c.row_factory = sqlite3.Row
        rows = c.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?",
                         (n,)).fetchall()
        c.close()
        return jsonify({"code": 200, "count": len(rows),
                        "rows": [dict(r) for r in rows]})
    except Exception as e:
        return jsonify({"code": 500, "message": str(e)}), 500


@app.route("/metrics")
def metrics():
    try:
        c = sqlite3.connect(DB)
        row = c.execute("""SELECT COUNT(*), AVG(readability), AVG(decodability),
                           AVG(word_count), AVG(fk_grade) FROM predictions""").fetchone()
        bands = c.execute("""SELECT band, COUNT(*) FROM predictions
                             GROUP BY band ORDER BY 2 DESC""").fetchall()
        c.close()
        return jsonify({
            "code": 200,
            "total_predictions": row[0] or 0,
            "mean_readability": round(row[1], 4) if row[1] is not None else None,
            "mean_decodability": round(row[2], 2) if row[2] is not None else None,
            "mean_word_count": round(row[3], 1) if row[3] is not None else None,
            "mean_fk_grade": round(row[4], 2) if row[4] is not None else None,
            "band_distribution": [{"band": b, "count": c_} for b, c_ in bands],
            "model_version": MODEL_VERSION,
            "model_loaded": MODEL is not None,
        })
    except Exception as e:
        return jsonify({"code": 500, "message": str(e)}), 500


@app.route("/lessons")
def lessons():
    return jsonify({"code": 200, "lessons": available_lessons()})


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL is not None,
                    "model_version": MODEL_VERSION, "service": "readfit"})


@app.route("/reload")
def reload_model():
    load_model()
    return jsonify({"code": 200, "model_loaded": MODEL is not None,
                    "model_version": MODEL_VERSION})


init_db()
load_model()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
