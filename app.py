from pathlib import Path
from contextlib import closing

from flask import Flask, request, jsonify, render_template, abort
from flask_cors import CORS
import joblib
import pandas as pd
import sqlite3
import os

MODEL_PATH = Path(os.getenv(r"C:\Users\hrock\AnemiaSense\model.pkl", "model.pkl"))
DB_URI     = "anemia.db"

app = Flask(__name__)
CORS(app)

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError as e:
    raise RuntimeError(f"❌  model.pkl not found at {MODEL_PATH!s}") from e

with closing(sqlite3.connect(DB_URI)) as conn, conn:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS history (
            hemoglobin REAL,
            rbc        REAL,
            mcv        REAL,
            prediction TEXT
        )
        """
    )

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)
    if not payload:
        abort(400, "No JSON body found.")

    EXPECTED_COLS = ["Gender", "Hemoglobin", "MCH", "MCHC", "MCV"]

    row = {
        "Gender":     float(payload.get("gender", 0)),
        "Hemoglobin": float(payload.get("hemoglobin", 0)),
        "MCH":        float(payload.get("mch", 0)),
        "MCHC":       float(payload.get("mchc", 0)),
        "MCV":        float(payload.get("mcv", 0)),
    }

    df = pd.DataFrame([[row[col] for col in EXPECTED_COLS]], columns=EXPECTED_COLS)

    prediction = model.predict(df)[0]

    with closing(sqlite3.connect(DB_URI, check_same_thread=False)) as conn, conn:
        conn.execute(
            "INSERT INTO history VALUES (?, ?, ?, ?)",
            (row["Hemoglobin"], row["Gender"], row["MCV"], str(prediction)),
        )

    return jsonify({"prediction": str(prediction)})

@app.route("/history", methods=["GET"])
def history():
    with closing(sqlite3.connect(DB_URI, check_same_thread=False)) as conn:
        rows = conn.execute("SELECT * FROM history").fetchall()
    return jsonify(rows)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
