"""
app.py
======
Flask API for the Product Sales Forecasting model.

Endpoints
---------
GET  /              → Interactive dashboard UI
GET  /health        → API health check
GET  /model/info    → Model metadata and validation metrics
POST /predict       → Single-record prediction (JSON)
POST /predict/batch → Batch predictions (JSON array or CSV upload)

Run
---
    python app.py               # development
    gunicorn app:app -w 4       # production
"""

import io
import json
import os
import time
import traceback
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# ── Load artifacts on startup ─────────────────────────────────────────────────
print("Loading model artifacts ...")

model         = joblib.load(os.path.join(MODELS_DIR, "lgb_model.pkl"))
FEATURES      = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))

with open(os.path.join(MODELS_DIR, "store_stats.json")) as f:
    STORE_STATS = {int(k): v for k, v in json.load(f).items()}

with open(os.path.join(MODELS_DIR, "metrics.json")) as f:
    MODEL_METRICS = json.load(f)

print(f"  Model loaded  |  {len(FEATURES)} features  |  {len(STORE_STATS)} stores")

# ── Inference helpers ─────────────────────────────────────────────────────────

def get_lag_features(store_id: int) -> dict:
    """Return lag feature values for a given store_id from the precomputed stats."""
    stats = STORE_STATS.get(store_id, {})
    mean28 = stats.get("mean28", np.mean([v["mean28"] for v in STORE_STATS.values()]))
    mean7  = stats.get("mean7",  np.mean([v["mean7"]  for v in STORE_STATS.values()]))
    std7   = stats.get("std7",   np.mean([v["std7"]   for v in STORE_STATS.values()]))
    return {
        "sales_lag_7":     mean7,
        "sales_lag_14":    mean28,
        "sales_lag_28":    mean28,
        "rolling_mean_7":  mean7,
        "rolling_mean_28": mean28,
        "rolling_std_7":   std7,
    }


def build_feature_row(record: dict) -> pd.DataFrame:
    """
    Convert a raw input record into the full engineered feature DataFrame.

    Expected input keys
    -------------------
    Store_id      : int
    Store_Type    : "S1" | "S2" | "S3" | "S4"
    Location_Type : "L1" | "L2" | "L3" | "L4" | "L5"
    Region_Code   : "R1" | "R2" | "R3" | "R4"
    Date          : "YYYY-MM-DD"
    Holiday       : 0 | 1
    Discount      : "Yes" | "No"
    """
    date = pd.Timestamp(record["Date"])
    store_id = int(record["Store_id"])

    row = {}

    # Lag features (from precomputed store statistics)
    row.update(get_lag_features(store_id))

    # Time features
    row["Year"]        = date.year
    row["Month"]       = date.month
    row["DayOfWeek"]   = date.dayofweek
    row["Quarter"]     = date.quarter
    row["WeekOfYear"]  = date.isocalendar()[1]
    row["DayOfYear"]   = date.dayofyear
    row["IsWeekend"]   = int(date.dayofweek >= 5)
    row["IsMonthStart"] = int(date.is_month_start)
    row["IsMonthEnd"]   = int(date.is_month_end)

    # Cyclical encodings
    row["Month_sin"] = np.sin(2 * np.pi * row["Month"] / 12)
    row["Month_cos"] = np.cos(2 * np.pi * row["Month"] / 12)
    row["Week_sin"]  = np.sin(2 * np.pi * row["WeekOfYear"] / 52)
    row["Week_cos"]  = np.cos(2 * np.pi * row["WeekOfYear"] / 52)
    row["DOW_sin"]   = np.sin(2 * np.pi * row["DayOfWeek"] / 7)
    row["DOW_cos"]   = np.cos(2 * np.pi * row["DayOfWeek"] / 7)

    # Label encodings
    discount_enc               = 1 if record.get("Discount", "No") == "Yes" else 0
    row["Store_id"]            = store_id
    row["Holiday"]             = int(record.get("Holiday", 0))
    row["Discount_enc"]        = discount_enc
    row["Store_Type_enc"]      = {"S1": 1, "S2": 2, "S3": 3, "S4": 4}.get(record["Store_Type"], 1)
    row["Location_Type_enc"]   = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5}.get(record["Location_Type"], 1)
    row["Region_Code_enc"]     = {"R1": 1, "R2": 2, "R3": 3, "R4": 4}.get(record["Region_Code"], 1)
    row["Holiday_Discount"]    = row["Holiday"] * discount_enc
    row["Weekend_Discount"]    = row["IsWeekend"] * discount_enc

    # One-hot: Store_Type
    for st in ["S1", "S2", "S3", "S4"]:
        row[f"Store_Type_{st}"] = int(record["Store_Type"] == st)

    # One-hot: Location_Type
    for lt in ["L1", "L2", "L3", "L4", "L5"]:
        row[f"Location_Type_{lt}"] = int(record["Location_Type"] == lt)

    # One-hot: Region_Code
    for rc in ["R1", "R2", "R3", "R4"]:
        row[f"Region_Code_{rc}"] = int(record["Region_Code"] == rc)

    df = pd.DataFrame([row])

    # Align with training feature order — fill any missing with 0
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0

    return df[FEATURES].astype(float)


def validate_record(record: dict) -> list:
    """Return a list of validation error messages (empty = valid)."""
    errors = []
    required = ["Store_id", "Store_Type", "Location_Type", "Region_Code", "Date"]
    for field in required:
        if field not in record:
            errors.append(f"Missing required field: '{field}'")

    if "Store_Type" in record and record["Store_Type"] not in ["S1","S2","S3","S4"]:
        errors.append(f"Invalid Store_Type '{record['Store_Type']}'. Must be S1–S4.")
    if "Location_Type" in record and record["Location_Type"] not in ["L1","L2","L3","L4","L5"]:
        errors.append(f"Invalid Location_Type '{record['Location_Type']}'. Must be L1–L5.")
    if "Region_Code" in record and record["Region_Code"] not in ["R1","R2","R3","R4"]:
        errors.append(f"Invalid Region_Code '{record['Region_Code']}'. Must be R1–R4.")
    if "Discount" in record and record["Discount"] not in ["Yes","No"]:
        errors.append(f"Invalid Discount '{record['Discount']}'. Must be 'Yes' or 'No'.")
    if "Holiday" in record and record["Holiday"] not in [0, 1, "0", "1"]:
        errors.append(f"Invalid Holiday '{record['Holiday']}'. Must be 0 or 1.")
    if "Date" in record:
        try:
            pd.Timestamp(record["Date"])
        except Exception:
            errors.append(f"Invalid Date format '{record['Date']}'. Use YYYY-MM-DD.")

    return errors


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the interactive dashboard."""
    return render_template("index.html", metrics=MODEL_METRICS)


@app.route("/health", methods=["GET"])
def health():
    """API health check."""
    return jsonify({
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model": "LightGBM Sales Forecasting",
        "version": "1.0.0",
        "features": len(FEATURES),
        "stores": len(STORE_STATS),
    })


@app.route("/model/info", methods=["GET"])
def model_info():
    """Return model metadata and validation performance metrics."""
    return jsonify({
        "model_type": "LightGBM (Optuna-tuned)",
        "training_period": "Jan 2018 – May 2019",
        "prediction_target": "Daily Sales (₹) per store",
        "validation_metrics": MODEL_METRICS,
        "feature_count": len(FEATURES),
        "store_count": len(STORE_STATS),
        "features": FEATURES,
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Single-record prediction.

    Request body (JSON)
    -------------------
    {
        "Store_id"      : 1,
        "Store_Type"    : "S1",
        "Location_Type" : "L3",
        "Region_Code"   : "R1",
        "Date"          : "2019-06-15",
        "Holiday"       : 0,
        "Discount"      : "Yes"
    }

    Response
    --------
    {
        "predicted_sales": 42500.75,
        "store_id": 1,
        "date": "2019-06-15",
        "store_type": "S1",
        "discount": "Yes",
        "holiday": 0,
        "inference_time_ms": 12.4
    }
    """
    t0 = time.time()
    try:
        record = request.get_json(force=True)
        if not record:
            return jsonify({"error": "Request body must be valid JSON."}), 400

        errors = validate_record(record)
        if errors:
            return jsonify({"error": "Validation failed.", "details": errors}), 422

        features_df = build_feature_row(record)
        prediction  = float(np.clip(model.predict(features_df)[0], 0, None))

        return jsonify({
            "predicted_sales":    round(prediction, 2),
            "store_id":           int(record["Store_id"]),
            "date":               record["Date"],
            "store_type":         record["Store_Type"],
            "location_type":      record.get("Location_Type"),
            "region_code":        record.get("Region_Code"),
            "discount":           record.get("Discount", "No"),
            "holiday":            int(record.get("Holiday", 0)),
            "inference_time_ms":  round((time.time() - t0) * 1000, 2),
        })

    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error": "Internal server error.", "detail": str(e)}), 500


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """
    Batch prediction — accepts either a JSON array or a CSV file upload.

    JSON request
    ------------
    [
        {"Store_id": 1, "Store_Type": "S1", ..., "Date": "2019-06-01"},
        {"Store_id": 2, "Store_Type": "S4", ..., "Date": "2019-06-01"}
    ]

    CSV upload (multipart/form-data)
    --------------------------------
    Field name: file
    Required columns: Store_id, Store_Type, Location_Type,
                      Region_Code, Date, Holiday, Discount

    Response
    --------
    {
        "predictions": [...],
        "count": 2,
        "inference_time_ms": 18.3
    }
    """
    t0 = time.time()
    try:
        # ── CSV upload ────────────────────────────────────────────────────────
        if "file" in request.files:
            file = request.files["file"]
            if not file.filename.endswith(".csv"):
                return jsonify({"error": "Uploaded file must be a CSV."}), 400
            df_in = pd.read_csv(io.StringIO(file.read().decode("utf-8")))
            records = df_in.to_dict(orient="records")

        # ── JSON body ─────────────────────────────────────────────────────────
        else:
            records = request.get_json(force=True)
            if not isinstance(records, list):
                return jsonify({"error": "Request body must be a JSON array of records."}), 400

        if not records:
            return jsonify({"error": "No records provided."}), 400

        if len(records) > 5000:
            return jsonify({"error": "Batch limit is 5,000 records per request."}), 413

        # ── Validate & predict ────────────────────────────────────────────────
        all_errors = []
        feature_rows = []
        for i, rec in enumerate(records):
            errs = validate_record(rec)
            if errs:
                all_errors.append({"row": i, "errors": errs})
            else:
                feature_rows.append(build_feature_row(rec))

        if all_errors:
            return jsonify({
                "error": "Validation failed for one or more records.",
                "failed_rows": all_errors,
            }), 422

        X_batch = pd.concat(feature_rows, ignore_index=True)
        preds   = np.clip(model.predict(X_batch), 0, None)

        predictions = []
        for rec, pred in zip(records, preds):
            predictions.append({
                "store_id":        int(rec["Store_id"]),
                "date":            rec["Date"],
                "store_type":      rec["Store_Type"],
                "discount":        rec.get("Discount", "No"),
                "holiday":         int(rec.get("Holiday", 0)),
                "predicted_sales": round(float(pred), 2),
            })

        return jsonify({
            "predictions":       predictions,
            "count":             len(predictions),
            "inference_time_ms": round((time.time() - t0) * 1000, 2),
        })

    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error": "Internal server error.", "detail": str(e)}), 500


# ── Error handlers ────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found.", "available_routes": [
        "GET  /",
        "GET  /health",
        "GET  /model/info",
        "POST /predict",
        "POST /predict/batch",
    ]}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed."}), 405


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
