import os
import sys
import json
from pathlib import Path

# Ensure backend directory is in sys.path for absolute imports
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np
from pydantic import BaseModel
from typing import List

import database as db

BASE_DIR = CURRENT_DIR.parent
UNSUP_PROFILES_JSON = BASE_DIR / "models" / "unsupervise" / "cluster_profiles.json"
MODEL_PERF_JSON = BASE_DIR / "models" / "supervise" / "model_performance.json"
IMAGES_DIR = BASE_DIR / "models" / "images"

app = FastAPI(title="Analytics API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")


@app.get("/api/overview")
def get_overview():
    cleansing_users = db.count_rows(where="use_cleansing_water = 'ใช้'")
    total_rows = db.count_rows()

    return {
        "total_respondents": total_rows,
        "total_features": db.total_columns(),
        "cleansing_water_users": cleansing_users,
        "non_cleansing_water_users": total_rows - cleansing_users,
        "missing_values": db.missing_values_count(),
    }


@app.get("/api/demographics")
def get_demographics():
    return {
        "sex": db.value_counts("sex"),
        "age": db.value_counts("age"),
        "skin_type": db.value_counts("skin_type"),
        "occupation": db.value_counts("occupation", top_n=8),
        "monthly_income": db.value_counts("monthly_income"),
        "acne_level": db.value_counts("acne_level"),
    }


@app.get("/api/brands")
def get_brands():
    return {
        "brand_primary": db.value_counts("brand_primary", top_n=10),
    }


@app.get("/api/factors")
def get_factors():
    means = db.factor_means()
    factors = {}
    for col, mean_val in means.items():
        if mean_val is not None:
            clean_name = col.replace("factor_", "").replace("_", " ").title()
            factors[clean_name] = float(mean_val)

    return {"factors": factors}


@app.get("/api/unsupervise")
def get_unsupervise_profiles():
    try:
        with open(str(UNSUP_PROFILES_JSON), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/model-performance")
def get_model_performance():
    return db.get_model_performance()


@app.get("/api/home-summary")
def get_home_summary():
    cleansing_users = db.count_rows(where="use_cleansing_water = 'ใช้'")
    total_rows = db.count_rows()
    
    means = db.factor_means()
    brand_vc = db.value_counts("brand_primary", top_n=5)

    factors = {}
    for col, mean_val in means.items():
        if mean_val is not None:
            clean_name = col.replace("factor_", "").replace("_", " ").title()
            factors[clean_name] = float(mean_val)

    factors = dict(sorted(factors.items(), key=lambda x: x[1], reverse=True))

    return {
        "overview": {
            "total_respondents": total_rows,
            "total_features": db.total_columns(),
            "cleansing_water_users": cleansing_users,
            "non_cleansing_water_users": total_rows - cleansing_users,
        },
        "demographics": {
            "sex": db.value_counts("sex"),
            "age": db.value_counts("age"),
            "skin_type": db.value_counts("skin_type"),
        },
        "top_brands": {str(k): int(v) for k, v in brand_vc.items()},
        "factors": factors,
    }
    
# ── Prediction ─────────────────────────────────────────
MODEL_PKL = BASE_DIR / "models" / "supervise" / "best_model.pkl"

class PredictRequest(BaseModel):
    sex: str
    skin_type: str
    concerns: List[str]
    factor_deep_cleansing: float = 3.0
    factor_sensitive_friendly: float = 3.0
    factor_oil_control: float = 3.0

@app.get("/api/business-insight")
def get_business_insight():
    total = db.count_rows()
    cleansing_users = db.count_rows(where="use_cleansing_water = 'ใช้'")

    # Kiyora users (brands_used contains 'Kiyora')
    kiyora_users = db.count_rows(where="brands_used LIKE '%Kiyora%'")
    kiyora_primary = db.count_rows(where="brand_primary LIKE '%Kiyora%'")

    # Demographics for Kiyora users
    kiyora_age = db.value_counts("age", where="brands_used LIKE '%Kiyora%'")
    kiyora_sex = db.value_counts("sex", where="brands_used LIKE '%Kiyora%'")
    kiyora_skin = db.value_counts("skin_type", where="brands_used LIKE '%Kiyora%'")
    kiyora_income = db.value_counts("monthly_income", where="brands_used LIKE '%Kiyora%'")
    kiyora_occupation = db.value_counts("occupation", top_n=5, where="brands_used LIKE '%Kiyora%'")
    kiyora_concerns = db.value_counts("concerns", top_n=8, where="brands_used LIKE '%Kiyora%'")
    kiyora_switch = db.value_counts("switch_factors", top_n=5, where="brands_used LIKE '%Kiyora%'")

    # Factor means for Kiyora vs overall
    kiyora_factors = db.factor_means(where="brands_used LIKE '%Kiyora%'")
    overall_factors = db.factor_means(where="use_cleansing_water = 'ใช้'")

    # Top brands for comparison
    brand_vc = db.value_counts("brand_primary", top_n=5)

    # Brand core values (factor means per top brand)
    brand_core = {}
    for brand_name in list(brand_vc.keys())[:5]:
        brand_factors = db.factor_means(where=f"brand_primary = ?", params=(brand_name,))
        cleaned = {}
        for col, val in brand_factors.items():
            clean_name = col.replace("factor_", "").replace("_", " ").title()
            cleaned[clean_name] = float(val) if val else 0.0
        brand_core[brand_name] = cleaned

    # Clean factor names
    def clean_factors(raw):
        cleaned = {}
        for col, val in raw.items():
            clean_name = col.replace("factor_", "").replace("_", " ").title()
            cleaned[clean_name] = float(val) if val else 0.0
        return dict(sorted(cleaned.items(), key=lambda x: x[1], reverse=True))

    kiyora_f = clean_factors(kiyora_factors)
    overall_f = clean_factors(overall_factors)

    # Find Kiyora core values (top 3 strengths)
    core_values = list(kiyora_f.items())[:3]

    # Find gaps (areas where Kiyora scores lower than overall)
    gaps = []
    for k, v in kiyora_f.items():
        ov = overall_f.get(k, 0)
        if ov > v:
            gaps.append({"factor": k, "kiyora": v, "overall": ov, "gap": round(ov - v, 2)})
    gaps.sort(key=lambda x: x["gap"], reverse=True)

    return {
        "overview": {
            "total_respondents": total,
            "cleansing_water_users": cleansing_users,
            "kiyora_users": kiyora_users,
            "kiyora_primary_users": kiyora_primary,
            "kiyora_usage_rate": round((kiyora_users / cleansing_users * 100), 1) if cleansing_users else 0,
        },
        "demographics": {
            "age": kiyora_age,
            "sex": kiyora_sex,
            "skin_type": kiyora_skin,
            "income": kiyora_income,
            "occupation": kiyora_occupation,
        },
        "concerns": kiyora_concerns,
        "switch_factors": kiyora_switch,
        "factors": {
            "kiyora": kiyora_f,
            "overall": overall_f,
        },
        "core_values": [{"factor": cv[0], "score": cv[1]} for cv in core_values],
        "improvement_gaps": gaps[:5],
        "brand_comparison": brand_core,
        "brand_market_share": {str(k): int(v) for k, v in brand_vc.items()},
    }


@app.post("/api/predict")
def predict_customer(req: PredictRequest):
    try:
        model_data = joblib.load(str(MODEL_PKL))
        if isinstance(model_data, dict):
            model = model_data['model']
            scaler = model_data.get('scaler')
            feature_names = model_data.get('feature_names')
        else:
            model = model_data
            scaler = None
            feature_names = list(model.feature_names_in_)
    except Exception as e:
        return {"error": f"Cannot load model: {e}"}

    if feature_names is None:
        feature_names = list(model.feature_names_in_)

    feature_vec = {f: 0.0 for f in feature_names}

    # Factor scores
    if "factor_deep_cleansing" in feature_vec:
        feature_vec["factor_deep_cleansing"] = req.factor_deep_cleansing
    if "factor_sensitive_friendly" in feature_vec:
        feature_vec["factor_sensitive_friendly"] = req.factor_sensitive_friendly
    if "factor_oil_control" in feature_vec:
        feature_vec["factor_oil_control"] = req.factor_oil_control

    # Skin type one-hot
    for f in feature_names:
        if f.startswith("skin_") and req.skin_type in f:
            feature_vec[f] = 1.0

    # Concerns one-hot
    for concern in req.concerns:
        for f in feature_names:
            if f.startswith("concern") and concern in f:
                feature_vec[f] = 1.0

    import pandas as pd_inner
    X_input = pd_inner.DataFrame([feature_vec], columns=feature_names)

    if scaler is not None:
        X_scaled = scaler.transform(X_input)
    else:
        X_scaled = X_input

    prediction = int(model.predict(X_scaled)[0])
    probabilities = model.predict_proba(X_scaled)[0].tolist()

    return {
        "prediction": prediction,
        "is_kiyora_customer": bool(prediction == 1),
        "probability": {
            "not_kiyora": round(probabilities[0], 4),
            "kiyora": round(probabilities[1], 4),
        },
        "input_features": feature_vec,
        "sex": req.sex,
    }


# ── Run ────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
