import os
import json
from pathlib import Path
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ── Paths ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_CSV = BASE_DIR / "models" / "data" / "cleansing_water_data.csv"
MODEL_PERF_JSON = BASE_DIR / "models" / "supervise" / "model_performance.json"
IMAGES_DIR = BASE_DIR / "models" / "images"

# ── FastAPI app ────────────────────────────────────────
app = FastAPI(title="Analytics API", version="1.0.0")

# Allow frontend to call backend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve model images as static files
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")


# ── Helpers ────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    return pd.read_csv(str(DATA_CSV))


def safe_value_counts(series: pd.Series, top_n: int | None = None) -> dict:
    vc = series.dropna().value_counts()
    if top_n:
        vc = vc.head(top_n)
    return {str(k): int(v) for k, v in vc.items()}


# ── API Endpoints ──────────────────────────────────────

@app.get("/api/overview")
def get_overview():
    df = load_data()
    cleansing_users = int((df["use_cleansing_water"] == "ใช้").sum())
    non_users = int((df["use_cleansing_water"] != "ใช้").sum())

    return {
        "total_respondents": len(df),
        "total_features": len(df.columns),
        "cleansing_water_users": cleansing_users,
        "non_cleansing_water_users": non_users,
        "missing_values": int(df.isnull().sum().sum()),
    }


@app.get("/api/demographics")
def get_demographics():
    df = load_data()

    return {
        "sex": safe_value_counts(df["sex"]),
        "age": safe_value_counts(df["age"]),
        "skin_type": safe_value_counts(df["skin_type"]),
        "occupation": safe_value_counts(df["occupation"], top_n=8),
        "monthly_income": safe_value_counts(df["monthly_income"]),
        "acne_level": safe_value_counts(df["acne_level"]),
    }


@app.get("/api/brands")
def get_brands():
    df = load_data()

    return {
        "brand_primary": safe_value_counts(df["brand_primary"], top_n=10),
    }


@app.get("/api/factors")
def get_factors():
    df = load_data()
    factor_cols = [c for c in df.columns if c.startswith("factor_")]
    means = df[factor_cols].mean().round(2)

    factors = {}
    for col in factor_cols:
        clean_name = col.replace("factor_", "").replace("_", " ").title()
        factors[clean_name] = float(means[col])

    return {"factors": factors}


@app.get("/api/model-performance")
def get_model_performance():
    with open(str(MODEL_PERF_JSON), "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@app.get("/api/home-summary")
def get_home_summary():
    df = load_data()
    cleansing_users = int((df["use_cleansing_water"] == "ใช้").sum())
    factor_cols = [c for c in df.columns if c.startswith("factor_")]
    means = df[factor_cols].mean().round(2)

    # Top 5 brands
    brand_vc = df["brand_primary"].dropna().value_counts().head(5)

    # Factor scores formatted
    factors = {}
    for col in factor_cols:
        clean_name = col.replace("factor_", "").replace("_", " ").title()
        factors[clean_name] = float(means[col])

    # Sort factors by value descending
    factors = dict(sorted(factors.items(), key=lambda x: x[1], reverse=True))

    return {
        "overview": {
            "total_respondents": len(df),
            "total_features": len(df.columns),
            "cleansing_water_users": cleansing_users,
            "non_cleansing_water_users": len(df) - cleansing_users,
        },
        "demographics": {
            "sex": safe_value_counts(df["sex"]),
            "age": safe_value_counts(df["age"]),
            "skin_type": safe_value_counts(df["skin_type"]),
        },
        "top_brands": {str(k): int(v) for k, v in brand_vc.items()},
        "factors": factors,
    }


# ── Run ────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
