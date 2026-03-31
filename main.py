"""
skpro Uncertainty Explorer — FastAPI Backend
A web app that accepts CSV uploads and returns probabilistic predictions with uncertainty bands.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
import numpy as np
import io
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from skpro.regression.residual import ResidualDouble
from skpro.metrics import CRPS, PinballLoss
from sklearn.metrics import mean_absolute_error

app = FastAPI(
    title="skpro Uncertainty Explorer",
    description="Upload a CSV dataset and get probabilistic predictions with uncertainty estimates.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("templates/index.html", "r") as f:
        return f.read()


@app.get("/health")
async def health():
    return {"status": "ok", "message": "skpro Uncertainty Explorer is running"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), target_column: str = "target"):
    """
    Upload a CSV file and receive probabilistic regression results.

    - **file**: CSV file with numeric columns
    - **target_column**: name of the column to predict (default: 'target')
    """

    # ── 1. Read CSV ──────────────────────────────────────────────────────────
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {str(e)}")

    if target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Column '{target_column}' not found. Available columns: {list(df.columns)}"
        )

    # ── 2. Prepare features ──────────────────────────────────────────────────
    df = df.dropna()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != target_column]

    if len(feature_cols) == 0:
        raise HTTPException(status_code=400, detail="No numeric feature columns found.")

    if len(df) < 20:
        raise HTTPException(status_code=400, detail="Dataset too small — need at least 20 rows.")

    X = df[feature_cols]
    y = df[target_column]

    # ── 3. Train/test split ──────────────────────────────────────────────────
    test_size = min(0.25, max(0.1, 50 / len(df)))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # ── 4. Scale features ────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
    X_test_sc  = pd.DataFrame(scaler.transform(X_test),      columns=feature_cols, index=X_test.index)

    # ── 5. Standard Ridge (point predictions) ────────────────────────────────
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_sc, y_train)
    y_point = ridge.predict(X_test_sc)
    mae_standard = mean_absolute_error(y_test, y_point)

    # ── 6. skpro Probabilistic Regression ────────────────────────────────────
    prob_model = ResidualDouble(
        estimator=Ridge(alpha=1.0),
        estimator_resid=Ridge(alpha=1.0)
    )
    prob_model.fit(X_train_sc, y_train)
    y_dist = prob_model.predict_proba(X_test_sc)

    y_mean = y_dist.mean().values.flatten()
    y_std  = y_dist.var().values.flatten() ** 0.5

    lower_80 = (y_mean - 1.282 * y_std).tolist()
    upper_80 = (y_mean + 1.282 * y_std).tolist()
    lower_95 = (y_mean - 1.960 * y_std).tolist()
    upper_95 = (y_mean + 1.960 * y_std).tolist()

    # ── 7. Metrics ───────────────────────────────────────────────────────────
    mae_prob = mean_absolute_error(y_test, y_mean)
    crps_score  = float(CRPS()(y_test, y_dist))
    pinball_10  = float(PinballLoss(alpha=0.1)(y_test, y_dist))
    pinball_90  = float(PinballLoss(alpha=0.9)(y_test, y_dist))

    actual_arr = y_test.values
    coverage_80 = float(np.mean((actual_arr >= lower_80) & (actual_arr <= upper_80)))
    coverage_95 = float(np.mean((actual_arr >= lower_95) & (actual_arr <= upper_95)))

    # ── 8. Sort by actual for clean chart ────────────────────────────────────
    n_show = min(100, len(y_test))
    sort_idx = np.argsort(actual_arr[:n_show])

    return JSONResponse({
        "dataset_info": {
            "rows": len(df),
            "features": feature_cols,
            "target": target_column,
            "train_size": len(X_train),
            "test_size": len(X_test),
        },
        "metrics": {
            "mae_standard": round(mae_standard, 4),
            "mae_probabilistic": round(mae_prob, 4),
            "crps": round(crps_score, 4),
            "pinball_10": round(pinball_10, 4),
            "pinball_90": round(pinball_90, 4),
            "coverage_80": round(coverage_80 * 100, 1),
            "coverage_95": round(coverage_95 * 100, 1),
        },
        "chart_data": {
            "actual":   actual_arr[:n_show][sort_idx].tolist(),
            "mean":     y_mean[:n_show][sort_idx].tolist(),
            "lower_80": np.array(lower_80)[:n_show][sort_idx].tolist(),
            "upper_80": np.array(upper_80)[:n_show][sort_idx].tolist(),
            "lower_95": np.array(lower_95)[:n_show][sort_idx].tolist(),
            "upper_95": np.array(upper_95)[:n_show][sort_idx].tolist(),
            "std":      y_std[:n_show][sort_idx].tolist(),
        }
    })


@app.get("/sample-csv")
async def sample_csv():
    """Returns info about the built-in sample dataset."""
    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing(as_frame=True)
    df = pd.concat([housing.data, housing.target], axis=1)
    sample = df.head(200)
    return JSONResponse({
        "message": "Sample dataset: California Housing (200 rows)",
        "columns": list(sample.columns),
        "target_column": "MedHouseVal",
        "csv": sample.to_csv(index=False)
    })
