"""Hotspot prediction logic (XGBoost + LSTM feature).

This module is designed to be imported by an API server.

Expected project layout (relative to base_dir):
  data/
    test_hourly.parquet
    test_lstm_pred.csv
    taxi_zone_centroids.csv
  model/
    xgb_xgb_lstm_feat.model
  outputs/
    (optional; will be created)

Output schema (zones list):
  [{PULocationID, pred_rides, Borough, Zone, lat_wgs, lon_wgs}, ...]
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from pyproj import Transformer


ZONE_COL = "PULocationID"
TIME_COL = "pickup_hour"
Y_COL = "rides"
LSTM_COL = "lstm_pred_rides"

LAGS = [1, 2, 3, 24]
ROLLS = [3, 6, 24]


def _time_features_for(dt: pd.Timestamp) -> Dict[str, float]:
    hour = int(dt.hour)
    dow = int(dt.dayofweek)
    is_weekend = 1 if dow >= 5 else 0
    hour_sin = float(np.sin(2 * np.pi * hour / 24))
    hour_cos = float(np.cos(2 * np.pi * hour / 24))
    dow_sin = float(np.sin(2 * np.pi * dow / 7))
    dow_cos = float(np.cos(2 * np.pi * dow / 7))
    return {
        "hour": hour,
        "dow": dow,
        "is_weekend": is_weekend,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
    }


def _series_lag_roll_features(y: np.ndarray) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    # Lags
    for k in LAGS:
        feats[f"lag_{k}"] = float(y[-k]) if len(y) >= k else 0.0
    # Rolls on shifted(1): for next-hour row, that's just last w values.
    for w in ROLLS:
        window = y[-w:] if len(y) >= w else y
        if window.size == 0:
            feats[f"roll_mean_{w}"] = 0.0
            feats[f"roll_std_{w}"] = 0.0
        else:
            feats[f"roll_mean_{w}"] = float(np.mean(window))
            feats[f"roll_std_{w}"] = float(np.std(window, ddof=1)) if window.size >= 2 else 0.0
    return feats


def _load_lstm_next_hour_map(lstm_path: Path, next_hour: pd.Timestamp) -> Dict[int, float]:
    if not lstm_path.exists():
        return {}
    lstm = pd.read_csv(lstm_path)
    if TIME_COL not in lstm.columns:
        # allow 'predict_hour' as fallback
        if "predict_hour" in lstm.columns:
            lstm[TIME_COL] = lstm["predict_hour"]
        else:
            return {}
    lstm[TIME_COL] = pd.to_datetime(lstm[TIME_COL])
    if LSTM_COL not in lstm.columns:
        return {}
    lstm = lstm[[ZONE_COL, TIME_COL, LSTM_COL]].copy()
    lstm = lstm.groupby([ZONE_COL, TIME_COL], as_index=False)[LSTM_COL].mean()
    lstm_next = lstm[lstm[TIME_COL] == next_hour]
    return {int(r[ZONE_COL]): float(r[LSTM_COL]) for _, r in lstm_next.iterrows()}


def _load_centroids_wgs84(centroid_path: Path) -> pd.DataFrame:
    df_cent = pd.read_csv(centroid_path)
    # Expected EPSG:2263 lon/lat columns named lon/lat in source.
    transformer = Transformer.from_crs("EPSG:2263", "EPSG:4326", always_xy=True)
    lon_wgs, lat_wgs = transformer.transform(df_cent["lon"].values, df_cent["lat"].values)
    df_cent["lon_wgs"] = np.round(lon_wgs, 6)
    df_cent["lat_wgs"] = np.round(lat_wgs, 6)
    keep_cols = ["LocationID", "Borough", "Zone", "lat_wgs", "lon_wgs"]
    return df_cent[keep_cols]


def _feature_columns() -> List[str]:
    return [
        "hour",
        "dow",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        *[f"lag_{k}" for k in LAGS],
        *[f"roll_mean_{w}" for w in ROLLS],
        *[f"roll_std_{w}" for w in ROLLS],
        LSTM_COL,
    ]


def compute_zones_payload(base_dir: Path) -> Dict[str, Any]:
    """Compute next-hour prediction for all zones and return JSON-ready payload."""

    data_dir = base_dir / "data"
    model_dir = base_dir / "model"
    out_dir = base_dir / "outputs"
    out_dir.mkdir(exist_ok=True)

    hourly_path = data_dir / "test_hourly.parquet"
    lstm_path = data_dir / "test_lstm_pred.csv"
    centroid_path = data_dir / "taxi_zone_centroids.csv"
    model_path = model_dir / "xgb_xgb_lstm_feat.model"

    if not hourly_path.exists():
        raise FileNotFoundError(f"Missing hourly data: {hourly_path}")
    if not centroid_path.exists():
        raise FileNotFoundError(f"Missing centroid data: {centroid_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")

    df = pd.read_parquet(hourly_path)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    # Safety aggregation
    df = df.groupby([ZONE_COL, TIME_COL], as_index=False)[Y_COL].sum()

    last_hour = df[TIME_COL].max()
    next_hour = last_hour + pd.Timedelta(hours=1)

    # LSTM predictions for next hour (optional)
    lstm_map = _load_lstm_next_hour_map(lstm_path, next_hour)

    rows: List[Dict[str, Any]] = []
    time_feats = _time_features_for(next_hour)

    # build one feature row per zone
    for loc_id, g in df.groupby(ZONE_COL):
        g = g.sort_values(TIME_COL)
        y = g[Y_COL].to_numpy(dtype=float)

        # Keep behavior similar to your original script: require 24 hours to be stable.
        if len(y) < 24:
            continue

        feats = {ZONE_COL: int(loc_id), TIME_COL: next_hour}
        feats.update(time_feats)
        feats.update(_series_lag_roll_features(y))
        feats[LSTM_COL] = float(lstm_map.get(int(loc_id), 0.0))
        rows.append(feats)

    if not rows:
        raise RuntimeError("No zones have enough history to predict (need >= 24 hours)")

    df_feat = pd.DataFrame(rows)
    feat_cols = _feature_columns()
    df_feat[feat_cols] = df_feat[feat_cols].fillna(0.0)

    booster = xgb.Booster()
    booster.load_model(str(model_path))
    dmat = xgb.DMatrix(df_feat[feat_cols], feature_names=feat_cols)
    df_feat["pred_rides"] = booster.predict(dmat)

    # Save raw prediction output (optional, keeps compatibility with your existing scripts)
    csv_path = out_dir / "pred_next_hour_advanced.csv"
    df_feat[[ZONE_COL, TIME_COL, "pred_rides"]].to_csv(csv_path, index=False, encoding="utf-8-sig")

    # Merge centroids to generate zones list for front-end
    df_cent = _load_centroids_wgs84(centroid_path)
    df_out = df_feat.merge(df_cent, left_on=ZONE_COL, right_on="LocationID", how="left")
    df_out = df_out.dropna(subset=["lat_wgs", "lon_wgs"]).copy()
    df_out["pred_rides"] = df_out["pred_rides"].astype(float).round(3)
    df_out["lat_wgs"] = df_out["lat_wgs"].astype(float)
    df_out["lon_wgs"] = df_out["lon_wgs"].astype(float)

    df_out = df_out[[ZONE_COL, "pred_rides", "Borough", "Zone", "lat_wgs", "lon_wgs"]]

    payload: Dict[str, Any] = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "last_hour": pd.Timestamp(last_hour).isoformat(),
        "next_hour": pd.Timestamp(next_hour).isoformat(),
        "zones": df_out.to_dict(orient="records"),
    }

    # Save JSON for debugging / static serving (optional)
    (out_dir / "zones.json").write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload
