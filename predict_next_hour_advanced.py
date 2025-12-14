from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
import folium
from folium.plugins import HeatMap
from pyproj import Transformer

# ============================================================
# 下一小時需求預測（XGBoost + LSTM feature 版本）
#
# 目標：
# - 用 xgb_xgb_lstm_feat.model 取代原本 xgb_demand_poisson.model 的預測方法
# - 產出的 outputs/pred_next_hour_advanced.csv 欄位維持可被 rank / driver view 使用
# - 熱點分析（ranking / 地圖視覺化）流程不變
# ============================================================

# === 資料夾設定 ===
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# === Model / Data ===
MODEL_PATH = MODEL_DIR / "xgb_xgb_lstm_feat.model"  # 新模型（XGB + LSTM feature）
HOURLY_PATH = DATA_DIR / "test_hourly.parquet"
CENTROID_PATH = DATA_DIR / "taxi_zone_centroids.csv"

# LSTM 預測檔（若沒有會自動用 0.0 補齊）
# 建議放在 data/test_lstm_pred.csv（已跟模型包 examples 同名）
LSTM_CANDIDATES = [
    DATA_DIR / "test_lstm_pred.csv",
    DATA_DIR / "lstm_pred.csv",
    DATA_DIR / "test_lstm_pred_rides.csv",
]

# === Feature schema：需與訓練/推論程式 predict.py 一致 ===
ZONE_COL = "PULocationID"
TIME_COL = "pickup_hour"
Y_COL = "rides"
LSTM_COL = "lstm_pred_rides"

LAGS = [1, 2, 3, 24]
ROLLS = [3, 6, 24]

FEATURE_COLS = [
    "hour", "dow", "is_weekend",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    *[f"lag_{k}" for k in LAGS],
    *[f"roll_mean_{w}" for w in ROLLS],
    *[f"roll_std_{w}" for w in ROLLS],
    LSTM_COL
]


def _load_lstm_df(next_hour: pd.Timestamp) -> pd.DataFrame:
    """讀取 LSTM 預測，回傳 next_hour 的 (PULocationID, lstm_pred_rides)。不存在則回空表。"""
    lstm_path = None
    for p in LSTM_CANDIDATES:
        if p.exists():
            lstm_path = p
            break

    if lstm_path is None:
        print("⚠️ 找不到 LSTM 預測檔（會用 0.0 補齊）：", [str(p) for p in LSTM_CANDIDATES])
        return pd.DataFrame(columns=[ZONE_COL, TIME_COL, LSTM_COL])

    lstm = pd.read_csv(lstm_path)
    # 容錯：若檔案欄位名不同，嘗試做映射
    rename_map = {}
    if "pickup_hour" not in lstm.columns and "datetime" in lstm.columns:
        rename_map["datetime"] = "pickup_hour"
    if "PULocationID" not in lstm.columns and "zone_id" in lstm.columns:
        rename_map["zone_id"] = "PULocationID"
    if rename_map:
        lstm = lstm.rename(columns=rename_map)

    missing = [c for c in [ZONE_COL, TIME_COL, LSTM_COL] if c not in lstm.columns]
    if missing:
        raise ValueError(f"LSTM 預測檔缺少必要欄位 {missing}，檔案：{lstm_path}")

    lstm[TIME_COL] = pd.to_datetime(lstm[TIME_COL])
    lstm = lstm.groupby([ZONE_COL, TIME_COL], as_index=False)[LSTM_COL].mean()

    # 只取 next_hour
    lstm_next = lstm[lstm[TIME_COL] == next_hour][[ZONE_COL, LSTM_COL]].copy()
    print("✅ LSTM 預測檔：", lstm_path, "；next_hour rows：", len(lstm_next))
    return lstm_next


def _time_features(ts: pd.Timestamp) -> dict:
    hour = int(ts.hour)
    dow = int(ts.dayofweek)
    is_weekend = 1 if dow >= 5 else 0
    return {
        "hour": hour,
        "dow": dow,
        "is_weekend": is_weekend,
        "hour_sin": float(np.sin(2*np.pi*hour/24)),
        "hour_cos": float(np.cos(2*np.pi*hour/24)),
        "dow_sin": float(np.sin(2*np.pi*dow/7)),
        "dow_cos": float(np.cos(2*np.pi*dow/7)),
    }


def _lag_roll_features(y: np.ndarray) -> dict:
    # y 必須是依時間排序後，最後一個點對應 last_hour 的 rides
    feats = {}
    # lags
    for k in LAGS:
        feats[f"lag_{k}"] = float(y[-k]) if len(y) >= k else 0.0

    # rolling mean/std (與訓練版一致：shift(1).rolling(w) → 對 next hour 等價於「用最後 w 個歷史值」)
    for w in ROLLS:
        if len(y) >= w:
            window = y[-w:].astype(float)
            feats[f"roll_mean_{w}"] = float(np.mean(window))
            # pandas rolling std 預設 ddof=1
            feats[f"roll_std_{w}"] = float(np.std(window, ddof=1))
        else:
            feats[f"roll_mean_{w}"] = 0.0
            feats[f"roll_std_{w}"] = 0.0

    return feats


def main():
    print("模型：", MODEL_PATH)
    print("每小時資料：", HOURLY_PATH)
    print("centroid：", CENTROID_PATH)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)
    if not HOURLY_PATH.exists():
        raise FileNotFoundError(HOURLY_PATH)
    if not CENTROID_PATH.exists():
        raise FileNotFoundError(CENTROID_PATH)

    # 1) Load model
    model = xgb.Booster()
    model.load_model(str(MODEL_PATH))
    print("✅ XGB+LSTM 模型載入完成")

    # 2) Load hourly data (保險起見，先聚合)
    df = pd.read_parquet(HOURLY_PATH)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.groupby([ZONE_COL, TIME_COL], as_index=False)[Y_COL].sum()

    last_hour = df[TIME_COL].max()
    next_hour = last_hour + pd.Timedelta(hours=1)
    print("最後資料時間：", last_hour)
    print("預測時間（下一小時）：", next_hour)

    # 3) Load LSTM predictions for next hour
    lstm_next = _load_lstm_df(next_hour)
    lstm_map = dict(zip(lstm_next[ZONE_COL].astype(int), lstm_next[LSTM_COL].astype(float)))

    # 4) Build next-hour feature rows
    tf = _time_features(next_hour)
    rows = []

    for loc_id, g in df.groupby(ZONE_COL):
        g = g.sort_values(TIME_COL)
        y = g[Y_COL].to_numpy()

        # 訓練用的最大 lag/roll 都到 24；不足會讓特徵品質很差，這裡保持跟原本「至少 24」的邏輯一致
        if len(y) < 24:
            continue

        feats = {}
        feats.update(tf)
        feats.update(_lag_roll_features(y))

        feats[LSTM_COL] = float(lstm_map.get(int(loc_id), 0.0))

        rows.append({
            ZONE_COL: int(loc_id),
            "predict_hour": next_hour,
            **feats
        })

    df_feat = pd.DataFrame(rows)
    print("可預測的地區數量：", len(df_feat))
    if df_feat.empty:
        raise RuntimeError("沒有任何地區有足夠資料可預測（需要至少 24 筆歷史）")

    # 5) Predict
    df_feat[FEATURE_COLS] = df_feat[FEATURE_COLS].fillna(0.0)
    dtest = xgb.DMatrix(df_feat[FEATURE_COLS], feature_names=FEATURE_COLS)
    df_feat["pred_rides"] = model.predict(dtest)

    # 6) Output CSV（保持下游 rank / driver view 可讀）
    csv_path = OUT_DIR / "pred_next_hour_advanced.csv"
    out_cols = [ZONE_COL, "predict_hour", "pred_rides"]
    # 同時保留特徵欄位（方便 debug）
    out_cols = out_cols + FEATURE_COLS
    df_feat[out_cols].to_csv(csv_path, index=False, encoding="utf-8-sig")
    print("✅ 下一小時預測輸出：", csv_path)

    # 7) Heatmap (與原本一致)
    df_cent = pd.read_csv(CENTROID_PATH)

    # EPSG:2263 → WGS84(EPSG:4326)
    transformer = Transformer.from_crs("EPSG:2263", "EPSG:4326", always_xy=True)

    df_merged = df_feat.merge(
        df_cent[["LocationID", "lat", "lon"]],
        left_on=ZONE_COL,
        right_on="LocationID",
        how="left"
    )

    lons, lats = transformer.transform(df_merged["lon"].values, df_merged["lat"].values)
    df_merged["lat_wgs"] = lats
    df_merged["lon_wgs"] = lons

    center_lat = float(df_merged["lat_wgs"].mean())
    center_lon = float(df_merged["lon_wgs"].mean())

    m = folium.Map(location=[center_lat, center_lon], zoom_start=11)

    heat_data = df_merged[["lat_wgs", "lon_wgs", "pred_rides"]].dropna().values.tolist()
    HeatMap(heat_data, radius=15, blur=10).add_to(m)

    html_path = OUT_DIR / "pred_next_hour_advanced_heatmap.html"
    m.save(html_path)
    print("✅ 熱力圖輸出：", html_path)


if __name__ == "__main__":
    main()
