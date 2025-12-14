# Hotspot API (real-time)

This patch adds an HTTP backend so your front-end can **poll** for the newest next-hour hotspots.

## Files
- `api_server.py` : FastAPI app (endpoints + caching + optional static hosting)
- `hotspot_predictor.py` : XGBoost + LSTM-feature next-hour prediction logic
- `requirements.txt` : Python dependencies for the API service

## Expected project layout
Place these files in your project root (same level as `data/` and `model/`).

```
project/
  api_server.py
  hotspot_predictor.py
  data/
    test_hourly.parquet
    test_lstm_pred.csv
    taxi_zone_centroids.csv
  model/
    xgb_xgb_lstm_feat.model
```

## Run locally
```bash
pip install -r requirements.txt
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

## Endpoints
- `GET  /api/hotspots/zones` (returns all zones)
- `GET  /api/hotspots/zones?top_k=30` (returns top K zones)
- `POST /api/hotspots/refresh` (force recompute)

## Caching
By default the server recomputes at most once per 60 seconds.
Set `HOTSPOT_TTL_SECONDS=10` to recompute more frequently.

## Render start command
```bash
uvicorn api_server:app --host 0.0.0.0 --port $PORT
```

## CORS
If your front-end is on a different domain, set:
- `HOTSPOT_CORS_ORIGINS=https://your-frontend-domain.com`
(you can provide multiple, comma-separated)
