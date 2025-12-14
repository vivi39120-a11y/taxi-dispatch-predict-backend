"""
FastAPI service for real-time hotspot prediction.

Endpoints
---------
GET  /api/hotspots/zones
POST /api/hotspots/refresh
GET  /healthz

Static
------
If you put your front-end files under ./public (project root preferred),
they will be served at /.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from hotspot_predictor import compute_zones_payload


# This file lives in .../專題_XGB_LSTM_覆蓋版/hotspot_api_patch/api_server.py
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent  # ✅ 指向 .../專題_XGB_LSTM_覆蓋版


# Cache (avoid heavy prediction on every request)
_CACHE: Dict[str, Any] = {
    "ts": 0.0,
    "payload": None,
}


def _get_ttl_seconds() -> int:
    # Default: recompute at most once per 60 seconds.
    try:
        return int(os.getenv("HOTSPOT_TTL_SECONDS", "60"))
    except Exception:
        return 60


def get_or_compute_payload(force: bool = False) -> Dict[str, Any]:
    ttl = _get_ttl_seconds()
    now = time.time()

    if (not force) and _CACHE["payload"] is not None and (now - float(_CACHE["ts"])) < ttl:
        return _CACHE["payload"]

    # ✅ 用專題根目錄當 base_dir，讓 hotspot_predictor 去 PROJECT_ROOT/data、PROJECT_ROOT/model 找檔
    payload = compute_zones_payload(base_dir=PROJECT_ROOT)

    _CACHE["payload"] = payload
    _CACHE["ts"] = now
    return payload


app = FastAPI(title="Taxi Hotspot API", version="1.0")

# CORS
origins_env = os.getenv("HOTSPOT_CORS_ORIGINS", "*")
allow_origins = [o.strip() for o in origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins if allow_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"ok": "true"}


@app.get("/api/hotspots/zones")
def get_hotspot_zones(top_k: Optional[int] = None, response: Response = None) -> Dict[str, Any]:
    """
    Return latest next-hour zones payload.

    top_k (optional): if provided, return only top_k zones by pred_rides.
    """
    payload = get_or_compute_payload(force=False)

    if top_k is not None:
        try:
            k = max(1, int(top_k))
        except Exception:
            k = 20
        zones = sorted(payload.get("zones", []), key=lambda z: z.get("pred_rides", 0), reverse=True)[:k]
        payload = {**payload, "zones": zones, "top_k": k}

    # Avoid browser/proxy caching so UI sees fresh results
    if response is not None:
        response.headers["Cache-Control"] = "no-store"
    return payload


@app.post("/api/hotspots/refresh")
def refresh_hotspot_zones(response: Response = None) -> Dict[str, Any]:
    payload = get_or_compute_payload(force=True)
    if response is not None:
        response.headers["Cache-Control"] = "no-store"
    return payload


# ✅ 靜態前端：優先用「專題根目錄/public」，沒有才用「hotspot_api_patch/public」
public_dir = PROJECT_ROOT / "public"
fallback_public_dir = BASE_DIR / "public"

if public_dir.exists():
    app.mount("/", StaticFiles(directory=str(public_dir), html=True), name="public")
elif fallback_public_dir.exists():
    app.mount("/", StaticFiles(directory=str(fallback_public_dir), html=True), name="public_fallback")
