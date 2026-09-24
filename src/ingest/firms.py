"""NASA FIRMS active hotspots -- DISPLAY LAYER ONLY (spec D7).

Nothing here may become a model feature. Fires cluster in space and time, so
"a fire is burning 10km away" predicts "a fire starts here today" trivially and
without foresight; it is target leakage dressed up as a signal.

Two corrections to spec section 4.1:

1. The spec hardcodes VIIRS_SNPP_NRT. Any one platform sees the bbox only on its
   own overpasses, so a short window from a single source is often empty by luck
   rather than because nothing is burning -- probing on 2026-09-24, SNPP returned
   0 rows at a 2-day window where NOAA-20 returned 2, then 28 vs 24 at 3 days.
   All three VIIRS platforms are queried and unioned so the display layer does
   not flicker with overpass timing.

2. The spec's URL puts the bbox as `-120.5,49.0,-118.0,50.8`. FIRMS wants
   west,south,east,north, which that happens to be -- but it is written out from
   cfg.BBOX_WSEN here so the two can never drift apart.
"""
from __future__ import annotations

import io
import logging

import pandas as pd

from src import config as cfg
from src.utils.http import get_text

log = logging.getLogger(__name__)

# VERIFIED: for VIIRS this column is categorical ('l'/'n'/'h'), NOT the 0-100
# integer MODIS uses. Storing it as a number would coerce every value to NaN
# (spec gotcha 15.15).
VIIRS_CONFIDENCE_LEVELS = ["l", "n", "h"]

MAX_DAY_RANGE = 5


def key_status() -> dict:
    """FIRMS rate-limit budget for the configured MAP_KEY."""
    import json
    txt = get_text(
        f"https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/"
        f"?MAP_KEY={cfg.FIRMS_MAP_KEY}", secrets=(cfg.FIRMS_MAP_KEY,))
    return json.loads(txt)


def fetch_hotspots(day_range: int = 1,
                   sources: list[str] | None = None) -> pd.DataFrame:
    """Active hotspots over the study bbox for the last `day_range` days."""
    if not cfg.FIRMS_MAP_KEY:
        raise RuntimeError("FIRMS_MAP_KEY missing from .env")
    # VERIFIED: the area API rejects anything above 5 with a 400 and
    # "Invalid day range. Expects [1..5]." -- despite 10 appearing in some docs.
    if not 1 <= day_range <= MAX_DAY_RANGE:
        raise ValueError(
            f"FIRMS area API accepts a day range of 1-{MAX_DAY_RANGE}, got {day_range}")

    west, south, east, north = cfg.BBOX_WSEN
    bbox = f"{west},{south},{east},{north}"
    frames = []
    for src in (sources or cfg.FIRMS_SOURCES):
        url = f"{cfg.FIRMS_BASE}/{cfg.FIRMS_MAP_KEY}/{src}/{bbox}/{day_range}"
        body = get_text(url, secrets=(cfg.FIRMS_MAP_KEY,))
        header = body.split("\n", 1)[0]
        if "latitude" not in header:
            # FIRMS signals errors with a 200 and a plain-text body.
            log.warning("FIRMS %s returned no usable CSV: %s", src, body[:160])
            continue
        df = pd.read_csv(io.StringIO(body), dtype={"confidence": "string"})
        df["source"] = src
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=[
            "fire_id", "lat", "lon", "bright_ti4", "confidence", "frp",
            "acq_datetime", "source"])

    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"latitude": "lat", "longitude": "lon"})

    # acq_time is an HHMM integer that loses its leading zero: 45 means 00:45.
    df["acq_datetime"] = pd.to_datetime(
        df["acq_date"].astype(str) + " "
        + df["acq_time"].astype(int).astype(str).str.zfill(4),
        format="%Y-%m-%d %H%M", utc=True)

    bad = set(df["confidence"].dropna().unique()) - set(VIIRS_CONFIDENCE_LEVELS)
    if bad:
        log.warning("unexpected VIIRS confidence values %s -- schema may have changed", bad)

    # The same fire is seen by several platforms on several overpasses; a stable
    # id keeps repeated daily loads idempotent (spec gotcha 15.21).
    df["fire_id"] = (df["source"] + "_" + df["lat"].round(5).astype(str) + "_"
                     + df["lon"].round(5).astype(str) + "_"
                     + df["acq_datetime"].dt.strftime("%Y%m%d%H%M"))

    cols = ["fire_id", "lat", "lon", "bright_ti4", "bright_ti5", "confidence",
            "frp", "daynight", "acq_datetime", "source"]
    return df[[c for c in cols if c in df.columns]].drop_duplicates("fire_id")
