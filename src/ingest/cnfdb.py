"""CNFDB / NFDB point layer (NRCan) -- an independent check on the BCWS labels.

Spec section 6.1: if BCWS and CNFDB disagree on annual counts by more than a few
percent, one of the two extractions is wrong and it must be resolved before
training. This module produces that comparison; it never feeds features.

The URL in spec section 4.2 (`.../current_version/NFDB_point.zip`) is a 404. The
directory actually publishes `NFDB_point_txt.zip` (~25 MB CSV) and
`NFDB_point_shp.zip` (~42 MB shapefile). We take the CSV: it already carries
LATITUDE/LONGITUDE, so the shapefile buys nothing and costs 17 MB.

Expect CNFDB to lag BCWS by a year or two -- it is a national compilation of
provincial submissions, so the most recent seasons are usually absent. That is a
reason to compare only the overlapping years, not a reason to distrust either.
"""
from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import pandas as pd
import requests

from src import config as cfg

log = logging.getLogger(__name__)

CNFDB_DIR = "https://cwfis.cfs.nrcan.gc.ca/downloads/nfdb/fire_pnt/current_version"
CNFDB_TXT_ZIP = f"{CNFDB_DIR}/NFDB_point_txt.zip"


def download(dest_dir: Path | None = None, *, force: bool = False) -> Path:
    """Cache the national point zip locally. Idempotent (spec gotcha 15.21)."""
    dest_dir = dest_dir or (cfg.RAW / "cnfdb")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "NFDB_point_txt.zip"
    if dest.exists() and not force:
        log.info("CNFDB already cached at %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
        return dest
    log.info("downloading CNFDB (~25 MB)...")
    with requests.get(CNFDB_TXT_ZIP, stream=True, timeout=600) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(".zip.part")
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        tmp.replace(dest)  # atomic, so an interrupted download is never mistaken for a good one
    return dest


def load_okanagan(zip_path: Path) -> pd.DataFrame:
    """Rows inside the study bbox and year range, from the national CSV."""
    with zipfile.ZipFile(zip_path) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".txt")
                    or n.lower().endswith(".csv"))
        with zf.open(name) as fh:
            df = pd.read_csv(io.TextIOWrapper(fh, encoding="latin-1"),
                             low_memory=False)
    df.columns = [c.upper() for c in df.columns]

    lat, lon, yr = "LATITUDE", "LONGITUDE", "YEAR"
    missing = [c for c in (lat, lon, yr) if c not in df.columns]
    if missing:
        raise RuntimeError(f"CNFDB schema changed; missing {missing}. Have: {list(df.columns)[:25]}")

    m = (df[lat].between(cfg.LAT_MIN, cfg.LAT_MAX)
         & df[lon].between(cfg.LON_MIN, cfg.LON_MAX)
         & df[yr].between(cfg.FIRST_YEAR, cfg.LAST_YEAR))
    return df[m].copy()


def compare_annual_counts(bcws: pd.DataFrame, cnfdb: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side annual ignition counts, restricted to overlapping years."""
    b = bcws[bcws.in_fire_season].groupby("fire_year").size().rename("bcws")
    c = (cnfdb[cnfdb["MONTH"].between(cfg.FIRE_SEASON_START[0], cfg.FIRE_SEASON_END[0])]
         if "MONTH" in cnfdb.columns else cnfdb).groupby("YEAR").size().rename("cnfdb")
    out = pd.concat([b, c], axis=1)
    out = out.loc[out.notna().all(axis=1)]  # overlapping years only
    out["diff"] = out.cnfdb - out.bcws
    out["pct_diff"] = (100 * out["diff"] / out.bcws).round(1)
    return out.reset_index(names="year")
