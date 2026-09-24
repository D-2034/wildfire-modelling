"""NASA AppEEARS -- MOD13A1.061 NDVI over the Okanagan (spec sections 4.2, 5.4).

AppEEARS is asynchronous: submit a task, poll until it is done, then download.
A full 2015-present area request takes hours, so submit/poll/download are three
separate functions and the task id is cached on disk. Nothing here blocks a
worker for hours -- in Airflow this becomes a submit task, a `reschedule`-mode
sensor, and a download task (spec section 10.4).

The `available_from` column is the reason this module exists in the shape it
does. MOD13A1 is a 16-day composite published some days after its window closes,
so the freshest NDVI available on any given day describes vegetation 2-4 weeks
earlier. Joining training rows to "the composite whose window contains this date"
hands the model vegetation from the future, which inference can never have --
silent train/serve skew that costs live accuracy and raises no error (spec gotcha
15.13). `measure_processing_lag` measures the lag rather than assuming it, and
`available_from` = composite end + that lag drives an ASOF join in BOTH paths.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from src import config as cfg

log = logging.getLogger(__name__)

# Conservative default used only until measure_processing_lag() has run on real
# downloaded composites. It is deliberately NOT treated as ground truth.
ASSUMED_PROCESSING_LAG_DAYS = 10
NDVI_COMPOSITE_DAYS = 16


class AppEEARSError(RuntimeError):
    pass


def login() -> str:
    if not cfg.APPEEARS_USER or not cfg.APPEEARS_PASS:
        raise AppEEARSError("APPEARS_USER / APPEARS_PASS missing from .env")
    r = requests.post(f"{cfg.APPEEARS_BASE}/login",
                      auth=(cfg.APPEEARS_USER, cfg.APPEEARS_PASS), timeout=120)
    if r.status_code != 200:
        raise AppEEARSError(f"AppEEARS login failed: {r.status_code} {r.text[:200]}")
    return r.json()["token"]


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _bbox_geojson() -> dict:
    w, s, e, n = cfg.BBOX_WSEN
    return {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature", "properties": {},
            "geometry": {"type": "Polygon", "coordinates": [[
                [w, s], [e, s], [e, n], [w, n], [w, s]]]},
        }],
    }


def submit_area_task(token: str, start: date, end: date, *,
                     task_name: str = "wildfire_ndvi") -> str:
    """Submit an NDVI area request over the study bbox. Returns the task id."""
    payload = {
        "task_type": "area",
        "task_name": task_name,
        "params": {
            "dates": [{"startDate": start.strftime("%m-%d-%Y"),
                       "endDate": end.strftime("%m-%d-%Y")}],
            "layers": [{"product": cfg.NDVI_PRODUCT, "layer": lyr}
                       for lyr in cfg.NDVI_LAYERS],
            "output": {"format": {"type": "netcdf4"}, "projection": "geographic"},
            "geo": _bbox_geojson(),
        },
    }
    r = requests.post(f"{cfg.APPEEARS_BASE}/task", json=payload,
                      headers=_headers(token), timeout=180)
    if r.status_code not in (200, 202):
        raise AppEEARSError(f"task submit failed: {r.status_code} {r.text[:300]}")
    task_id = r.json()["task_id"]
    log.info("submitted AppEEARS task %s (%s to %s)", task_id, start, end)
    return task_id


def task_status(token: str, task_id: str) -> str:
    r = requests.get(f"{cfg.APPEEARS_BASE}/task/{task_id}",
                     headers=_headers(token), timeout=120)
    r.raise_for_status()
    return r.json().get("status", "unknown")


def list_tasks(token: str) -> pd.DataFrame:
    r = requests.get(f"{cfg.APPEEARS_BASE}/task", headers=_headers(token), timeout=120)
    r.raise_for_status()
    rows = [{"task_id": t.get("task_id"), "task_name": t.get("task_name"),
             "status": t.get("status"), "created": t.get("created")}
            for t in r.json()]
    return pd.DataFrame(rows)


def wait_for(token: str, task_id: str, *, poll_seconds: int = 60,
             timeout_seconds: int = 900) -> bool:
    """Poll until done. Returns False on timeout rather than raising.

    A False here is not an error -- the task id is cached, so the caller resumes
    later. Only in Airflow should this become a sensor.
    """
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        st = task_status(token, task_id)
        log.info("task %s: %s", task_id, st)
        if st == "done":
            return True
        if st in ("error", "failed"):
            raise AppEEARSError(f"AppEEARS task {task_id} ended as {st}")
        time.sleep(poll_seconds)
    return False


def list_bundle(token: str, task_id: str) -> pd.DataFrame:
    r = requests.get(f"{cfg.APPEEARS_BASE}/bundle/{task_id}",
                     headers=_headers(token), timeout=180)
    r.raise_for_status()
    return pd.DataFrame(r.json()["files"])


def download_bundle(token: str, task_id: str, dest_dir: Path | None = None,
                    *, only_suffixes: tuple[str, ...] = (".nc", ".csv")) -> list[Path]:
    """Download the task's output files, skipping any already on disk."""
    dest_dir = dest_dir or (cfg.RAW / "ndvi" / task_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    files = list_bundle(token, task_id)
    out: list[Path] = []
    for row in files.itertuples():
        name = Path(row.file_name).name
        if only_suffixes and not name.lower().endswith(only_suffixes):
            continue
        dest = dest_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            out.append(dest)
            continue
        url = f"{cfg.APPEEARS_BASE}/bundle/{task_id}/{row.file_id}"
        with requests.get(url, headers=_headers(token), stream=True, timeout=1800) as r:
            r.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
            tmp.replace(dest)
        out.append(dest)
        log.info("downloaded %s (%.1f MB)", name, dest.stat().st_size / 1e6)
    return out


@dataclass
class LagMeasurement:
    median_days: int
    p90_days: int
    per_granule: pd.DataFrame

    def report(self) -> str:
        return (f"MOD13A1 processing lag: median {self.median_days} d, "
                f"p90 {self.p90_days} d, n={len(self.per_granule)} granules "
                f"(range {self.per_granule.lag_days.min()}-"
                f"{self.per_granule.lag_days.max()} d)")


# MOD13A1.A2025145.h10v03.061.2025218223526
#          ^acq (yyyyddd)              ^production (yyyydddHHMMSS)
_GRANULE_RE = re.compile(
    r"MOD13A1\.A(?P<acq>\d{7})\.(?P<tile>h\d{2}v\d{2})\.(?P<coll>\d{3})\."
    r"(?P<prod>\d{13})")


def _yyyyddd_to_date(s: str) -> date:
    return date(int(s[:4]), 1, 1) + timedelta(days=int(s[4:7]) - 1)


def measure_processing_lag(granule_list_text: str) -> LagMeasurement:
    """Measure composite-end -> published lag from MODIS granule filenames.

    Spec section 5.4 insists on measuring this rather than assuming it, and the
    AppEEARS bundle listing carries no timestamp that would serve. The granule
    name does: `A<yyyyddd>` is the composite start and the 13-digit field is the
    production timestamp, so lag = production date - (start + 15 days).

    Reprocessed granules inflate the tail -- a composite originally published in
    days can be reissued months later, and the URL then shows only the reissue.
    That makes the max useless as an availability bound, so p90 is reported
    alongside the median and the median is what `available_from` should use.
    """
    rows = []
    for m in _GRANULE_RE.finditer(granule_list_text):
        start = _yyyyddd_to_date(m.group("acq"))
        end = start + timedelta(days=NDVI_COMPOSITE_DAYS - 1)
        produced = _yyyyddd_to_date(m.group("prod")[:7])
        rows.append({"granule": m.group(0), "composite_date": start,
                     "composite_end_date": end, "produced": produced,
                     "lag_days": (produced - end).days})
    if not rows:
        raise AppEEARSError("no MOD13A1 granules found in the granule list")
    df = pd.DataFrame(rows).sort_values("composite_date").reset_index(drop=True)
    return LagMeasurement(
        median_days=int(df.lag_days.median()),
        p90_days=int(df.lag_days.quantile(0.90)),
        per_granule=df,
    )


def fetch_granule_list(token: str, task_id: str) -> str:
    b = list_bundle(token, task_id)
    hits = b[b.file_name.str.contains("granule-list", case=False, na=False)]
    if hits.empty:
        raise AppEEARSError(f"task {task_id} has no granule list in its bundle")
    r = requests.get(f"{cfg.APPEEARS_BASE}/bundle/{task_id}/{hits.file_id.iloc[0]}",
                     headers=_headers(token), timeout=180)
    r.raise_for_status()
    return r.text


def composite_availability(composite_start_dates: pd.Series,
                           lag_days: int = ASSUMED_PROCESSING_LAG_DAYS) -> pd.DataFrame:
    """composite_date -> (composite_end_date, available_from) for the ASOF join.

    MOD13A1 composites start on fixed day-of-year steps of 16 days; the last one
    of a year is short. available_from = end + measured processing lag.
    """
    s = pd.to_datetime(composite_start_dates)
    end = s + pd.Timedelta(days=NDVI_COMPOSITE_DAYS - 1)
    return pd.DataFrame({
        "composite_date": s,
        "composite_end_date": end,
        "available_from": end + pd.Timedelta(days=lag_days),
    })


def cache_task_id(task_id: str, path: Path | None = None) -> Path:
    path = path or (cfg.RAW / "ndvi" / "task_id.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"task_id": task_id}))
    return path


def load_cached_task_id(path: Path | None = None) -> str | None:
    path = path or (cfg.RAW / "ndvi" / "task_id.json")
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("task_id")
