"""Run every data source at sample scale and write Parquet to data/sample/.

This is the Phase 1 probe of spec section 14, widened to cover all sources
rather than just Open-Meteo. It is meant to run on a laptop in a couple of
minutes and to exercise exactly the code paths the full backfill uses -- same
functions, smaller arguments -- so that a green run here means the plumbing is
real, not that a mock passed.

    python scripts/run_sample_ingest.py
    python scripts/run_sample_ingest.py --skip ndvi,fuel

Every step is independent and idempotent; a failure in one is reported and the
rest still run, because finding out about four broken sources takes four runs
otherwise.
"""
from __future__ import annotations

import argparse
import logging
import sys
import traceback
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import config as cfg

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("sample")

OUT = cfg.SAMPLE
RESULTS: list[dict] = []


def step(name: str):
    """Decorator: run a step, record pass/fail, never abort the whole script."""
    def deco(fn):
        def wrapped(*a, **k):
            log.info("=" * 62)
            log.info("STEP %s", name)
            try:
                detail = fn(*a, **k)
                RESULTS.append({"step": name, "ok": True, "detail": detail})
                log.info("OK   %s -- %s", name, detail)
            except Exception as exc:
                RESULTS.append({"step": name, "ok": False,
                                "detail": f"{type(exc).__name__}: {exc}"})
                log.error("FAIL %s -- %s", name, exc)
                log.debug(traceback.format_exc())
        return wrapped
    return deco


def _write(df: pd.DataFrame, name: str) -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.parquet"
    df.to_parquet(path, index=False)
    return f"{len(df):,} rows -> {path.name}"


@step("weather_nodes")
def s_nodes() -> str:
    from src.grid.nodes import build_weather_nodes, to_albers, estimate_backfill_units
    nodes = to_albers(build_weather_nodes())
    days = (cfg.LAST_YEAR - cfg.FIRST_YEAR + 1) * 273
    units = estimate_backfill_units(len(nodes), days, len(cfg.OM_HOURLY_VARS))
    log.info("full backfill estimate: %,.0f call-units (~%.1f days at 10k/day)"
             .replace("%,", "%"), units, units / 10_000)
    return _write(nodes, "weather_nodes") + f"; backfill ~{units:,.0f} call-units"


@step("era5_archive")
def s_archive() -> str:
    from src.grid.nodes import build_weather_nodes
    from src.ingest.openmeteo import fetch_archive_daily
    nodes = build_weather_nodes().head(cfg.SAMPLE_NODE_LIMIT)
    df = fetch_archive_daily(nodes, cfg.SAMPLE_ARCHIVE_START, cfg.SAMPLE_ARCHIVE_END)

    expected = len(nodes) * ((cfg.SAMPLE_ARCHIVE_END - cfg.SAMPLE_ARCHIVE_START).days + 1)
    if len(df) != expected:
        raise AssertionError(f"expected {expected} node-days, got {len(df)} -- "
                             "a gap here silently corrupts every rolling feature")
    for col in ("temp_noon_c", "rh_noon_pct", "precip_24h_mm", "sm_0_7"):
        if df[col].isna().any():
            raise AssertionError(f"{col} has {df[col].isna().sum()} nulls")
    return _write(df, "weather_daily_era5")


@step("ifs_forecast")
def s_forecast() -> str:
    from src.grid.nodes import build_weather_nodes
    from src.ingest.openmeteo import fetch_forecast_daily
    nodes = build_weather_nodes().head(cfg.SAMPLE_NODE_LIMIT)
    df = fetch_forecast_daily(nodes, past_days=2, forecast_days=7)
    full = df[df.is_complete_day]
    if full.empty:
        raise AssertionError("no complete forecast days returned")
    return _write(df, "weather_daily_ifs") + f"; {len(full)} complete node-days"


@step("era5_vs_ifs_skew")
def s_skew() -> str:
    """Compare the two sources where they overlap (spec section 11 note).

    Any systematic soil-moisture offset between ERA5 and IFS on the same day at
    the same node is a train/serve skew warning, and it is much cheaper to see it
    now than after the model is trained.
    """
    from src.grid.nodes import build_weather_nodes
    from src.ingest.openmeteo import fetch_archive_daily, fetch_forecast_daily
    nodes = build_weather_nodes().head(cfg.SAMPLE_NODE_LIMIT)

    # ERA5 lands in the archive ~5 days behind real time, so the comparison
    # window has to sit behind that and the forecast side has to reach back past
    # it -- past_days=2 (the daily-inference default) never overlaps at all.
    end = date.today() - timedelta(days=7)
    start = end - timedelta(days=4)
    recent = fetch_archive_daily(nodes, start, end)
    ifs = fetch_forecast_daily(nodes, past_days=14, forecast_days=1)
    ifs = ifs[ifs.is_complete_day]
    m = recent.merge(ifs, on=["node_id", "date"], suffixes=("_era5", "_ifs"))
    if m.empty:
        return "no overlapping dates (ERA5 archive lags ~5 days) -- rerun later"
    m["sm_diff"] = m.sm_0_7_ifs - m.sm_0_7_era5
    m["temp_diff"] = m.temp_noon_c_ifs - m.temp_noon_c_era5
    _write(m, "era5_vs_ifs_overlap")
    return (f"n={len(m)} overlapping node-days; "
            f"sm_0_7 bias {m.sm_diff.mean():+.4f} m3/m3, "
            f"temp_noon bias {m.temp_diff.mean():+.2f} C")


@step("per_source_climatology")
def s_clim() -> str:
    """Build ERA5 and IFS day-of-year climatologies and measure bias cancellation.

    This is the fix for the D6 train/serve skew: each source's anomaly is taken
    against its own normals, so any bias constant per (node, doy) cancels. The
    step does not assert the fix works -- it *measures* how much survives, and
    prints it, because a partial fix that is reported is worth more than one that
    is assumed.
    """
    from src.features import climatology as cl
    from src.grid.nodes import build_weather_nodes
    from src.ingest.openmeteo import fetch_archive_daily, fetch_ifs_history_daily

    nodes = build_weather_nodes().head(4)  # 10 seasons x N nodes; keep N small
    end = date.today() - timedelta(days=8)

    def cached(name: str, fn):
        path = OUT / f"{name}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        df = fn()
        df.to_parquet(path, index=False)
        return df

    era5 = cached("clim_input_era5", lambda: pd.concat(
        [fetch_archive_daily(nodes, date(y, 4, 1), date(y, 10, 31))
         for y in range(cfg.FIRST_YEAR, cfg.LAST_YEAR + 1)], ignore_index=True))
    ifs = cached("clim_input_ifs", lambda: pd.concat(
        [fetch_ifs_history_daily(nodes, max(cfg.IFS_HISTORY_START, date(y, 4, 1)),
                                 min(date(y, 10, 31), end))
         for y in range(cfg.IFS_HISTORY_START.year, end.year + 1)], ignore_index=True))
    overlap = cached("clim_overlap_era5", lambda: fetch_archive_daily(
        nodes, max(cfg.IFS_HISTORY_START, date(cfg.IFS_HISTORY_START.year, 4, 1)), end))
    overlap = overlap[overlap.date.isin(ifs.date)]

    era5_clim = cl.build_climatology(era5)
    ifs_clim = cl.build_climatology(ifs)
    _write(pd.concat([era5_clim, ifs_clim], ignore_index=True), "weather_clim")

    bias = cl.validate_bias_cancellation(overlap, ifs, era5_clim, ifs_clim)
    print(bias.round(4).to_string(index=False))
    _write(bias, "clim_bias_cancellation")

    # The guard that actually matters: an IFS row must never be scored against
    # ERA5 normals. Confirm the join key refuses it.
    try:
        cl.apply_anomaly(ifs, era5_clim)
        raise AssertionError("cross-source anomaly join was allowed -- guard is broken")
    except ValueError:
        pass

    sm = bias[bias["var"] == "sm_0_7"].iloc[0]
    if abs(sm.anom_bias) >= abs(sm.raw_bias):
        raise AssertionError(
            f"per-source anomaly did not reduce the sm_0_7 bias "
            f"(raw {sm.raw_bias:+.4f} -> anom {sm.anom_bias:+.4f})")
    return (f"sm_0_7 bias {sm.raw_bias:+.4f} -> {sm.anom_bias:+.4f} "
            f"({1 - sm.residual_frac:.0%} cancelled); "
            f"ERA5 clim {len(era5_clim):,} cells, IFS clim {len(ifs_clim):,} "
            f"(n_min={ifs_clim.n.min()}, only {cfg.IFS_HISTORY_START.year}+ available)")


@step("bcws_labels")
def s_bcws() -> str:
    from src.ingest import bcws
    ex = bcws.clean(bcws.fetch_raw())
    print(ex.report())
    checks = bcws.validate(ex)
    print(checks.to_string(index=False))
    if not checks.passed.all():
        raise AssertionError(
            f"failed spec 6.6 checks: {checks[~checks.passed].check.tolist()}")
    _write(pd.DataFrame([{"reason": k, "n": v} for k, v in ex.rejections.items()]),
           "bcws_rejections")
    return _write(ex.ignitions, "ignitions")


@step("cnfdb_crosscheck")
def s_cnfdb() -> str:
    from src.ingest import bcws, cnfdb
    ok = cnfdb.load_okanagan(cnfdb.download())
    ex = bcws.clean(bcws.fetch_raw())
    cmp = cnfdb.compare_annual_counts(ex.ignitions, ok)
    print(cmp.to_string(index=False))
    _write(cmp, "bcws_vs_cnfdb")
    worst = cmp.pct_diff.abs().max()
    if worst > 5:
        raise AssertionError(
            f"BCWS and CNFDB disagree by up to {worst:.1f}% -- spec 6.1 says "
            "resolve this before training")
    return f"max annual disagreement {worst:.1f}% across {len(cmp)} years"


@step("firms_hotspots")
def s_firms() -> str:
    from src.ingest import firms
    df = firms.fetch_hotspots(day_range=firms.MAX_DAY_RANGE)
    if len(df) and df.confidence.isna().all():
        raise AssertionError("confidence came back all-null -- VIIRS schema change?")
    return _write(df, "active_fires") + f"; sources={df.source.nunique() if len(df) else 0}"


@step("static_roads_settlements")
def s_static_vec() -> str:
    from src.ingest import static_layers as sl
    roads = sl.fetch_roads(count=2000)
    towns = sl.fetch_settlements()
    roads.to_parquet(OUT / "roads_sample.parquet", index=False)
    towns.to_parquet(OUT / "settlements.parquet", index=False)
    return f"{len(roads):,} road segments (capped), {len(towns)} municipalities"


@step("static_fuel")
def s_fuel() -> str:
    from src.ingest import static_layers as sl
    arr, _, crs = sl.read_fuel_window()
    summary = sl.fuel_code_summary(arr)
    print(summary.to_string(index=False))
    unknown = summary[summary.label.isna() & ~summary.provisional_nonfuel]
    _write(summary, "fuel_code_summary")
    msg = f"{arr.shape} px in {crs.to_string() if hasattr(crs,'to_string') else crs}"
    if len(unknown):
        msg += (f"; {len(unknown)} UNMAPPED codes {unknown.code.tolist()} -- "
                "confirm against the CIFFC legend before trusting the non-fuel mask")
    return msg


@step("ndvi_lag")
def s_ndvi() -> str:
    from src.ingest import appeears as ap
    tok = ap.login()
    tid = ap.load_cached_task_id()
    if not tid:
        return "no cached AppEEARS task id; submit one first"
    st = ap.task_status(tok, tid)
    if st != "done":
        return f"task {tid} is '{st}' -- AppEEARS is async, rerun to pick it up"
    lag = ap.measure_processing_lag(ap.fetch_granule_list(tok, tid))
    print(lag.report())
    _write(lag.per_granule, "ndvi_processing_lag")
    files = ap.download_bundle(tok, tid)
    return lag.report() + f"; {len(files)} bundle files on disk"


STEPS = {
    "nodes": s_nodes, "archive": s_archive, "forecast": s_forecast,
    "skew": s_skew, "clim": s_clim, "bcws": s_bcws, "cnfdb": s_cnfdb,
    "firms": s_firms, "static": s_static_vec, "fuel": s_fuel, "ndvi": s_ndvi,
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--skip", default="", help="comma-separated step names")
    p.add_argument("--only", default="", help="comma-separated step names")
    args = p.parse_args()

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    for name, fn in STEPS.items():
        if name in skip or (only and name not in only):
            continue
        fn()

    print("\n" + "=" * 70)
    summary = pd.DataFrame(RESULTS)
    if summary.empty:
        print("no steps ran")
        return 1
    for r in RESULTS:
        print(f"{'PASS' if r['ok'] else 'FAIL'}  {r['step']:<26} {r['detail']}")
    n_fail = int((~summary.ok).sum())
    print(f"\n{len(summary) - n_fail}/{len(summary)} steps passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
