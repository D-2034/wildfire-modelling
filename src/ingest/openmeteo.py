"""Open-Meteo ingest: ERA5 archive (training) and ECMWF IFS forecast (inference).

Both paths land in the same `weather_daily` shape so that src/features/ has a
single code path (spec section 5). Everything is fetched in UTC and converted to
local STANDARD time here -- never `timezone=America/Vancouver`, which is
DST-shifted and would slide "noon" by an hour in spring (spec gotcha 15.6).
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

from src import config as cfg
from src.utils.http import get_json

log = logging.getLogger(__name__)


class VariableContractError(RuntimeError):
    """The API returned a different variable set/units than we asked for.

    This is the failure mode spec section 4.1 calls out: a model fallback silently
    swaps the soil-moisture band for one at a different depth, the numbers stay
    plausible, and the model quietly degrades. Fail loudly instead.
    """


def _assert_contract(block: dict, expect_vars: list[str], where: str) -> None:
    units = block.get("hourly_units", {})
    got = {k for k in block.get("hourly", {}) if k != "time"}
    want = set(expect_vars)
    if got != want:
        raise VariableContractError(
            f"{where}: variable set mismatch. missing={sorted(want - got)} "
            f"unexpected={sorted(got - want)}"
        )
    for v in expect_vars:
        exp = cfg.OM_EXPECTED_UNITS.get(v)
        if exp is not None and units.get(v) != exp:
            raise VariableContractError(
                f"{where}: {v} returned units {units.get(v)!r}, expected {exp!r}"
            )


def _chunks(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _normalise(payload) -> list[dict]:
    """Open-Meteo returns a bare dict for one location and a list for many."""
    return payload if isinstance(payload, list) else [payload]


def fetch_hourly(nodes: pd.DataFrame, *, url: str, model: str,
                 variables: list[str], batch_size: int = 40,
                 coord_tolerance_deg: float = 0.0,
                 **extra) -> pd.DataFrame:
    """Fetch hourly UTC series for a set of nodes.

    Asserts that every returned coordinate matches the coordinate we asked for.
    With the default `coord_tolerance_deg=0.0` this is exact equality, which is
    the right check for era5 / ecmwf_ifs025: our nodes ARE those models' own
    0.25 deg grid points (src/grid/nodes.py), so any silent fallback to a
    different lattice trips it immediately.

    `coord_tolerance_deg` is loosened only for the era5_land snow fetch, which
    lives on a 0.1 deg lattice our nodes cannot sit on exactly. That mismatch is
    acceptable there and only there, because snow is a mask and never a feature.
    """
    frames: list[pd.DataFrame] = []
    for batch in _chunks(list(nodes.itertuples(index=False)), batch_size):
        params = {
            "latitude": ",".join(str(n.lat) for n in batch),
            "longitude": ",".join(str(n.lon) for n in batch),
            "hourly": ",".join(variables),
            "models": model,
            "timezone": "UTC",
            **extra,
        }
        payload = _normalise(get_json(url, params))
        if len(payload) != len(batch):
            raise VariableContractError(
                f"asked for {len(batch)} locations, got {len(payload)} back"
            )
        for node, block in zip(batch, payload):
            _assert_contract(block, variables, f"node {node.node_id} ({model})")
            dlat = abs(block["latitude"] - node.lat)
            dlon = abs(block["longitude"] - node.lon)
            if max(dlat, dlon) > coord_tolerance_deg:
                raise VariableContractError(
                    f"node {node.node_id}: asked ({node.lat},{node.lon}), "
                    f"model returned ({block['latitude']},{block['longitude']}); "
                    f"offset {max(dlat, dlon):.4f} deg exceeds tolerance "
                    f"{coord_tolerance_deg}. Nodes must sit on the {model} grid."
                )
            df = pd.DataFrame(block["hourly"])
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df.insert(0, "node_id", node.node_id)
            df["elevation_m_model"] = block.get("elevation")
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def hourly_to_daily(hourly: pd.DataFrame, *, source: str) -> pd.DataFrame:
    """Collapse hourly UTC to one row per (node_id, local-standard-time date).

    Conventions, stated explicitly because getting them wrong is silent:
      * "noon" = 12:00 LST = 20:00 UTC (cfg.NOON_LST_UTC_HOUR).
      * daily max/min are taken over the LST calendar day, not the UTC day.
      * precip_24h_mm is the 24 hours accumulated *to noon LST*, which is what
        Van Wagner FWI is defined on (spec section 9.4) -- NOT the LST-day total.
        Open-Meteo hourly precipitation is a preceding-hour sum, so the window
        (noon-24h, noon] is a right-closed 24-period rolling sum evaluated at
        20:00 UTC.
    """
    out_rows = []
    for node_id, g in hourly.groupby("node_id", sort=True):
        g = g.sort_values("time").set_index("time")
        lst_date = (g.index + pd.Timedelta(hours=cfg.LST_UTC_OFFSET_HOURS)).date
        g = g.assign(lst_date=lst_date)

        precip24 = g["precipitation"].rolling(24, min_periods=24).sum()

        daily = g.groupby("lst_date").agg(
            temp_max_c=("temperature_2m", "max"),
            rh_min_pct=("relative_humidity_2m", "min"),
            wind_gust_max_kmh=("wind_gusts_10m", "max"),
            n_hours=("temperature_2m", "size"),
        )

        noon = g[g.index.hour == cfg.NOON_LST_UTC_HOUR]
        noon_vals = pd.DataFrame({
            "lst_date": noon["lst_date"].values,
            "temp_noon_c": noon["temperature_2m"].values,
            "rh_noon_pct": noon["relative_humidity_2m"].values,
            "wind_speed_noon_kmh": noon["wind_speed_10m"].values,
            "wind_dir_deg": noon["wind_direction_10m"].values,
            "sm_0_7": (noon["soil_moisture_0_to_7cm"].values
                       if "soil_moisture_0_to_7cm" in noon.columns else np.nan),
            "precip_24h_mm": precip24.reindex(noon.index).values,
        }).set_index("lst_date")

        merged = daily.join(noon_vals, how="left").reset_index()
        merged.insert(0, "node_id", node_id)
        out_rows.append(merged)

    daily = pd.concat(out_rows, ignore_index=True)
    daily = daily.rename(columns={"lst_date": "date"})
    daily["date"] = pd.to_datetime(daily["date"])
    daily["source"] = source
    # An LST day needs all 24 hours present; the first and last day of any fetch
    # window are partial by construction and must not masquerade as complete.
    daily["is_complete_day"] = daily["n_hours"] == 24
    return daily


def fetch_archive_daily(nodes: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """ERA5 training weather for [start, end] plus the snow-depth mask column.

    Two calls, deliberately:
      * models=era5 for the 8 feature variables -- same 0.25 deg lattice and
        same surface elevation as ecmwf_ifs025, which is D6 holding exactly.
      * models=era5_land for snow_depth alone, because era5 returns it as all
        NULL (verified, both winter and summer). Snow is only a mask (spec
        section 3), never a feature, so sourcing it separately cannot leak a
        second land-surface model into the feature matrix.

    Padded a day on each end. The front pad keeps the rolling 24h precipitation
    window at `start` from being a partial sum; the back pad exists because an
    LST day runs 8 hours past the UTC day, so without it the final requested
    date comes back with only 16 of its 24 hours and silently under-reports
    temp_max / wind_gust_max.
    """
    pad_start = start - timedelta(days=1)
    pad_end = end + timedelta(days=1)
    feature_vars = [v for v in cfg.OM_HOURLY_VARS if v != "snow_depth"]

    hourly = fetch_hourly(
        nodes, url=cfg.OM_ARCHIVE_URL, model=cfg.OM_ARCHIVE_MODEL,
        variables=feature_vars,
        start_date=pad_start.isoformat(), end_date=pad_end.isoformat(),
    )
    daily = hourly_to_daily(hourly, source="era5")

    snow = fetch_hourly(
        nodes, url=cfg.OM_ARCHIVE_URL, model=cfg.OM_ARCHIVE_SNOW_MODEL,
        variables=["snow_depth"],
        coord_tolerance_deg=0.1,  # era5_land is a 0.1 deg lattice; see fetch_hourly
        start_date=pad_start.isoformat(), end_date=pad_end.isoformat(),
    )
    snow = snow.sort_values("time")
    snow["d"] = (snow["time"] + pd.Timedelta(hours=cfg.LST_UTC_OFFSET_HOURS)).dt.date
    snow_noon = snow[snow["time"].dt.hour == cfg.NOON_LST_UTC_HOUR][
        ["node_id", "d", "snow_depth"]
    ].rename(columns={"d": "date", "snow_depth": "snow_depth_m"})
    snow_noon["date"] = pd.to_datetime(snow_noon["date"])

    daily = daily.merge(snow_noon, on=["node_id", "date"], how="left")
    daily["snow_source"] = cfg.OM_ARCHIVE_SNOW_MODEL
    keep = (daily["date"].dt.date >= start) & (daily["date"].dt.date <= end)
    out = daily[keep].reset_index(drop=True)
    incomplete = int((~out["is_complete_day"]).sum())
    if incomplete:
        raise VariableContractError(
            f"{incomplete} of {len(out)} node-days came back with fewer than 24 "
            "hours after padding; the LST day boundary or the fetch window is wrong."
        )
    return out


def fetch_ifs_history_daily(nodes: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Past ECMWF IFS runs, for building the IFS side of the climatology.

    The live forecast API's `past_days` reaches back only ~50 days in practice,
    so it cannot produce a day-of-year climatology. This archive of past forecast
    runs can -- but only from cfg.IFS_HISTORY_START (2024-03-05), which is when
    ecmwf_ifs025 soil moisture first appears in it.

    Like the ERA5 archive, this endpoint returns snow_depth as all-NULL, so the
    snow variable is excluded. That is harmless: snow is a mask applied from the
    live forecast (which does carry it), never a climatology input.
    """
    if start < cfg.IFS_HISTORY_START:
        raise ValueError(
            f"IFS history begins {cfg.IFS_HISTORY_START}; asked for {start}. "
            "Earlier dates return silent nulls rather than an error.")
    pad_start = start - timedelta(days=1)
    pad_end = end + timedelta(days=1)
    feature_vars = [v for v in cfg.OM_HOURLY_VARS if v != "snow_depth"]

    hourly = fetch_hourly(
        nodes, url=cfg.OM_HIST_FORECAST_URL, model=cfg.OM_FORECAST_MODEL,
        variables=feature_vars,
        start_date=pad_start.isoformat(), end_date=pad_end.isoformat(),
    )
    daily = hourly_to_daily(hourly, source="ifs")
    keep = (daily["date"].dt.date >= start) & (daily["date"].dt.date <= end)
    return daily[keep & daily["is_complete_day"]].reset_index(drop=True)


def fetch_forecast_daily(nodes: pd.DataFrame, *, past_days: int = 2,
                         forecast_days: int = 7) -> pd.DataFrame:
    """IFS inference weather. snow_depth IS populated here, unlike the archive."""
    hourly = fetch_hourly(
        nodes, url=cfg.OM_FORECAST_URL, model=cfg.OM_FORECAST_MODEL,
        variables=cfg.OM_HOURLY_VARS,
        past_days=past_days, forecast_days=forecast_days,
    )
    daily = hourly_to_daily(hourly, source="ifs")
    noon = hourly[hourly["time"].dt.hour == cfg.NOON_LST_UTC_HOUR].copy()
    noon["date"] = pd.to_datetime(
        (noon["time"] + pd.Timedelta(hours=cfg.LST_UTC_OFFSET_HOURS)).dt.date)
    snow = noon[["node_id", "date", "snow_depth"]].rename(
        columns={"snow_depth": "snow_depth_m"})
    daily = daily.merge(snow, on=["node_id", "date"], how="left")
    daily["snow_source"] = cfg.OM_FORECAST_MODEL
    return daily
