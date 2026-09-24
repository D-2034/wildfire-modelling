"""Per-source day-of-year climatology and anomalies (spec §4.3, §5.2, §5.3).

**Why per-source.** Spec D6 assumed that because ERA5 (training) and ECMWF IFS
(inference) share the HTESSEL land-surface model, the same depth band and the
same units, their values would be interchangeable -- "no train/serve feature skew
by construction". Measuring it showed otherwise: over 140 overlapping node-days,
IFS `sm_0_7` runs **0.068 m3/m3 (~22%) drier** than ERA5, negative at 19 of 20
nodes, with RH and temperature leaning the same way. A model trained at ~0.31 and
served at ~0.25 reads every live day as drier than it is, biasing risk upward.

Spec §5.3 already asks for `sm_0_7_anom` against "that node's day-of-year
climatology". The fix is to make that climatology **per source**: subtract the
ERA5 normal from ERA5 rows and the IFS normal from IFS rows. Any component of the
bias that is constant for a given (node, day-of-year) then cancels exactly, with
no fitted correction to version or maintain.

    anom_era5 = x_era5 - mean_era5[node, doy]
    anom_ifs  = x_ifs  - mean_ifs [node, doy]

If the bias is b, the IFS mean is shifted by b too, so it drops out of the
difference. This is why the anomaly is the robust feature and the raw level is
not -- and it is why `apply_anomaly` refuses to join a row to another source's
climatology.

**What this does NOT fix.** Only the part of the bias that is stable per
(node, doy). A state-dependent bias -- IFS drying faster after rain, say --
survives as residual skew. `validate_bias_cancellation` measures how much is left
so the claim can be checked rather than assumed.

**The reference-period caveat.** ERA5 climatology draws on 10 seasons; IFS
history begins 2024-03-05 (cfg.IFS_HISTORY_START), so its climatology draws on
about 3. Different reference periods mean the two normals describe slightly
different climates, which reintroduces a small confound.
`validate_reference_period` checks this directly by recomputing the ERA5
climatology restricted to the IFS years and comparing; if the two ERA5
climatologies agree closely, the confound is negligible and the full-record ERA5
climatology is the better choice. Run it before trusting the anomalies.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import config as cfg

log = logging.getLogger(__name__)

# Below this, a (node, doy) cell's mean is too noisy to subtract with confidence.
MIN_SAMPLES_PER_DOY = 20


def _add_doy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["doy"] = pd.to_datetime(out["date"]).dt.dayofyear
    # Feb 29 collapses onto doy 60 in leap years, shifting every later doy by one
    # relative to non-leap years. Normalising removes that off-by-one from the
    # climatology join, which would otherwise show up as a spurious anomaly.
    is_leap = pd.to_datetime(out["date"]).dt.is_leap_year
    out.loc[is_leap & (out["doy"] > 59), "doy"] -= 1
    return out


def build_climatology(daily: pd.DataFrame, *, source: str | None = None,
                      variables: list[str] | None = None,
                      halfwindow_days: int = cfg.CLIM_HALFWINDOW_DAYS,
                      ) -> pd.DataFrame:
    """Per (node_id, doy, var_name) mean/sd over a circular day-of-year window.

    Returns long format matching the `weather_clim` schema of spec §11, plus
    `source` and `n`. The window is circular so that doy 1 borrows from late
    December rather than being estimated from a half-window.
    """
    variables = variables or cfg.CLIM_VARS
    missing = [v for v in variables if v not in daily.columns]
    if missing:
        raise KeyError(f"climatology inputs missing from frame: {missing}")

    if source is None:
        srcs = daily["source"].unique()
        if len(srcs) != 1:
            raise ValueError(
                f"build_climatology got mixed sources {list(srcs)}. Build one "
                "climatology per source -- that is the whole point of this module.")
        source = srcs[0]

    df = _add_doy(daily)
    rows = []
    for doy in range(1, 366):
        offset = (df["doy"] - doy + 182) % 365 - 182  # circular distance
        window = df[offset.abs() <= halfwindow_days]
        if window.empty:
            continue
        agg = window.groupby("node_id")[variables].agg(["mean", "std", "count"])
        for var in variables:
            part = agg[var].reset_index()
            part.columns = ["node_id", "mean", "sd", "n"]
            part["doy"] = doy
            part["var_name"] = var
            rows.append(part)

    clim = pd.concat(rows, ignore_index=True)
    clim["source"] = source
    clim = clim[["node_id", "doy", "var_name", "mean", "sd", "n", "source"]]

    thin = clim[clim.n < MIN_SAMPLES_PER_DOY]
    if len(thin):
        log.warning("%s climatology: %d of %d (node, doy, var) cells have n < %d "
                    "-- anomalies there are noisy", source, len(thin), len(clim),
                    MIN_SAMPLES_PER_DOY)
    return clim


def apply_anomaly(daily: pd.DataFrame, clim: pd.DataFrame,
                  variables: list[str] | None = None) -> pd.DataFrame:
    """Attach `<var>_anom` (difference) and `<var>_anom_z` (standardised).

    Joins on (node_id, doy, var_name, **source**). The source is part of the key
    on purpose: joining an IFS row to the ERA5 climatology is precisely the bug
    this module exists to prevent, and it would not otherwise raise -- the
    numbers would simply come out biased.
    """
    variables = variables or cfg.CLIM_VARS
    df = _add_doy(daily)

    if "source" not in df.columns:
        raise KeyError("daily frame has no `source` column; cannot pick a climatology")
    unknown = set(df["source"].unique()) - set(clim["source"].unique())
    if unknown:
        raise ValueError(
            f"no climatology for source(s) {sorted(unknown)}. Build one per source; "
            "falling back to another source's normals reintroduces the ~22% "
            "ERA5/IFS soil-moisture bias this module exists to cancel.")

    for var in variables:
        c = (clim[clim.var_name == var]
             .rename(columns={"mean": "_m", "sd": "_s", "n": "_n"})
             [["node_id", "doy", "source", "_m", "_s", "_n"]])
        df = df.merge(c, on=["node_id", "doy", "source"], how="left")
        df[f"{var}_anom"] = df[var] - df["_m"]
        df[f"{var}_anom_z"] = np.where(
            df["_s"].to_numpy() > 0, (df[var] - df["_m"]) / df["_s"], np.nan)
        df[f"{var}_clim_n"] = df["_n"]
        df = df.drop(columns=["_m", "_s", "_n"])

    return df


def validate_bias_cancellation(era5_daily: pd.DataFrame, ifs_daily: pd.DataFrame,
                               era5_clim: pd.DataFrame, ifs_clim: pd.DataFrame,
                               variables: list[str] | None = None) -> pd.DataFrame:
    """Does the per-source anomaly actually remove the ERA5/IFS bias?

    Compares the two sources on the node-days where both exist, in raw levels and
    in anomalies. `residual_frac` is the share of the raw bias that survives:
    near 0 means the per-source climatology did its job, near 1 means the bias is
    state-dependent rather than a per-(node, doy) offset and a real correction
    is needed after all.
    """
    variables = variables or cfg.CLIM_VARS
    e = apply_anomaly(era5_daily, era5_clim, variables)
    i = apply_anomaly(ifs_daily, ifs_clim, variables)
    m = e.merge(i, on=["node_id", "date"], suffixes=("_e", "_i"))
    if m.empty:
        raise ValueError("no overlapping (node_id, date) between the two sources")

    rows = []
    for var in variables:
        raw = (m[f"{var}_i"] - m[f"{var}_e"]).mean()
        anom = (m[f"{var}_anom_i"] - m[f"{var}_anom_e"]).mean()
        rows.append({
            "var": var, "n": len(m),
            "raw_bias": raw, "anom_bias": anom,
            "residual_frac": (abs(anom) / abs(raw)) if abs(raw) > 1e-9 else np.nan,
        })
    return pd.DataFrame(rows)


def validate_reference_period(era5_daily: pd.DataFrame, ifs_start,
                              variables: list[str] | None = None,
                              halfwindow_days: int = cfg.CLIM_HALFWINDOW_DAYS,
                              ) -> pd.DataFrame:
    """Is the ERA5 climatology sensitive to using 10 seasons vs the IFS 3?

    Recomputes the ERA5 climatology restricted to the years IFS covers and
    compares it to the full-record one. Small differences mean the two sources'
    differing reference periods do not meaningfully confound the anomaly
    comparison, and the full ERA5 record can be used. Large ones mean both
    climatologies should be built over the same overlapping years instead.
    """
    variables = variables or cfg.CLIM_VARS
    full = build_climatology(era5_daily, variables=variables,
                             halfwindow_days=halfwindow_days)
    recent = era5_daily[pd.to_datetime(era5_daily["date"]).dt.date >= ifs_start]
    if recent.empty:
        raise ValueError(f"no ERA5 rows on or after {ifs_start}")
    restricted = build_climatology(recent, variables=variables,
                                   halfwindow_days=halfwindow_days)

    j = full.merge(restricted, on=["node_id", "doy", "var_name"],
                   suffixes=("_full", "_restricted"))
    j["delta"] = j["mean_restricted"] - j["mean_full"]
    return (j.groupby("var_name")
             .agg(mean_shift=("delta", "mean"),
                  abs_shift=("delta", lambda s: s.abs().mean()),
                  max_abs_shift=("delta", lambda s: s.abs().max()),
                  full_sd=("sd_full", "mean"))
             .assign(shift_vs_sd=lambda d: (d.abs_shift / d.full_sd).round(3))
             .reset_index())
