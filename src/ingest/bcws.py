"""BCWS Fire Incident Locations (Historical) -- the ground truth (spec section 6).

Pulled from the BC Data Catalogue WFS. Everything downstream of this is only as
good as this extract, which is why the module cleans loudly and returns a
rejection report rather than silently dropping rows.

Three things the spec did not anticipate, each found by inspecting the live
layer. See docs/data-source-findings.md.

1. FIRE_TYPE must be filtered. The layer is an *incident* log, not an ignition
   log: 'Smoke Chase' (a report with no fire), 'Nuisance Fire', 'Prescribed
   Fire', 'Training', 'Duplicate' and 'Field Activity' all live here. Only
   FIRE_TYPE == 'Fire' is a wildfire ignition. Taking every row, as spec section
   6.2 literally says, doubles the positive count and labels deliberate burns as
   wildfire ignitions.

2. The date column is IGNITION_DATE, not `discovery_date`, and it is null for
   36% of all rows -- but for 0% of FIRE_TYPE == 'Fire' rows. That null pattern
   is independent corroboration that the type filter is the right one.

3. IGNITION_DATE disagrees with FIRE_YEAR on ~45 Okanagan records, almost all of
   them falling in Feb-Mar of the *following* year, which looks like a data-entry
   default rather than a real winter ignition. They are quarantined, not kept.

The BBOX() CQL spatial filter on this endpoint was unreliable in testing (one
axis ordering returned 0 rows, the other returned the whole province), so the
bbox is applied on the LATITUDE / LONGITUDE attribute columns, which are the
reported origin coordinates and the thing spec section 6.1 actually wants.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from src import config as cfg
from src.utils.http import get_json

log = logging.getLogger(__name__)

# The only BCWS incident type that is a confirmed wildfire ignition. This is an
# invariant of the label definition, not a tuning knob -- see the post-condition
# in clean() and docs/data-source-findings.md section 2.
CONFIRMED_FIRE_TYPES = ("Fire",)

KEEP_COLS = [
    "FIRE_ID", "FIRE_NUMBER", "FIRE_YEAR", "IGNITION_DATE", "FIRE_OUT_DATE",
    "FIRE_CAUSE", "FIRE_TYPE", "LATITUDE", "LONGITUDE", "CURRENT_SIZE",
    "FIRE_CENTRE", "ZONE", "GEOGRAPHIC_DESCRIPTION",
]


@dataclass
class IgnitionExtract:
    """Cleaned positives plus an auditable account of everything dropped."""
    ignitions: pd.DataFrame
    raw_count: int
    rejections: dict[str, int] = field(default_factory=dict)

    def report(self) -> str:
        lines = [f"BCWS extract: {self.raw_count} raw rows -> {len(self.ignitions)} ignitions"]
        for reason, n in self.rejections.items():
            lines.append(f"  dropped {n:>5}  {reason}")
        return "\n".join(lines)


def fetch_raw(first_year: int = cfg.FIRST_YEAR,
              last_year: int = cfg.LAST_YEAR) -> pd.DataFrame:
    """All incident records in the bbox for [first_year, last_year], uncleaned."""
    cql = (
        f"FIRE_YEAR>={first_year} AND FIRE_YEAR<={last_year} "
        f"AND LATITUDE BETWEEN {cfg.LAT_MIN} AND {cfg.LAT_MAX} "
        f"AND LONGITUDE BETWEEN {cfg.LON_MIN} AND {cfg.LON_MAX}"
    )
    payload = get_json(cfg.BCWS_WFS_URL, {
        "service": "WFS", "version": "2.0.0", "request": "GetFeature",
        "typeName": cfg.BCWS_TYPENAME, "outputFormat": "application/json",
        "srsName": cfg.CRS_WGS84, "CQL_FILTER": cql,
    }, timeout=300)

    matched, returned = payload.get("numberMatched"), payload.get("numberReturned")
    if matched not in (None, "unknown") and returned != matched:
        # The server caps GetFeature at a default page size. If that cap is ever
        # hit we would silently lose ignitions, so refuse rather than paginate
        # blindly -- a truncated ground truth is worse than no ground truth.
        raise RuntimeError(
            f"BCWS WFS returned {returned} of {matched} matched features; "
            "the result was paginated. Add startIndex paging before trusting this."
        )
    if not payload.get("features"):
        raise RuntimeError("BCWS WFS returned no features; check the CQL filter.")

    df = pd.json_normalize([f["properties"] for f in payload["features"]])
    return df[[c for c in KEEP_COLS if c in df.columns]]


def clean(raw: pd.DataFrame) -> IgnitionExtract:
    """Apply spec section 6 label rules plus the three corrections documented above."""
    rej: dict[str, int] = {}
    df = raw.copy()
    n0 = len(df)

    # IGNITION_DATE arrives as '2023-07-02Z'. The Z is spurious -- BCWS records a
    # local calendar date, not an instant, so parsing it as UTC and converting
    # would shift a fraction of ignitions back a day (spec gotcha 15.6).
    df["ignition_date"] = pd.to_datetime(
        df["IGNITION_DATE"].astype("string").str.rstrip("Z"), errors="coerce")

    def drop(mask: pd.Series, reason: str) -> None:
        nonlocal df
        n = int(mask.sum())
        if n:
            rej[reason] = n
        df = df[~mask].copy()

    drop(~df["FIRE_TYPE"].isin(cfg.BCWS_VALID_FIRE_TYPES),
         f"FIRE_TYPE not in {cfg.BCWS_VALID_FIRE_TYPES} (smoke chases, prescribed burns, duplicates)")
    drop(df["ignition_date"].isna(), "IGNITION_DATE missing or unparseable")
    drop(df["ignition_date"].dt.year != df["FIRE_YEAR"],
         "IGNITION_DATE year disagrees with FIRE_YEAR (suspect data entry)")
    drop(df["LATITUDE"].isna() | df["LONGITUDE"].isna(), "missing coordinates")

    df["in_fire_season"] = df["ignition_date"].dt.month.between(
        cfg.FIRE_SEASON_START[0], cfg.FIRE_SEASON_END[0])

    # Post-condition, not decoration. "Confirmed fires only" is a decision, and
    # the cost of it silently regressing is a label set roughly twice the true
    # size containing smoke chases, prescribed burns and rows marked Duplicate.
    # Every metric downstream would still compute, and all of them would be wrong.
    #
    # Checked against CONFIRMED_FIRE_TYPES rather than cfg.BCWS_VALID_FIRE_TYPES
    # on purpose: asserting against the same value used to filter is tautological
    # and would wave through a widened config, which is the likeliest way this
    # regresses. Changing the label definition should require editing this
    # constant and reading this comment.
    leaked = set(df["FIRE_TYPE"].unique()) - set(CONFIRMED_FIRE_TYPES)
    if leaked:
        raise AssertionError(
            f"unconfirmed incident types survived cleaning: {sorted(leaked)}")
    if df["ignition_date"].isna().any():
        raise AssertionError("null ignition dates survived cleaning")

    out = df.rename(columns={
        "FIRE_ID": "incident_id", "FIRE_NUMBER": "fire_number",
        "FIRE_YEAR": "fire_year", "FIRE_CAUSE": "cause",
        "FIRE_TYPE": "fire_type", "LATITUDE": "lat", "LONGITUDE": "lon",
        "CURRENT_SIZE": "size_ha",
    })[[
        "incident_id", "fire_number", "fire_year", "ignition_date", "cause",
        "fire_type", "lat", "lon", "size_ha", "in_fire_season",
    ]].reset_index(drop=True)

    return IgnitionExtract(ignitions=out, raw_count=n0, rejections=rej)


def validate(extract: IgnitionExtract) -> pd.DataFrame:
    """The spec section 6.6 checks that can be run before the grid exists.

    The remaining two (every point lands in exactly one cell; panel row count)
    need grid_cells and run in src/labels/.
    """
    df = extract.ignitions
    season = df[df.in_fire_season]
    per_year = season.groupby("fire_year").size()
    by_month = season.groupby(season.ignition_date.dt.month).size()
    peak = by_month.idxmax() if len(by_month) else None

    checks = [
        ("annual counts within spec 6.4 range (150-250/season, tolerated 100-400)",
         bool(per_year.between(100, 400).all()), per_year.to_dict()),
        ("seasonal peak falls in July or August (spec 6.6)",
         peak in (7, 8), {"peak_month": peak}),
        ("total positives within spec 6.4 range (1500-2500)",
         1500 <= len(season) <= 2500, {"n": len(season)}),
        ("no duplicate (lat, lon, date) incidents",
         not season.duplicated(["lat", "lon", "ignition_date"]).any(),
         {"dupes": int(season.duplicated(["lat", "lon", "ignition_date"]).sum())}),
        ("no null sizes (needed for the 6.5.1 detection-bias sensitivity run)",
         not season.size_ha.isna().any(), {"nulls": int(season.size_ha.isna().sum())}),
    ]
    return pd.DataFrame(checks, columns=["check", "passed", "detail"])
