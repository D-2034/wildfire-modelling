"""Project-wide constants. Everything that a reviewer might want to change lives here.

Values marked VERIFIED were confirmed against the live API on 2026-09-24; see
docs/data-source-findings.md for the probe results behind each one.
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# --- paths -----------------------------------------------------------------
DATA = ROOT / "data"
RAW = DATA / "raw"
STATIC = DATA / "static"
PROCESSED = DATA / "processed"
SAMPLE = DATA / "sample"
DUCKDB_PATH = ROOT / "wildfire.duckdb"

# --- study area (spec section 3) -------------------------------------------
LAT_MIN, LAT_MAX = 49.0, 50.8
LON_MIN, LON_MAX = -120.5, -118.0
BBOX_WSEN = (LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)  # west, south, east, north
CRS_WGS84 = "EPSG:4326"
CRS_ALBERS = "EPSG:3005"  # BC Albers - ALL metric work happens here (spec section 3)

# --- time ------------------------------------------------------------------
FIRST_YEAR, LAST_YEAR = 2015, 2024
FIRE_SEASON_START = (4, 1)   # April 1
FIRE_SEASON_END = (10, 31)   # October 31
# Rolling features need 90 days of warm-up, so weather is backfilled from Feb 1.
BACKFILL_SEASON_START = (2, 1)

# Okanagan local STANDARD time is UTC-8 all year. FWI and the "noon" convention
# are defined on LST, NOT on DST-shifted local time (spec gotcha 15.6), so we
# fetch in UTC and convert with this fixed offset - never timezone=America/Vancouver.
LST_UTC_OFFSET_HOURS = -8
NOON_LST_UTC_HOUR = 12 - LST_UTC_OFFSET_HOURS  # 20:00 UTC

# --- Open-Meteo (spec sections 4.1, 4.2) ------------------------------------
OM_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OM_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
# Archive of past IFS *forecast* runs. This is the only source of multi-season
# IFS history: the live forecast API's past_days reaches back ~50 days in
# practice, which is far too short to build a day-of-year climatology from.
OM_HIST_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

# The 8 variables of spec section 4.1. Order is load-bearing: we assert the
# response echoes exactly this set (spec gotcha 15.1, 15.2).
OM_HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_gusts_10m",
    "wind_direction_10m",
    "precipitation",
    "snow_depth",
    "soil_moisture_0_to_7cm",
]
# Units we require back, verbatim. A silent switch to m/s would corrupt ISI with
# no error (spec gotcha 15.7); a soil-moisture band swap is worse (15.2).
OM_EXPECTED_UNITS = {
    "temperature_2m": "\u00b0C",
    "relative_humidity_2m": "%",
    "wind_speed_10m": "km/h",
    "wind_gusts_10m": "km/h",
    "wind_direction_10m": "\u00b0",
    "precipitation": "mm",
    "snow_depth": "m",
    "soil_moisture_0_to_7cm": "m\u00b3/m\u00b3",
}

# VERIFIED: models=era5 and models=ecmwf_ifs025 snap to the *identical* 0.25 deg
# lattice and report identical surface elevation per node. That is D6 holding by
# construction rather than by assumption.
OM_ARCHIVE_MODEL = "era5"
OM_FORECAST_MODEL = "ecmwf_ifs025"

# VERIFIED: models=era5 returns snow_depth as ALL NULL (both winter and summer).
# Only era5_land / era5_seamless carry it. Snow is a mask, not a feature, so it
# is fetched separately rather than contaminating the feature variables with a
# second land-surface model. See docs/data-source-findings.md finding 2.
OM_ARCHIVE_SNOW_MODEL = "era5_land"
SNOW_DEPTH_MASK_M = 0.02  # spec section 3

# ERA5 / IFS native grid. Weather nodes ARE these points (see src/grid/nodes.py).
ERA5_GRID_STEP_DEG = 0.25

# VERIFIED by binary search: ecmwf_ifs025 soil_moisture_0_to_7cm first appears in
# the historical-forecast archive on this date. It bounds how many seasons the
# IFS climatology can draw on (currently 3), which is the main limitation of the
# per-source anomaly scheme in src/features/climatology.py.
IFS_HISTORY_START = date(2024, 3, 5)

# Day-of-year climatology window (+/- days). Widening it trades resolution of the
# seasonal cycle for sample size, which matters much more for IFS (3 seasons)
# than for ERA5 (10).
CLIM_HALFWINDOW_DAYS = 15

# Variables that get a per-source day-of-year climatology and anomaly.
CLIM_VARS = ["sm_0_7", "temp_max_c", "temp_noon_c", "rh_min_pct",
             "rh_noon_pct", "precip_24h_mm"]

# --- NASA FIRMS (spec section 4.1, display layer only per D7) ---------------
FIRMS_MAP_KEY = (os.getenv("FIRMS_MAP_KEY") or "").strip()
FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
# VERIFIED: VIIRS_SNPP_NRT (the source named in the spec) returned 0 rows while
# VIIRS_NOAA20_NRT returned detections for the same bbox/window. SNPP NRT is
# winding down. Query all three and union.
FIRMS_SOURCES = ["VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT", "VIIRS_SNPP_NRT"]

# --- BCWS ground truth (spec section 6) -------------------------------------
BCWS_WFS_URL = (
    "https://openmaps.gov.bc.ca/geo/pub/"
    "WHSE_LAND_AND_NATURAL_RESOURCE.PROT_HISTORICAL_INCIDENTS_SP/ows"
)
BCWS_TYPENAME = "pub:WHSE_LAND_AND_NATURAL_RESOURCE.PROT_HISTORICAL_INCIDENTS_SP"

# VERIFIED and NOT in the spec: the layer holds 11 FIRE_TYPE values, only one of
# which is an actual wildfire ignition. Of 4,314 Okanagan records 2015-2024, only
# 2,155 are 'Fire'; the rest are Smoke Chase (false alarms), Nuisance Fire,
# Prescribed Fire (deliberate), Training, Duplicate, ... Taking all of them as
# positives, as spec section 6.2 literally says, roughly doubles the positive
# count and labels prescribed burns as wildfire ignitions.
# Corroborating evidence that 'Fire' is the right filter:
#   * 'Fire' has 0/2155 null IGNITION_DATE; every other type is 49-100% null.
#   * 'Fire' yields 124-381 ignitions/season, matching spec section 6.4's
#     predicted 150-250. All types together gives 310-587, which does not.
BCWS_VALID_FIRE_TYPES = ["Fire"]

# --- NASA AppEEARS (NDVI, spec sections 4.2, 5.4) ---------------------------
APPEEARS_BASE = "https://appeears.earthdatacloud.nasa.gov/api"
# Names must match .env.example exactly. Env vars are case-sensitive on Linux
# (Airflow in Docker), so the project's original `AppEARS_user` resolved on
# Windows and would have returned None in a container.
APPEEARS_USER = (os.getenv("APPEEARS_USER") or "").strip()
APPEEARS_PASS = (os.getenv("APPEEARS_PASS") or "").strip()
NDVI_PRODUCT = "MOD13A1.061"
NDVI_LAYERS = ["_500m_16_days_NDVI", "_500m_16_days_pixel_reliability", "_500m_16_days_VI_Quality"]
# VERIFIED -- and this contradicts spec gotcha 15.12. That warning ("NDVI fill
# values are -3000, screen on QA before any arithmetic") is true of the raw HDF
# granules, but AppEEARS NetCDF output has ALREADY applied the scale factor and
# already converted fill to NaN: the downloaded sample ranges -0.2 to 0.997 with
# zero occurrences of -3000 and 13,833 NaNs. Applying NDVI_SCALE a second time
# would divide by 10,000 again and produce values around 1e-5 -- plausible-looking
# nonsense with no error. So these two constants apply ONLY if someone switches
# the output format to raw HDF/GeoTIFF granules.
NDVI_FILL_VALUE = -3000        # raw-granule only; NOT present in AppEEARS NetCDF
NDVI_SCALE = 1e-4              # raw-granule only; AppEEARS NetCDF is pre-scaled
NDVI_APPEEARS_NETCDF_PRESCALED = True

# QA screening is still required (spec 5.4). pixel_reliability in the sample:
# 0 good 80.7%, 1 marginal 12.4%, 2 snow/ice 1.8%, 3 cloudy 4.6%.
NDVI_RELIABILITY_OK = (0, 1)

# --- CNFDB cross-check (spec section 6.1) -----------------------------------
CNFDB_POINT_URL = "https://cwfis.cfs.nrcan.gc.ca/downloads/nfdb/fire_pnt/current_version/NFDB_point.zip"

# --- sample mode ------------------------------------------------------------
# A cut small enough to run end to end on a laptop in a couple of minutes while
# exercising every code path the full backfill uses.
SAMPLE_YEARS = [2023]
SAMPLE_NODE_LIMIT = 6
SAMPLE_ARCHIVE_START = date(2023, 2, 1)
SAMPLE_ARCHIVE_END = date(2023, 10, 31)
