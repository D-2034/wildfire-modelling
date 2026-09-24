"""Static layers loaded once: fuel type, roads, settlements, DEM (spec section 4.3).

All of these are large national or provincial products and none of them needs to
be downloaded whole. Everything here reads only the study bbox:

  * fuel  -- windowed /vsicurl read of a 348 MB national GeoTIFF
  * roads -- CQL bbox filter on the BC Data Catalogue WFS
  * settlements -- same

Several spec section 4.3 URLs did not resolve as written; the working ones are
recorded here. See docs/data-source-findings.md.
"""
from __future__ import annotations

import logging
import os

import pandas as pd

from src import config as cfg
from src.utils.http import get_json

log = logging.getLogger(__name__)

# Keep GDAL from listing a whole remote directory on open, which turns a
# windowed read of a 348 MB raster into a very slow one.
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.TIF")

FUEL_TIF = ("/vsicurl/https://cwfis.cfs.nrcan.gc.ca/downloads/fuels/current/"
            "FBP_fueltypes_Canada_100m_EPSG3978_20240527.tif")

# VERIFIED present in the Okanagan window: -9999 (nodata) and codes
# 2,3,4,5,7,11,13,31,101,102,105,415,625,650,675.
#
# OPEN ITEM -- do not ship the non-fuel mask on this table alone. The codes below
# follow the CFFDRS/CWFIS convention, but the authoritative legend is
# CIFFC_standard_colours_for_FBPfueltypes.pdf in the same directory and the
# high-numbered codes (415/625/650/675) must be confirmed against it before the
# mask is trusted. Spec section 3 calls the non-fuel mask "the first thing any
# reviewer will notice", so a guess here is not good enough. Phase 2 gate.
FUEL_CODE_LABELS = {
    2: "C-2", 3: "C-3", 4: "C-4", 5: "C-5", 7: "C-7",
    11: "D-1", 13: "D-2",
    31: "O-1a",
    101: "M-1", 102: "M-2", 105: "M-1/M-2",
}
FUEL_NONFUEL_CODES_PROVISIONAL = {-9999, 415, 625, 650, 675}

ROADS_TYPENAME = "pub:WHSE_BASEMAPPING.DRA_DGTL_ROAD_ATLAS_MPAR_SP"
# The spec names "Populated places / census". WHSE_BASEMAPPING.BCGS_POPULATED_
# PLACES_500M is a 404 on this endpoint; municipality polygons are published and
# are a better basis for dist_to_settlement_m than a single place point anyway.
MUNICIPALITIES_TYPENAME = "pub:WHSE_LEGAL_ADMIN_BOUNDARIES.ABMS_MUNICIPALITIES_SP"
BCGW_OWS = "https://openmaps.gov.bc.ca/geo/pub/{typename}/ows"


def read_fuel_window():
    """Read the FBP fuel grid over the study bbox only. Returns (array, transform, crs)."""
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds, transform as window_transform

    with rasterio.open(FUEL_TIF) as src:
        bounds = transform_bounds(cfg.CRS_WGS84, src.crs, *cfg.BBOX_WSEN)
        win = from_bounds(*bounds, transform=src.transform)
        arr = src.read(1, window=win)
        return arr, window_transform(win, src.transform), src.crs


def fuel_code_summary(arr) -> pd.DataFrame:
    import numpy as np
    vals, counts = np.unique(arr, return_counts=True)
    df = pd.DataFrame({"code": vals, "pixels": counts})
    df["label"] = df.code.map(FUEL_CODE_LABELS)
    df["provisional_nonfuel"] = df.code.isin(FUEL_NONFUEL_CODES_PROVISIONAL)
    df["pct"] = (100 * df.pixels / df.pixels.sum()).round(2)
    return df.sort_values("pixels", ascending=False).reset_index(drop=True)


def geometry_column(typename: str) -> str:
    """Discover a BCGW layer's geometry column name via DescribeFeatureType.

    It is not consistent across the catalogue: the road atlas calls it GEOMETRY,
    the fire incidents layer calls it SHAPE. Hardcoding either one gives a bare
    400 on the other, with no hint as to why.
    """
    schema = get_json(BCGW_OWS.format(typename=typename.replace("pub:", "")), {
        "service": "WFS", "version": "2.0.0", "request": "DescribeFeatureType",
        "typeName": typename, "outputFormat": "application/json"}, timeout=180)
    props = schema["featureTypes"][0]["properties"]
    for p in props:
        if str(p.get("localType", "")).lower() in ("geometry", "point", "polygon",
                                                   "linestring", "multipolygon",
                                                   "multilinestring", "multipoint"):
            return p["name"]
    raise RuntimeError(f"{typename}: no geometry column found in {[p['name'] for p in props]}")


def _bcgw_features(typename: str, *, extra_cql: str = "", count: int | None = None):
    """Fetch a BCGW layer clipped to the study bbox, as a GeoDataFrame in EPSG:3005.

    The bbox goes through a BBOX() CQL filter in the layer's NATIVE CRS (3005),
    not lat/lon. This endpoint silently returned either 0 rows or the whole
    province when handed a 4326 bbox, depending on axis order -- see
    docs/data-source-findings.md.
    """
    import geopandas as gpd
    from shapely.geometry import box

    bbox_albers = (gpd.GeoSeries([box(*cfg.BBOX_WSEN)], crs=cfg.CRS_WGS84)
                   .to_crs(cfg.CRS_ALBERS).total_bounds)
    minx, miny, maxx, maxy = bbox_albers
    cql = f"BBOX({geometry_column(typename)},{minx},{miny},{maxx},{maxy})"
    if extra_cql:
        cql += f" AND {extra_cql}"

    params = {"service": "WFS", "version": "2.0.0", "request": "GetFeature",
              "typeName": typename, "outputFormat": "application/json",
              "srsName": cfg.CRS_ALBERS, "CQL_FILTER": cql}
    if count:
        params["count"] = count

    payload = get_json(BCGW_OWS.format(typename=typename.replace("pub:", "")),
                       params, timeout=600)
    feats = payload.get("features", [])
    if not feats:
        raise RuntimeError(f"{typename}: bbox query returned no features")
    gdf = gpd.GeoDataFrame.from_features(feats, crs=cfg.CRS_ALBERS)

    matched, returned = payload.get("numberMatched"), payload.get("numberReturned")
    if count is None and matched not in (None, "unknown") and returned != matched:
        log.warning("%s: got %s of %s features -- result is paginated, add startIndex",
                    typename, returned, matched)
    return gdf


def fetch_roads(count: int | None = None, *, paved_and_above_only: bool = False):
    """BC Digital Road Atlas segments in the bbox (spec section 5.7).

    `paved_and_above_only` exists for the section 6.5.1 detection-bias work: the
    distance-to-road feature means something different if resource roads count.
    """
    extra = "ROAD_CLASS <> 'trail'" if paved_and_above_only else ""
    return _bcgw_features(ROADS_TYPENAME, extra_cql=extra, count=count)


def fetch_settlements(count: int | None = None):
    """Municipality polygons in the bbox, for dist_to_settlement_m."""
    return _bcgw_features(MUNICIPALITIES_TYPENAME, count=count)
