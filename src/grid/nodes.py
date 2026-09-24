"""Weather nodes = the ERA5 / IFS native 0.25 degree lattice over the bbox.

Spec D10 puts weather on a "~10 km node grid, ~350 nodes" to avoid the ~4x
redundancy of fetching at 5 km. Probing the API shows that correction does not
go far enough: Open-Meteo snaps every request to the model's own 0.25 deg grid,
and there are only 88 such points inside the study bbox. A 10 km ask is still
~4x redundant -- 350 requests resolving to 88 distinct cells, each duplicate
costing budget and returning a byte-identical series.

So the nodes ARE the native grid points. Consequences:
  * backfill cost drops ~4x (see estimate_backfill_units)
  * zero interpolation between what we ask for and what we get
  * archive(era5) and forecast(ecmwf_ifs025) resolve to the same lattice AND
    report the same surface elevation, so D6's "no train/serve skew by
    construction" is literally true rather than approximately true.

`grid_cells` stays at 5 km (D10's other half is untouched): terrain, fuel, roads
and NDVI really do vary at that scale, and each cell still joins to its nearest
node.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as cfg


def build_weather_nodes() -> pd.DataFrame:
    """The 0.25 deg lattice points inside the study bbox.

    Open-Meteo's grid is aligned on exact multiples of 0.25 in both axes, which
    is why this uses multiples-of-step arithmetic rather than arange from the
    bbox corner.
    """
    step = cfg.ERA5_GRID_STEP_DEG
    lats = np.arange(np.ceil(cfg.LAT_MIN / step), np.floor(cfg.LAT_MAX / step) + 1) * step
    lons = np.arange(np.ceil(cfg.LON_MIN / step), np.floor(cfg.LON_MAX / step) + 1) * step
    rows = [
        {"node_id": i, "lat": round(float(la), 4), "lon": round(float(lo), 4)}
        for i, (la, lo) in enumerate(
            (la, lo) for la in lats for lo in lons
        )
    ]
    return pd.DataFrame(rows)


def to_albers(nodes: pd.DataFrame) -> pd.DataFrame:
    """Add x_albers / y_albers (spec section 11 schema). Metric work is EPSG:3005."""
    import geopandas as gpd

    gdf = gpd.GeoDataFrame(
        nodes.copy(),
        geometry=gpd.points_from_xy(nodes.lon, nodes.lat),
        crs=cfg.CRS_WGS84,
    ).to_crs(cfg.CRS_ALBERS)
    out = nodes.copy()
    out["x_albers"] = gdf.geometry.x.values
    out["y_albers"] = gdf.geometry.y.values
    return out


def estimate_backfill_units(n_nodes: int, n_days: int, n_vars: int) -> float:
    """Open-Meteo's own weighting: (variables/10) * (days/14) * locations."""
    return (n_vars / 10) * (n_days / 14) * n_nodes
