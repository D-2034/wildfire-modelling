# Data source findings

**Date:** 2026-09-24
**Scope:** Phase 1 of the spec, widened from "Open-Meteo probe" to every source in §4.
**How to reproduce:** `python scripts/run_sample_ingest.py`

Every claim below was measured against the live API, not read from documentation.
Items marked **SPEC CHANGE** contradict `wildfire-dashboard-spec.md` and need a
decision before the affected phase starts.

---

## 1. SPEC CHANGE — ERA5 and IFS soil moisture disagree by ~22%. D6 does not hold.

D6 says pairing ERA5 (training) with ECMWF IFS (inference) "eliminates
train/serve feature skew **by construction**" because both are HTESSEL, same
depth bands, same units, same variable names.

Two of those things check out, and the important one does not.

What *is* true, and is better than the spec claims: `models=era5` and
`models=ecmwf_ifs025` resolve to the **identical 0.25° lattice** and report the
**identical surface elevation** per node. There is no interpolation mismatch and
no grid mismatch at all.

What is not true: the two products disagree on the *state*. Over 140 overlapping
node-days (20 nodes × 7 days, mid-September):

| variable | ERA5 mean | IFS mean | bias (IFS−ERA5) | sd | corr |
|---|---|---|---|---|---|
| `temp_noon_c` | 14.28 | 15.24 | **+0.97 °C** | 1.15 | 0.974 |
| `rh_noon_pct` | 58.79 | 51.84 | **−6.94 pp** | 7.33 | 0.957 |
| `wind_speed_noon_kmh` | 4.70 | 4.66 | −0.04 | 2.44 | 0.422 |
| `precip_24h_mm` | 2.55 | 3.17 | +0.62 | 2.08 | 0.882 |
| `sm_0_7` | 0.314 | 0.247 | **−0.068 m³/m³** | 0.040 | 0.826 |

The soil-moisture bias is **−22% relative** and it is **negative at 19 of 20
nodes**, so it is a systematic offset, not noise. Temperature and RH lean the
same way: IFS runs hotter and drier than ERA5 here.

**Why this matters.** A model trained where `sm_0_7 ≈ 0.31` and served where
`sm_0_7 ≈ 0.25` reads every live day as drier than it is. Since the whole point
of `sm_0_7` is fine-fuel dryness, that biases live risk scores *upward* —
systematically, invisibly, and in the direction that produces false alarms.

**Caveat, stated plainly:** this is one 7-day window in late September. The
magnitude will move with season and with how wet the period was. The *sign* is
consistent enough to retire the "by construction" claim; the *size* needs a
proper multi-season measurement before anyone picks a correction.

Note this also weakens D6's argument for excluding ICON/GFS. The reason to pin
the model is still sound (D13's depth band is real and `0_to_10cm` is not a
drop-in), but "same physics ⇒ same numbers" is not the reason.

### 1a. Fix implemented: per-source day-of-year climatology

`src/features/climatology.py`. §5.3 already specifies `sm_0_7_anom` against
"that node's day-of-year climatology". Making that climatology **per source** —
ERA5 normals for ERA5 rows, IFS normals for IFS rows — cancels any component of
the bias that is constant per `(node, doy)`, with no fitted correction to version
or maintain. `apply_anomaly` joins on `(node_id, doy, var_name, source)` and
**raises** rather than falling back to another source's normals.

Measured on 2,388 overlapping node-days (4 nodes, 2024-04 → 2026-09):

| variable | raw bias | anomaly bias | cancelled |
|---|---|---|---|
| **`sm_0_7`** | **−0.0320** | **−0.0063** | **80%** |
| `rh_noon_pct` | −2.447 | +0.568 | 77% |
| `rh_min_pct` | −1.484 | +0.322 | 78% |
| `temp_max_c` | −0.466 | −0.347 | 26% |
| `temp_noon_c` | +0.079 | −0.054 | (raw bias negligible) |
| `precip_24h_mm` | −0.124 | +0.188 | none (raw bias negligible) |

**It works for the variable it was built for.** 80% of the soil-moisture bias is
gone, and RH behaves the same way.

**It does not work for `temp_max_c`** — only 26% cancels, meaning that bias is
*state-dependent* rather than a per-`(node, doy)` offset. If temperature features
turn out to matter at Phase 8, that residual needs a real correction
(quantile mapping), not this.

Two design questions settled by measurement rather than argument:

- **Reference period.** ERA5 has 10 seasons; IFS history starts **2024-03-05**
  (verified by binary search), so ~3. Different reference periods could
  reintroduce a confound. Tested directly by rebuilding the ERA5 climatology
  restricted to the IFS years: `sm_0_7` residual **19.8% (full) vs 17.2%
  (restricted)** — no meaningful gain, and the restricted version is much worse
  elsewhere (`temp_noon_c` residual 343%) because n drops to 1 per cell.
  **Keep the full ERA5 record.** An earlier `shift_vs_sd` of 0.458 that suggested
  otherwise was small-n noise.
- **Window width.** ±7 d → 20.8%, ±15 d → 19.8%, ±31 d → 16.8% residual.
  Marginal gains for a flatter seasonal cycle; **±15 d** is the default
  (`cfg.CLIM_HALFWINDOW_DAYS`).

**Remaining limitation.** The IFS climatology draws on 3 seasons and 360 of 5,856
`(node, doy, var)` cells have n < 20 (min 2, at season edges). It will improve on
its own as IFS history accumulates. Revisit before Phase 8.

Item 1 above still stands regardless: make the ERA5-vs-IFS comparison a standing
dbt test on `weather_daily.source`, not a one-off. `scripts/run_sample_ingest.py`
runs it as the `clim` step and writes `clim_bias_cancellation.parquet`.

---

## 2. SPEC CHANGE — `FIRE_TYPE` must be filtered or the labels are ~2× too many.

Spec §6.2 defines a positive as "≥ 1 BCWS incident point falls inside cell with
discovery_date == date", with no type filter.

The layer is an **incident** log, not an ignition log. Of 4,314 records in the
Okanagan bbox for 2015–2024:

| FIRE_TYPE | n | IGNITION_DATE null |
|---|---|---|
| **Fire** | **2,155** | **0.0%** |
| Nuisance Fire | 1,151 | 49.4% |
| Smoke Chase | 565 | 100% |
| Field Activity | 231 | 100% |
| Field Activity Other | 58 | 100% |
| Duplicate | 54 | 68.5% |
| Unknown | 38 | 100% |
| Training | 31 | 100% |
| Prescribed Fire | 24 | 100% |
| Fuels Management | 6 | 100% |
| Agency Assist | 1 | 100% |

Taking every row makes *smoke reports with no fire*, *deliberate prescribed
burns*, *training exercises* and *rows literally labelled Duplicate* into
wildfire ignition positives.

Three independent lines of evidence say `FIRE_TYPE == 'Fire'` is the right cut:

1. **Null pattern.** `Fire` has 0/2155 missing ignition dates. Every other type
   is 49–100% null. The spec's label is a date join; the junk types have no date.
2. **Counts match the spec's own prediction.** §6.4 expects 150–250 ignitions per
   season. `Fire` gives 124–381 (mean 211). All types gives 310–587, which does
   not.
3. **CNFDB agrees.** Cross-checking against NRCan's independently compiled
   national point layer (spec §6.1 asks for exactly this):

   | year | BCWS (`Fire`) | CNFDB | % diff |
   |---|---|---|---|
   | 2015 | 381 | 381 | 0.0 |
   | 2016 | 124 | 125 | +0.8 |
   | 2017 | 150 | 151 | +0.7 |
   | 2018 | 290 | 291 | +0.3 |
   | 2019 | 133 | 133 | 0.0 |
   | 2020 | 116 | 116 | 0.0 |
   | 2021 | 295 | 295 | 0.0 |
   | 2022 | 240 | 240 | 0.0 |
   | 2023 | 185 | 185 | 0.0 |
   | 2024 | 197 | 197 | 0.0 |

   Agreement within 0.8% every year, against a source compiled by a different
   agency. Had we kept all 4,314 records the disagreement would be ~100%.

**Also:** the column is `IGNITION_DATE`, not `discovery_date`, and `CURRENT_SIZE`,
not final size. And 38 `Fire` records have an `IGNITION_DATE` whose year differs
from `FIRE_YEAR` — almost all landing in Feb–Mar of the *following* year, which
looks like a data-entry default rather than a winter ignition. They are
quarantined in `src/ingest/bcws.py` and reported, not silently dropped.

After cleaning: **2,111 in-season positives** over 10 years — inside §6.4's
predicted 1,500–2,500. Cause split is 60% lightning / 35% person / 5% unknown,
matching §13's "~60% lightning".

All five pre-grid checks from §6.6 pass. The remaining two (every point lands in
exactly one cell; panel row count) need `grid_cells` and belong to Phase 2/4.

---

## 3. SPEC CHANGE — only 88 distinct weather nodes exist, not ~350.

D10 drops weather from a 5 km grid to a "~10 km node grid, ~350 nodes" because
"ERA5 is ~9–31 km native, so fetching 1,400 points from it is ~4x redundant".

The reasoning is right and the correction is one step short. Open-Meteo snaps
every request to the model's own grid, which is **0.25° in both axes**. Inside
the study bbox that is 8 latitudes × 11 longitudes = **88 distinct points**.
Verified directly: 36 requests on a ~5 km spacing collapsed onto **4** distinct
cells.

So a 350-node ask is still ~4× redundant; the duplicates cost budget and return
byte-identical series.

| | nodes | call-units | days at 10k/day |
|---|---|---|---|
| spec (§10.3) | 350 | 54,600 | ~5.5 |
| native grid | **88** | **13,728** | **~1.4** |

`src/grid/nodes.py` builds the nodes as the native lattice. Because the nodes
*are* grid points, `fetch_hourly` can assert exact coordinate equality on every
response — so a silent model fallback to a different lattice raises immediately
instead of degrading quietly. That check is only possible with this choice.

The 5 km `grid_cells` prediction grid is untouched; D10's other half stands.

---

## 4. `snow_depth` is all-NULL under `models=era5`.

§3 says the snow mask is free "in the same Open-Meteo call". It is not.

| model | `soil_moisture_0_to_7cm` | `snow_depth` |
|---|---|---|
| `era5` | 240/240 non-null | **0/240 non-null** (winter *and* summer) |
| `era5_land` | 240/240 | 240/240 |
| `era5_seamless` | 240/240 | 240/240 |
| *(no `models=`)* | 240/240 | 240/240 |

Handled by fetching the 8 feature variables from `era5` and `snow_depth` alone
from `era5_land`. Snow is a **mask**, never a feature (§3), so a second
land-surface model cannot reach the feature matrix. `era5_land` sits on a 0.1°
lattice, so that one call relaxes the exact-coordinate assertion to a 0.1°
tolerance — deliberately, and only there.

**Related open question for §3.** The snow mask is specified as applied
"dynamically at inference, not at build time", but training rows start April 1,
when the Okanagan highlands still hold snow. If the mask is applied live but not
in training, snowy April cells appear as easy negatives in training and are
masked at serve time — the asymmetry §3 warns about two paragraphs earlier.
Apply the same mask on both paths.

---

## 5. Same variable name, same API, 5× different values.

At one point, same date, `soil_moisture_0_to_7cm`, July 2023:

| `models=` | returned point | mean value |
|---|---|---|
| `era5` | 50.0, −119.5 | 0.1545 |
| `era5_land` | 49.9, −119.5 | 0.1142 |
| `era5_seamless` | 49.9, −119.5 | 0.1142 |
| *(omitted)* | 49.947, −119.477 | **0.0332** |

Gotcha §15.2 warns about mixing depth conventions across providers. It is
sharper than that: **the same variable name from the same endpoint differs by 5×
depending on a single query parameter**, and every value is plausible.
`src/ingest/openmeteo.py` asserts variable names, units *and* returned
coordinates on every response.

---

## 6. Corrections to specific URLs and parameters in §4.

| Spec says | Reality |
|---|---|
| CNFDB at `.../current_version/NFDB_point.zip` | **404.** Real files are `NFDB_point_txt.zip` (25 MB CSV) and `NFDB_point_shp.zip` (42 MB). CSV carries LAT/LON, so the shapefile buys nothing. |
| FIRMS `VIIRS_SNPP_NRT`, day range `/1` | Works, but any single platform sees the bbox only on its own overpasses. At a 2-day window SNPP returned 0 rows where NOAA-20 returned 2; at 3 days, 28 vs 24. Union all three VIIRS platforms. |
| FIRMS day range | The area API rejects >5 with `Invalid day range. Expects [1..5]`, despite 10 appearing in some docs. |
| "Populated places / census" (`BCGS_POPULATED_PLACES_500M`) | **404** on this endpoint. Used `WHSE_LEGAL_ADMIN_BOUNDARIES.ABMS_MUNICIPALITIES_SP` (25 municipalities in bbox) — polygons are a better basis for `dist_to_settlement_m` than place points anyway. |
| BCWS spatial filter | A `BBOX(...,'EPSG:4326')` CQL filter returned **0 rows** in one axis order and **38,498** (well outside the bbox) in the other. Filter on the `LATITUDE`/`LONGITUDE` attribute columns instead, or pass the bbox in the layer's native EPSG:3005. |
| BCGW geometry column | Not consistent across the catalogue — roads use `GEOMETRY`, fire incidents use `SHAPE`. Hardcoding either gives a bare 400 on the other. `static_layers.geometry_column()` discovers it via `DescribeFeatureType`. |

Confirmed as specified: §15.7 wind is km/h; §15.15 VIIRS `confidence` is
categorical `l`/`n`/`h`; §15.12 NDVI fill is −3000 *in raw granules* (see §8);
all three MOD13A1 layer names in §4.1; all 8 Open-Meteo variables present and
correctly united on both paths.

---

## 7. NDVI processing lag measured: 2 days, not assumed.

§5.4 insists the AppEEARS lag be measured. The bundle listing carries no usable
timestamp, but the MODIS granule name does —
`MOD13A1.A2025145.h10v03.061.2025218223526` encodes composite start (`A2025145`)
and production time (the 13-digit field).

Across the 48 granules of the 2023 area request: **median 2 days**, p90 7, range
1–9. (A smaller earlier point request showed one 58-day outlier — a granule
reprocessed and reissued months later. The max is therefore not a usable
availability bound; the median is what `available_from` should use.)

Combined with the 16-day composite window, the freshest NDVI available on any
date describes a composite starting **17–32 days earlier** — consistent with
§5.4's "2–4 weeks stale" spot-check, *provided staleness is measured from
`composite_date` (start), not from the composite end or centre*. Worth writing
that into the dbt test so it does not get checked against the wrong column.

---

## 8. SPEC CHANGE — do not re-apply the NDVI scale factor to AppEEARS NetCDF.

Gotcha §15.12 says "NDVI fill values are −3000. Screen on QA before any
arithmetic." That is true of raw HDF granules. It is **not** true of the AppEEARS
NetCDF output this project actually downloads.

The 2023 sample (`MOD13A1.061_500m_aid0001.nc`, 12 composites, 433×601 at 500 m):

- values range **−0.2 to 0.997** — already scaled
- occurrences of −3000: **0**
- NaN: 13,833 of 3,122,796 — fill has already been converted

So AppEEARS has applied the 1e-4 scale and the fill mask for us. Multiplying by
`NDVI_SCALE` again yields values around 1e-5: plausible-looking nonsense, no
error raised, and it would flow straight into `ndvi_anom` and `ndvi_delta_32d`.
`src/config.py` now marks `NDVI_SCALE` / `NDVI_FILL_VALUE` as raw-granule-only.

QA screening is still needed, just not the fill arithmetic. `pixel_reliability`
in the sample: 0 good 80.7%, 1 marginal 12.4%, 2 snow/ice 1.8%, 3 cloudy 4.6%.
Screen to reliability ∈ {0, 1} and gap-fill the rest per §5.4.

Also: `netcdf4` is missing from the §12.1 dependency list. Without it xarray
cannot open the AppEEARS output at all. Added to `environment.yml`.

---

## 9. Open items before Phase 2

- **Fuel legend is unresolved.** The Okanagan window of the national 100 m FBP
  grid contains codes `2,3,4,5,7,11,13,31,101,102,105,415,625,650,675` plus
  −9999 nodata. Codes ≤105 follow the CFFDRS convention; **415/625/650/675 are
  not identified** and together with nodata cover 17.2% of the window. The
  non-fuel mask depends on them, and §3 calls that mask "the first thing any
  reviewer will notice". `FUEL_NONFUEL_CODES_PROVISIONAL` in
  `src/ingest/static_layers.py` is a placeholder — confirm against
  `CIFFC_standard_colours_for_FBPfueltypes.pdf` before trusting it.
- **DEM source not yet settled.** The NRCan datacube elevation endpoint in §4.3
  redirects to a wrapper that rejects `GetCapabilities`. Everything else in
  Phase 2 can proceed without it.
- **Secrets in URLs.** FIRMS puts the MAP_KEY in the URL *path*, so an ordinary
  400 writes it into logs, tracebacks and notebook output. `src/utils/http.py`
  scrubs registered secrets from every log line and exception. Anything else
  that authenticates by URL needs the same treatment.
