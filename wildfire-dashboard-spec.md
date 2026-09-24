# Okanagan Wildfire Ignition Risk — Project Spec

**Status:** planning / pre-implementation
**Last revised:** 2026-09-19
**Audience:** implementers (human or AI agent) picking this project up cold.

---

## 1. Goal

Predict the probability that a wildfire **ignites** in a given ~5km pixel of the
Okanagan (BC) on a given day, using a Random Forest trained on historical
ignitions. Surface the result as a **0–100 risk score** on an interactive
choropleth dashboard, refreshed daily.

Accuracy is measured against **observed ignitions** (§6, §9). The Canadian Fire
Weather Index (FWI) is the incumbent operational system and is scored on the same
ground truth as a comparison baseline.

**Explicit non-goals:**
- Not predicting fire *spread*, final size, or severity. Ignition only.
- Not distinguishing lightning-caused from human-caused ignition. The label is
  cause-agnostic; both count as a positive.
- Not generalizing outside the Okanagan. The model is deliberately region-specific.
- Not extrapolating beyond historically observed conditions. Accepted limitation
  (§13).
- Not an operational or safety-critical product. See §16.

---

## 2. Decisions log

Read this before changing anything — these are settled choices with reasons, and
several earlier drafts of this spec had them wrong.

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | **Scope reduced from all of BC to the Okanagan.** | Makes the Open-Meteo API budget feasible, makes the NDVI baseline job tractable, and allows finer resolution. Cost: fewer positive labels, no transferability. |
| D2 | **Random Forest classifier replaces the hand-weighted composite score.** | The old 30/20/15/15/10/10 weights were guesses over collinear components. RF learns them and can be evaluated honestly. |
| D3 | **FWI is NOT a model feature.** Only raw observational inputs and aggregations of them. | User requirement. Keeps the baseline comparison clean — the model cannot win by being handed the baseline. |
| D4 | **FWI is still computed**, on the evaluation path only. | It is the comparison baseline (§9). The Van Wagner implementation is required; it just does not feed the feature matrix. Do not delete it. |
| D5 | **Rolling-window aggregations of raw weather ARE allowed as features.** | Drought memory is the single largest signal in fire risk. FWI's power comes from its recursion (DC has a ~50-day time constant), not from its functional form. A same-day-only feature vector has no memory and will badly underperform. Rolling sums/means of raw inputs restore that memory without importing a pre-computed fire index. This is the intended reading of D3. |
| D6 | **Soil moisture source: ERA5 / ECMWF HTESSEL, one source only** (surface layer only — see D13). | Training uses ERA5 via Open-Meteo's archive API; inference uses ECMWF IFS via the forecast API. IFS and ERA5 share the same land-surface model (HTESSEL), the same depth bands, the same units (m³/m³) and the same Open-Meteo variable names. This eliminates train/serve feature skew by construction. Do NOT mix in ICON (`0_to_1cm`/`1_to_3cm`/…) or GFS (`0_to_10cm`/…) bands — different depths, different model physics, silent degradation. |
| D7 | **Active-fire proximity (FIRMS) is a display layer, not a feature.** | Target leakage. Fires cluster spatially and temporally, so "a fire is burning 10km away" trivially predicts "a fire starts here today" without being useful foresight. Still ingested and still shown on the map. |
| D8 | **Historical fire count is a feature, but computed strictly out-of-sample.** | For a row dated in year Y, count only ignitions from years < Y. Using the full record leaks the labels. |
| D9 | **Daily time step, not hourly.** | Ignition labels are dated to the day, FWI is defined at noon local standard time on a daily cycle, and the drought signal moves on a scale of weeks. Hourly refresh added cost and no information. |
| D10 | **Weather is fetched on a coarse ~10km node grid; risk is scored on a ~5km cell grid.** | ERA5 is ~9–31km native, so fetching 1,400 points from it is ~4x redundant and blows the API budget. Terrain, fuel, roads and NDVI genuinely do vary at 5km in Okanagan valleys, so the risk grid stays fine. Each risk cell joins to its nearest weather node. |
| D11 | **Risk score = percentile rank of calibrated probability, pooled across the region and the reference period.** | Raw RF probabilities sit near the base rate (~0.1%), so `p × 100` would render every cell dark green forever. Percentile mapping is interpretable and uses the full 0–100 range. Pooled (not per-cell) so that a regionally bad day reads as bad everywhere — which is correct. |
| D12 | **Accuracy is scored against observed ignitions, not against FWI.** | Comparing predictions to FWI measures *agreement with FWI*, which is minimized by a model that merely reproduces FWI — the RF could tie but never beat it. Both models are instead scored against the same binary ground truth (§6). FWI-agreement survives only as a diagnostic (§9.3). |
| D13 | **Only the `soil_moisture_0_to_7cm` layer is used. No deeper layers.** | User requirement. The surface layer is the best single proxy for fine-fuel moisture, which governs whether an ignition source takes hold. Cost: it carries no multi-week drought memory (a single rain resets it), so the §5.2 rolling-precipitation block becomes the sole carrier of that signal. See §5.3 for the full trade-off and the conditions under which to revisit. |
| D14 | **Airflow orchestrates both the backfill and the daily inference path.** | The daily job alone would not justify it — it is four linear tasks. The *backfill* does: ~90 chunks of `(year, node_batch)` needing retries, rate-limit backoff, resumability and gap detection, which is precisely Airflow's dynamic task mapping. Running both under one orchestrator avoids maintaining two execution models. Accepted cost: a scheduler, metadata DB and webserver to operate (§15.17–15.22). |

---

## 3. Study area and grid

**Bounding box:** lat 49.0 to 50.8, lon -120.5 to -118.0
(Osoyoos at the south end through Vernon/Armstrong to the Salmon Arm approaches.)
Area ≈ 35,600 km².

**Projection:** all grid construction, distance and area work happens in
**BC Albers, EPSG:3005** — never in raw lat/lon degrees. A degree of longitude is
~30% shorter at 50.8°N than at 49.0°N, so a naive degree grid does not produce
square cells.

**Two grids:**

| Grid | Resolution | Count | Purpose |
|------|-----------|-------|---------|
| `weather_nodes` | ~10 km | ~350 | Every Open-Meteo fetch (historical and live). Sized to ERA5's native resolution. |
| `grid_cells` | ~5 km | ~1,400 raw, ~1,150 after masking | The unit of prediction and display. Carries terrain, fuel, roads, NDVI. |

Each `grid_cells` row stores a `weather_node_id` foreign key (nearest node,
computed once in EPSG:3005).

**Masking** — applied once, at grid build time, before anything else:
- **Non-fuel mask.** Drop or permanently flag cells that are water, rock, ice,
  glacier or dense urban. Okanagan Lake alone is ~350 km². Without this the map
  renders "Extreme" over open water on hot days, which is the first thing any
  reviewer will notice.
- **Snow mask.** Applied dynamically at inference, not at build time. If
  `snow_depth > 0.02 m` for a cell-day, the cell is masked (rendered grey,
  excluded from scoring). Ignition under snowpack is not a meaningful quantity, and
  the Okanagan highlands (Big White, Silver Star, Apex) carry snow roughly
  November–May. `snow_depth` is free in the same Open-Meteo call.
- **Fire season restriction.** Training rows and live scoring are limited to
  **April 1 – October 31**. Outside that window the dashboard shows an
  off-season state rather than a score.

Masked cells are excluded from the training panel entirely (§6.3) — they must not
appear as easy negatives, or every metric will be flattered.

---

## 4. Data sources

### 4.1 Live (inference path)

| Source | Variables | Endpoint | Auth | Cadence |
|--------|-----------|----------|------|---------|
| **Open-Meteo Forecast (ECMWF IFS)** | `temperature_2m`, `relative_humidity_2m`, `wind_speed_10m`, `wind_gusts_10m`, `wind_direction_10m`, `precipitation`, `snow_depth`, `soil_moisture_0_to_7cm` | `https://api.open-meteo.com/v1/forecast` with `&models=ecmwf_ifs025` | None | Daily |
| **NASA FIRMS** | Active hotspots (display layer only, per D7) | `https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/VIIRS_SNPP_NRT/-120.5,49.0,-118.0,50.8/1` | Free MAP_KEY | 2–4x/day |
| **MODIS NDVI** | `MOD13A1.061` NDVI, 500m 16-day composite | AppEEARS **area** request | Free Earthdata login | Weekly |

> **`models=ecmwf_ifs025` is mandatory, not optional.** Without it Open-Meteo
> picks a "best match" model per location, which in BC may be GEM/HRDPS or GFS —
> neither of which exposes the HTESSEL soil bands. You will get either an API
> error or silent nulls. After every fetch, assert that the returned variable list
> matches the requested list exactly and fail loudly if it does not. An earlier
> version of this project lost time to exactly this class of bug with FWI
> variables that Open-Meteo never provided.

### 4.2 Historical (training path)

| Source | Variables | Endpoint | Period |
|--------|-----------|----------|--------|
| **Open-Meteo Archive (ERA5)** | Same variable list as §4.1 | `https://archive-api.open-meteo.com/v1/archive` | 2015-02-01 → present |
| **BCWS Fire Incident Locations (Historical)** | **Ground truth.** Ignition points: lat, lon, discovery date, cause, final size. See §6. | BC Data Catalogue, `catalogue.data.gov.bc.ca` | 2015 → present |
| **CNFDB point layer (NRCan)** | Cross-check on the BCWS labels | `cwfis.cfs.nrcan.gc.ca` | 2015 → present |
| **AppEEARS NDVI history** | MOD13A1 over the Okanagan bbox | AppEEARS area request | 2015 → present |

### 4.3 Static (load once)

| Source | Derived fields | Where |
|--------|---------------|-------|
| **DEM (NRCan CDEM / BC Open Data)** | elevation, slope, aspect, TPI | GeoGratis / BC Open Data Portal |
| **NRCan CFFDRS Fuel Type Grid** | fuel class (C1–C7, M1–M4, D1, S1–S3, O1), non-fuel flag | cwfis.cfs.nrcan.gc.ca |
| **BC Digital Road Atlas** | distance to nearest road, road density | BC Open Data Portal |
| **Populated places / census** | distance to settlement, population density | StatCan / BC Open Data |
| **NDVI climatology** | per-cell, per-composite-period mean and sd | derived from AppEEARS history |
| **Weather climatology** | per-node, per-day-of-year mean and sd | derived from the ERA5 backfill |

---

## 5. Feature set

All features are computed by **one shared code path** used by both training and
inference (`src/features/`). This is non-negotiable: divergence between training
features and serving features is the most common and most silent failure mode in
this kind of project.

Unit of observation: **one cell-day**, valid at noon local standard time.

### 5.1 Same-day weather (from the cell's weather node)

`temp_noon_c`, `temp_max_c`, `rh_noon_pct`, `rh_min_pct`,
`wind_speed_noon_kmh`, `wind_gust_max_kmh`, `wind_dir_sin`, `wind_dir_cos`,
`precip_24h_mm`

### 5.2 Rolling weather — the memory block (per D5)

This block is what replaces FWI's recursion. Do not drop it.

- `precip_sum_3d`, `_7d`, `_14d`, `_30d`, `_60d`, `_90d`
- `days_since_precip_gt_1mm`, `days_since_precip_gt_5mm`
- `dry_spell_length` — consecutive days with < 1 mm
- `temp_max_mean_3d`, `_7d`, `_30d`
- `rh_min_mean_3d`, `_7d`, `_30d`
- `wind_max_3d`, `_7d`
- `temp_max_anom_30d` — 30-day mean vs. that node's day-of-year climatology
- `precip_anom_30d` — 30-day sum vs. that node's day-of-year climatology

> Rolling windows require up to 90 days of prior data. The historical backfill must
> therefore start on **February 1** of each year even though training rows begin
> April 1, and the live pipeline must retain a ≥90-day rolling weather history.

### 5.3 Soil moisture (D6 — ERA5/IFS HTESSEL, surface layer only)

**`soil_moisture_0_to_7cm` is the only depth used.** All deeper HTESSEL layers
(7–28cm, 28–100cm, 100–255cm) are excluded by decision D13.

`sm_0_7` (volumetric, m³/m³),
`sm_0_7_anom` (vs. that node's day-of-year climatology),
`sm_0_7_delta_30d` (current minus 30 days ago — the drying *trend*)

> **What this costs, so a future implementer does not rediscover it as a bug.**
> The 0–7cm layer responds within hours to rain and to evaporative demand, which
> makes it a good proxy for fine-fuel moisture — the thing that governs whether an
> ignition source actually takes. But it is also the layer that recovers fastest,
> so it carries almost no *drought* memory: a single wetting rain resets it even in
> the middle of a severe season. That deep-drying signal is what FWI's Drought Code
> captures via the 28–100cm range.
>
> Two consequences: (a) `sm_0_7` is substantially collinear with
> `precip_sum_3d`/`_7d` (§5.2), so expect its feature importance to be modest and
> split with those; (b) the §5.2 memory block is now the *only* source of
> multi-week dryness information in the feature set, which raises the stakes on
> keeping `precip_sum_30d/60d/90d` and `dry_spell_length`. Do not trim those.
>
> If evaluation (§9) shows the model underperforming FWI specifically in
> long-drought conditions — late-season, low-DC-agreement cells — reinstating
> `sm_28_100` is the first thing to try.

### 5.4 Vegetation

`ndvi`, `ndvi_anom` (vs. per-cell per-composite-period climatology),
`ndvi_delta_32d` (change over two composite periods)

> MOD13A1 has cloud gaps, a −3000 fill value and QA bitflags. Screen on the QA
> layer and the pixel reliability band, then gap-fill by temporal interpolation
> per cell. Never let raw fill values reach the feature matrix.

> **Join NDVI as-of the date it was actually available, not the date it is
> centred on.** MOD13A1 is a 16-day composite with several days of further
> processing latency, so the freshest NDVI available on any given day reflects
> vegetation from roughly 2–4 weeks earlier. At inference you have no choice about
> this. In training you do — and the tempting join (take the composite whose window
> contains the row's date) hands the model vegetation observations from the future.
>
> That mismatch is train/serve skew and it is silent: the model learns to lean on
> an NDVI signal that is fresher than anything it will ever see live, then quietly
> underperforms in production with no error anywhere.
>
> **Implementation:** store an explicit `available_from` date on every NDVI
> composite (= composite end date + observed AppEEARS processing lag; measure the
> lag, do not assume it). Both paths then use the same rule:
>
> ```sql
> -- the most recent composite that had been published on or before the row's date
> ndvi AS (
>   SELECT cell_id, date, ndvi, ndvi_anom
>   FROM features f
>   ASOF LEFT JOIN ndvi n
>     ON f.cell_id = n.cell_id AND n.available_from <= f.date
> )
> ```
>
> DuckDB's `ASOF JOIN` does this directly. `ndvi_delta_32d` must be built from two
> composites that were *both* available by the row's date, for the same reason.
> Verify by spot-checking a mid-season training row: the NDVI attached to it should
> be 2–4 weeks stale, never same-week.

### 5.5 Static terrain

`elevation_m`, `slope_deg`, `aspect_sin`, `aspect_cos`,
`northness` (= cos(aspect) × sin(slope)), `tpi`

### 5.6 Static fuel and land cover

`fuel_type` (categorical — one-hot, or native categorical support),
`pct_conifer`, `pct_grass`
(`is_nonfuel` is a mask, §3, not a feature.)

### 5.7 Human exposure

`dist_to_road_m`, `dist_to_rail_m`, `road_density_5km`,
`dist_to_settlement_m`, `population_density`

Retained despite the cause-agnostic label: roughly 40% of BC ignitions are
human-caused and these are the features that carry that signal. **But see §6.5 —
these features are entangled with detection bias in the labels.**

### 5.8 Temporal

`doy_sin`, `doy_cos`, `is_weekend`

> **`year` is deliberately excluded.** Including it lets the RF memorize which
> seasons happened to be severe, which will not generalize forward.

### 5.9 Out-of-sample fire history (D8)

`hist_ignition_count_prior` — count of ignitions within the cell in all years
strictly before the row's year, normalized by the number of prior years available.

> Must be recomputed per training year. A single full-record count computed once
> is label leakage and will inflate your metrics substantially.

### 5.10 Explicitly excluded

FFMC, DMC, DC, ISI, BUI, FWI (D3) · active-fire proximity (D7) · `year` (§5.8) ·
full-record fire counts (D8)

---

## 6. Ground truth — the accuracy label

This is the foundation of every metric in §9. Get it right first.

### 6.1 Source

**BC Wildfire Service — Fire Incident Locations (Historical)**, from the BC Data
Catalogue. A **point** layer: one row per recorded wildfire incident, carrying
fire number, latitude/longitude, discovery date, cause (lightning / person /
unknown), and final size in hectares.

> **Use incident *points*, not fire *perimeters*.** These are two different BCWS
> datasets. A perimeter describes where a fire ended up after spreading — its
> centroid is not the ignition location and its area is not the ignition cell.
> This model predicts ignition, so it needs the point of origin.

Cross-check the extracted points against the **CNFDB point layer** from NRCan for
the same years. If the two disagree on annual counts by more than a few percent,
resolve it before training — one of your two extractions is wrong.

### 6.2 Label definition

```
label(cell, date) = 1  if >= 1 BCWS incident point falls inside cell
                         with discovery_date == date
                    0  otherwise
```

Cause-agnostic: lightning, person and unknown all count as positives.

### 6.3 Construction

```python
# 1. Load ignition points, reproject to the grid's CRS
pts = gpd.read_file(BCWS_INCIDENTS).to_crs("EPSG:3005")
pts = pts[(pts.discovery_date.dt.year >= 2015)]

# 2. Spatial join -> which cell did each ignition fall in
pts = gpd.sjoin(pts, grid_cells[["cell_id", "geometry"]], predicate="within")

# 3. Collapse to unique (cell_id, date) positives
positives = (pts.groupby(["cell_id", "discovery_date"])
                .size().rename("n_ignitions").reset_index())

# 4. Build the FULL cell-day panel (unmasked cells x fire-season days x years)
panel = cross_join(unmasked_cells, fire_season_dates)

# 5. Left join, fill zero
labels = panel.merge(positives, on=["cell_id", "discovery_date"], how="left")
labels["label"] = (labels.n_ignitions.fillna(0) > 0).astype(int)
```

Step 4 is the one people skip. **The zeros must be constructed explicitly**, from
the full panel of cells × days. If you only ever materialize the fire rows you
have no negatives at all, and if you build the panel over *all* cells rather than
unmasked ones you fill the dataset with lake and glacier pixels that trivially
never burn.

Panel size: ~1,150 unmasked cells × ~214 fire-season days × 10 years ≈
**2.5M cell-days**.

### 6.4 Expected class balance

The Okanagan sees roughly 150–250 ignitions per season, so expect
**~1,500–2,500 positives** against ~2.5M rows — a prevalence of about
**0.06–0.1%**.

Handling it:
- Keep all positives.
- Downsample negatives to roughly **20:1**, stratified by year and month so the
  seasonal cycle is preserved.
- Set `class_weight='balanced_subsample'`.
- Do **not** use SMOTE or other synthetic oversampling — interpolating between
  cell-days in a spatially autocorrelated dataset manufactures leakage.
- **Record the downsampling ratio.** It must be undone when calibrating
  probabilities (§7.2) or output scores are biased high by roughly that ratio.

### 6.5 Known label problems

Document these; do not try to engineer them away.

1. **Detection bias — the important one.** Small fires near roads and settlements
   are far more likely to be spotted and recorded than identical fires in the
   backcountry. Because `dist_to_road_m` and `population_density` are features
   (§5.7), the model will partly learn *detection probability* rather than
   *ignition probability*, and those features' importance will be overstated.
   **Mitigation:** fires above ~1 ha are detected essentially regardless of
   location, so a size threshold largely removes the bias. Train the primary
   model on all recorded ignitions, then **re-run with a ≥1 ha threshold as a
   sensitivity check**. If road-proximity importance drops sharply between the
   two, that gap is the detection-bias estimate — report it.
2. **Discovery date ≠ ignition date.** BCWS records discovery, which can lag true
   ignition by hours to a couple of days for remote fires. Accepted as noise.
   Optional robustness check: allow a ±1-day match window and confirm metrics do
   not move much.
3. **Coordinate precision.** Reported origin points carry positional error, so
   fires near a cell boundary may land in the neighbouring cell. At 5km cells this
   is a small effect; do not add buffering complexity for it.
4. **Suppression is invisible.** The record contains fires that were reported.
   Ignitions extinguished immediately by the person who caused them never appear.
5. **Small-fire reporting has changed over time.** Check the annual count of
   sub-0.1 ha fires across 2015–2024; a step change means a reporting-practice
   shift, not a real trend, and argues for the size threshold.

### 6.6 Validation before use

Do not train until all of these pass:
- Annual positive counts match published BCWS Okanagan/Kamloops Fire Centre
  statistics for each year.
- Every ignition point falls inside exactly one cell; log and inspect any that
  fall in masked cells (a fire recorded "in a lake" means a bad mask or a bad
  coordinate).
- The seasonal distribution of positives peaks in July–August. If it does not,
  the date join is wrong.
- Panel row count equals `unmasked_cells × fire_season_days × years` exactly.

---

## 7. Model

### 7.1 Estimator

`sklearn.ensemble.RandomForestClassifier`

```python
RandomForestClassifier(
    n_estimators=500,
    max_features='sqrt',
    min_samples_leaf=5,        # tune: 1, 5, 10, 25
    class_weight='balanced_subsample',
    n_jobs=-1,
    random_state=42,
)
```

### 7.2 Calibration

RF probabilities are poorly calibrated out of the box, and the negative
downsampling in §6.4 shifts them further. Both must be corrected, because the
output is a *score* people will read, not just a ranking.

1. Wrap in `CalibratedClassifierCV(method='isotonic')`, fit on a held-out season
   not used for training.
2. Apply a prior correction for the downsampling ratio to recover
   true-prevalence probabilities.

### 7.3 Cross-validation

**Leave-one-fire-season-out** over the training years. Report mean ± sd across
folds.

> Random k-fold is invalid here on two counts: neighbouring cells on the same day
> are strongly correlated (spatial leakage), and fire seasons are highly
> autocorrelated at the year level — 2017, 2018, 2021 and 2023 were extreme
> across BC. A single random split mostly measures which years you happened to
> draw. Leave-one-season-out is the honest version and also reports the
> between-year variance, which will be large and is worth knowing.

**Final held-out test:** the two most recent complete fire seasons, touched once,
at the end. Not during development.

---

## 8. Risk score (0–100)

```
p_raw = calibrated_model.predict_proba(X)[:, 1]
score = 100 * percentile_rank(p_raw, reference=REFERENCE_DISTRIBUTION)
```

`REFERENCE_DISTRIBUTION` is the pooled set of predicted probabilities across all
unmasked cells and all fire-season days of the **training period**. It is computed
once and **frozen as a model artifact**, versioned alongside the model. Scores are
not re-normalized against live data — otherwise a quiet day would still show red
somewhere, since percentiles within a single day always span 0–100.

Pooled rather than per-cell, per D11: a regionally severe day should read as
severe across the whole map. Per-cell normalization would make a damp alpine cell
show 100 on its personal driest day, which misrepresents absolute ignition risk.

**Bands:**

| Score | Colour | Label |
|-------|--------|-------|
| 0–25 | Green | Low |
| 26–50 | Yellow | Moderate |
| 51–75 | Orange | High |
| 76–100 | Red | Extreme |

Because the mapping is a percentile against a fixed reference, these bands spread
across the full range over a season rather than collapsing into the middle as a
weighted-average composite would.

> Conditions more extreme than anything in the reference period saturate at 100.
> Accepted (§1 non-goals). The dashboard should not imply resolution above 100.

---

## 9. Evaluation

Everything here is scored against the §6 ignition labels. Both the RF and the FWI
baseline are evaluated on **the same held-out cell-days with the same ground
truth** — that is the only valid comparison (D12).

### 9.1 Primary — PR-AUC

**Average precision** against the binary ignition label, RF vs. FWI baseline.

ROC-AUC is not used: at ~0.1% prevalence it is dominated by true negatives and
will read ~0.95 for almost anything, including a bad model.

Report the no-skill line (= prevalence) alongside, since a PR-AUC of 0.05 sounds
terrible but is a ~50x lift over a base rate of 0.001.

### 9.2 Operational — capture rate

> *Of all ignitions that actually occurred in the held-out period, what fraction
> fell in the top 5% / top 10% of cell-days by score?*

Reported for RF and FWI side by side, plus lift over random. **Lead with this
number when presenting** — it is directly interpretable as "if you could watch
10% of the region, how much of the problem would you see?"

### 9.3 Diagnostic — FWI agreement

```
fwi_norm      = percentile_rank(FWI, reference=training-period FWI distribution)
agreement_mae = mean(|score/100 - fwi_norm|)
```

> **Not an accuracy metric.** It measures how far the model departs from FWI, and
> is minimized to zero by a model that merely reproduces FWI — so the RF can match
> but never beat the baseline on it. Never train against it and never report it as
> a headline.
>
> It earns its place diagnostically: isolate the cell-days where RF and FWI
> disagree most, then check the §6 labels for those cells. **If the RF is right
> where it disagrees with FWI, that is the single strongest piece of evidence the
> project works** — and it is a much better figure than any aggregate number.

### 9.4 Baseline construction

The FWI baseline requires a full Van Wagner (1987) implementation over the same
historical record (D4), then percentile-normalized against the training period so
it is on the same 0–1 scale as the model score.

- FFMC, DMC, DC are **recursive** — each day depends on the previous day.
- Inputs are **noon LST** temperature, RH, wind (km/h at 10m) and the **24-hour
  precipitation accumulated to noon**.
- Startup values FFMC=85, DMC=6, DC=15 are *spring-after-snowmelt* values. Start
  each season's recursion at snowmelt, not at an arbitrary pipeline start date.
  Because the whole record is backfilled up front this is straightforward — but do
  not carry DC across the winter gap without overwintering logic.
- **Validate the implementation against CWFIS published values** for a station in
  the region (Kelowna or Vernon) before trusting any result. A subtle bug in the
  DMC rain routine is otherwise invisible, and it would corrupt the baseline —
  which is to say, the project's central comparison.

---

## 10. Pipeline

Two distinct paths. Keep them separated in code; they share only `src/features/`.

### 10.1 Training path (run occasionally, offline)

```
BACKFILL
├── Build weather_nodes (~350) and grid_cells (~1,150) in EPSG:3005
├── Load static layers: DEM, fuel type, roads, settlements -> grid_cells
├── Open-Meteo Archive (ERA5): 350 nodes x Feb 1-Oct 31 x 10 years
│     -> backfill_era5 DAG: ~90 mapped tasks, retries, gap gate (10.4)
├── AppEEARS area request: MOD13A1 NDVI 2015-present -> QA screen -> zonal stats
├── BCWS incident points -> cell-day labels (section 6) -> validate (6.6)
└── Derive climatologies (weather day-of-year, NDVI per composite period)

FEATURE BUILD  (src/features/ - SHARED with inference)
├── Rolling windows, anomalies, deltas
├── Out-of-sample historical ignition counts (per training year)
└── -> training_matrix.parquet

TRAIN
├── Negative downsampling (20:1, stratified)
├── Leave-one-season-out CV -> hyperparameters
├── Fit final RF + isotonic calibration + prior correction
├── Freeze REFERENCE_DISTRIBUTION for the 0-100 mapping
└── -> models/rf_YYYYMMDD.joblib  (versioned, with a metadata sidecar)

EVALUATE
├── Compute Van Wagner FWI over the same record (baseline)
├── PR-AUC + capture rate, RF vs FWI, on the same labels
├── FWI-disagreement diagnostic (9.3)
└── Held-out final test on the two most recent seasons
```

### 10.2 Inference path (daily, automated)

```
INGEST
├── Open-Meteo Forecast (ecmwf_ifs025), 350 nodes, today + horizon
├── FIRMS hotspots (display layer)
└── NDVI refresh (weekly)

FEATURE BUILD
└── Same src/features/ code, reading the >=90-day rolling weather history

SCORE
├── Apply snow mask, non-fuel mask, off-season check
├── Load model artifact -> predict_proba -> percentile -> 0-100
└── -> risk_scores table

SERVE
└── FastAPI reads latest scores -> GeoJSON -> React dashboard
```

### 10.3 API budget

This is why D10 exists. Open-Meteo weights a call by roughly
`(variables/10) x (days/14) x locations`.

**Backfill:** 350 nodes × 8 variables × ~2,730 days ≈ **~44,000 call-units**
against a free-tier allowance of ~10,000/day. So the backfill is a **multi-day,
resumable job** — chunk by (year, node-batch), checkpoint every chunk to Parquet,
and make re-running skip completed chunks. Budget several days of wall clock and
**start it early**; nothing else can be validated until it lands.

**Daily inference:** 350 nodes × 8 variables × ~7 days ≈ **~140 call-units/day.**
Trivial.

> Open-Meteo's free tier is non-commercial and requires attribution (CC-BY-4.0).

### 10.4 Airflow DAGs (D14)

Two DAGs. Both import from `src/` — **no pipeline logic lives in the DAG files**,
which contain only task wiring. This keeps everything runnable and testable
outside Airflow, which matters when debugging.

**`backfill_era5`** — run manually, not scheduled.

```python
@task
def chunk_list() -> list[dict]:
    # ~10 years x ~9 node batches of 40 = ~90 chunks
    return [{"year": y, "node_batch": b} for y in YEARS for b in NODE_BATCHES]

fetch = fetch_chunk.expand(chunk=chunk_list())   # dynamic task mapping
fetch >> verify_no_gaps() >> load_to_duckdb()
```

- `max_active_tasks=4` — the constraint is Open-Meteo's rate limit, not local CPU.
- `retries=5`, `retry_exponential_backoff=True`, `max_retry_delay=timedelta(hours=1)`.
  A 429 is expected, not exceptional; it should cost a retry, not a DAG failure.
- Each mapped task writes **its own Parquet file** and nothing else. Do not write
  to DuckDB from mapped tasks (§15.18).
- Re-running skips chunks whose Parquet already exists, so a partial run resumes
  by simply clearing failed tasks in the UI.
- `verify_no_gaps` asserts every `(node_id, date)` in the expected range is present
  before the load proceeds. A silent gap corrupts every rolling feature that spans
  it (§15.9), so this gate is the point of the DAG.

**`daily_score`** — `schedule="0 14 * * *"` (≈06:00 Pacific, after the IFS run lands).

```
fetch_forecast >> build_features >> score >> export_geojson
                                 >> dbt_test  (>> alert on failure)
fetch_firms    >> load_firms                  # parallel, display layer only
```

- `catchup=False`. This DAG produces today's nowcast; backfilling old runs is
  meaningless and would stampede the scheduler.
- `max_active_runs=1` — the DuckDB write lock makes concurrent runs unsafe (§15.5).
- The whole DAG should finish in minutes. If it does not, something is wrong.

**Weekly** NDVI refresh can be a third DAG or a weekday-gated branch in
`daily_score`; the AppEEARS submit→poll→download cycle is asynchronous and can
take hours, so if it becomes its own DAG use a sensor with a generous timeout and
`reschedule` mode rather than blocking a worker slot.

---

## 11. Storage schema

DuckDB (single file) + Parquet on disk.

```
weather_nodes    (node_id, lat, lon, x_albers, y_albers)
grid_cells       (cell_id, geometry, centroid_lat, centroid_lon, weather_node_id,
                  elevation_m, slope_deg, aspect_deg, tpi, fuel_type, is_nonfuel,
                  pct_conifer, pct_grass, dist_to_road_m, dist_to_rail_m,
                  road_density_5km, dist_to_settlement_m, population_density)
weather_daily    (node_id, date, temp_noon_c, temp_max_c, rh_noon_pct, rh_min_pct,
                  wind_speed_noon_kmh, wind_gust_max_kmh, wind_dir_deg,
                  precip_24h_mm, snow_depth_m, sm_0_7,
                  source)   -- sm_0_7 only, per D13; source: 'era5' | 'ifs'
weather_clim     (node_id, doy, var_name, mean, sd)
ndvi             (cell_id, composite_date, composite_end_date, available_from,
                  ndvi, qa_ok)   -- available_from drives the ASOF join, section 5.4
ndvi_clim        (cell_id, composite_period, mean, sd)
ignitions        (incident_id, cell_id, discovery_date, cause, size_ha, lat, lon)
labels           (cell_id, date, label, n_ignitions)   -- the full cell-day panel
features         (cell_id, date, <all features from section 5>)
fwi_daily        (node_id, date, ffmc, dmc, dc, isi, bui, fwi)   -- baseline only
risk_scores      (cell_id, run_date, valid_date, p_calibrated, score,
                  classification, model_version)
active_fires     (fire_id, lat, lon, bright_ti4, confidence, frp, acq_datetime, source)

VIEW current_risk_map  -- latest run_date, valid_date = today
```

> `risk_scores` carries **both** `run_date` (when the model ran) and `valid_date`
> (the day being predicted). With only one timestamp you can never ask "what did
> we predict yesterday for today," you can never verify a forecast, and re-runs
> silently duplicate or overwrite. Primary key `(cell_id, run_date, valid_date)`,
> upsert on conflict.

> `weather_daily.source` records whether a row came from ERA5 or IFS. Where the
> two overlap for the same date, any systematic offset in soil moisture is a
> train/serve skew warning worth acting on.

---

## 12. Tech stack

| Layer | Tool | Notes |
|-------|------|-------|
| **Language** | Python 3.11 (conda, `environment.yml`) | |
| **HTTP / ingest** | `requests` | Add retry with exponential backoff; Open-Meteo returns 429 under load. |
| **Dataframes** | `pandas`, `numpy` | |
| **Geospatial** | `geopandas`, `shapely`, `pyproj`, `fiona`, `rasterio`, `rioxarray`, `xarray` | `sjoin` for labels (§6.3); `rioxarray`/`xarray` for NDVI zonal stats. All metric work in EPSG:3005. |
| **File format** | Apache Parquet (`pyarrow`) | All raw and intermediate data. |
| **Database** | DuckDB + `spatial` extension | Embedded, no server. Reads Parquet directly. |
| **Transform** | dbt-duckdb | Staging → intermediate → marts, plus data-quality tests. |
| **ML** | `scikit-learn` | RandomForestClassifier, CalibratedClassifierCV, `average_precision_score`. |
| **Model persistence** | `joblib` | Model + reference distribution + feature-name list + metadata sidecar. |
| **FWI baseline** | Custom Van Wagner (1987) implementation in Python | `cffdrs` is primarily an R package; Python ports vary in quality. Whatever you use, validate against CWFIS (§9.4). Evaluation path only. |
| **Config / secrets** | `python-dotenv`, `.env` (gitignored) | `FIRMS_MAP_KEY`, `AppEARS_user`, `AppEARS_pass`. |
| **Orchestration** | **Apache Airflow** (D14) | Two DAGs (§10.4). Runs in its own containers via Docker Compose — **not** installed into the project conda env (§12.1). LocalExecutor is sufficient. |
| **API** | FastAPI + `uvicorn` | Serves GeoJSON and cell drill-downs. |
| **Frontend** | React + Deck.gl (or Leaflet) | ~1,150 cells is small; either works. Send geometry once as a static file, then only `cell_id → score` per day. |
| **Plotting (analysis)** | `matplotlib`, `plotly` | PR curves, capture-rate curves, feature importance, disagreement maps. |
| **Testing** | `pytest` + dbt tests | |
| **Deployment** | Docker Compose | |

### 12.1 `environment.yml` corrections needed

The current file predates these decisions.

**Add:** `scikit-learn`, `duckdb`, `dbt-duckdb`, `joblib`, `xarray`, `rioxarray`, `scipy`
**Remove:** `sqlalchemy`, `geoalchemy2` (Postgres ORM stack — superseded by DuckDB), `schedule` (superseded by Airflow)
**Keep:** everything else, including `psycopg2` — Airflow's metadata database is Postgres.

> **Do not add `apache-airflow` to `environment.yml`.** Airflow pins a large,
> opinionated dependency set that conflicts badly with the scientific stack
> (`pandas`, `numpy`, `geopandas`, `scikit-learn` version ranges in particular),
> and resolving them together produces a fragile env that breaks on any upgrade.
>
> Run Airflow from the official Docker image via `docker-compose.yml`, with `src/`
> and `dags/` mounted in and the project's own dependencies installed into a
> derived image. The conda env stays for local development, notebooks and running
> pipeline steps by hand — which should always remain possible, since `dags/`
> holds only wiring.

### 12.2 Repository layout

```
wildfire-modelling/
├── src/
│   ├── grid/           # Grid construction, masking, static layer joins
│   ├── ingest/         # Open-Meteo (archive + forecast), FIRMS, AppEEARS, BCWS
│   ├── labels/         # Section 6 - ignition points -> cell-day panel
│   ├── features/       # SHARED feature computation - train and serve
│   ├── fwi/            # Van Wagner implementation (baseline only)
│   ├── model/          # Train, calibrate, evaluate, score
│   └── utils/
├── dags/               # Airflow DAGs - task wiring ONLY, logic lives in src/
│   ├── backfill_era5.py
│   └── daily_score.py
├── dbt/                # Staging/intermediate/mart models + tests
├── api/                # FastAPI
├── frontend/           # React + Deck.gl
├── models/             # Versioned .joblib artifacts + metadata (gitignored)
├── notebooks/          # EDA, evaluation writeups
├── data/               # ALL data - gitignored, regenerable
│   ├── raw/{weather_archive,weather_forecast,firms,ndvi,bcws}/
│   ├── static/
│   └── processed/
├── wildfire.duckdb
├── environment.yml
└── wildfire-dashboard-spec.md   # this file
```

**`.gitignore` must contain:**
```
data/
models/
*.duckdb
*.duckdb.wal
*.parquet
.env
```

> The current `.gitignore` uses `data/*` (not `data/`), omits `*.duckdb`, and
> ignores both itself and this spec file. Fix it — this spec should be tracked.

---

## 13. Known limitations

Be upfront about these in any writeup.

1. **No lightning data.** ~60% of BC ignitions are lightning-caused, and the CLDN
   feed is not freely available in real time. With a cause-agnostic label and no
   lightning covariate, a large share of positives are effectively unpredictable
   from the feature set. This caps achievable performance and is the single
   largest source of irreducible error. Expect it to show up as low recall on
   remote high-elevation ignitions.
2. **Detection bias in the labels.** See §6.5.1 — partially mitigated by the size
   threshold, not eliminated.
3. **RF cannot extrapolate.** Predictions flatten beyond the range of conditions
   in the training record — exactly the record-breaking season where the signal
   matters most. Accepted per §1. FWI, being physical, degrades more gracefully,
   which is a reason to keep displaying it alongside.
4. **Region-specific.** Trained on ~1,500–2,500 Okanagan positives. Will not
   transfer to the coast or the north. Do not claim otherwise.
5. **Small positive count.** Between-year variance will be large; report CV
   spread, not just the mean.
6. **Reanalysis vs. forecast.** Training on ERA5 and serving on IFS keeps the
   variables and physics aligned (D6), but reanalysis is still better-conditioned
   than a forecast. Expect live performance slightly below CV performance.

---

## 14. Implementation phases

Each phase ends in something verifiable. Do not build ahead.

| Phase | Deliverable | Done when |
|-------|-------------|-----------|
| **0** | Fix `environment.yml` and `.gitignore` (§12.1, §12.2); stand up Airflow via Docker Compose with `src/` and `dags/` mounted | Env builds clean; a trivial DAG runs end to end in the Airflow UI |
| **1** | Open-Meteo probe: one node, ECMWF IFS, assert all 8 variables return | Archive and forecast APIs both return `soil_moisture_0_to_7cm` (exact name, D13) with no nulls |
| **2** | Grid build: weather nodes + risk cells in EPSG:3005, static layers joined, non-fuel mask applied | Cell count and lake masking visually verified on a map |
| **3** | ERA5 backfill via the `backfill_era5` DAG (§10.4). **Start early, runs for days** | 10 seasons × 350 nodes on disk; `verify_no_gaps` passes |
| **4** | **Ground truth: BCWS points → cell-day panel (§6)** | All four §6.6 validation checks pass |
| **5** | Van Wagner FWI + validation against CWFIS for Kelowna | Computed FWI matches CWFIS within tolerance |
| **6** | NDVI history via AppEEARS area request, QA-screened, `available_from` measured, climatology built | No fill values in the feature matrix; a spot-checked mid-season training row carries NDVI 2–4 weeks stale (§5.4), not same-week |
| **7** | Feature build → training matrix | Row count, null audit, leakage check on §5.9 |
| **8** | Train + calibrate + leave-one-season-out CV | PR-AUC and capture rate vs. FWI on the same labels, with CV spread |
| **9** | Freeze model artifact + reference distribution; daily inference path | Scores produced end to end for today |
| **10** | FastAPI + dashboard | Map renders, drill-down works, stale-data state handled |
| **11** | Held-out test on the two most recent seasons | Touched exactly once |

> Phase 4 gates everything downstream — without trustworthy labels no metric means
> anything. Do it before the modelling work, not alongside it.
>
> The project's central question is answered at **Phase 8**: *does the RF beat FWI
> on PR-AUC and capture rate against observed ignitions?* Everything before that is
> plumbing and everything after is presentation. If the answer is no, that is a
> legitimate and reportable finding — say so rather than tuning until it flips.

---

## 15. Implementation gotchas

Concrete things that will silently break. Each has cost time on projects like this.

1. **Pin `models=ecmwf_ifs025`.** Assert the returned variable list matches the
   request. (§4.1)
2. **Never mix soil moisture depth conventions.** The only variable used is
   ERA5/IFS `soil_moisture_0_to_7cm` (D6, D13). ICON's nearest band is
   `0_to_1cm`/`1_to_3cm` and GFS's is `0_to_10cm` — **different depths, not
   drop-in substitutes**. If a model switch ever silently swaps the band, the
   values still look plausible (all are volumetric m³/m³ in roughly the same
   range) while meaning something different. Assert the exact variable name in
   the response, not just that a soil moisture column exists.
3. **Build the negatives explicitly** from the full cell-day panel (§6.3 step 4),
   over unmasked cells only.
4. **`ST_Distance` in DuckDB is planar and returns *input units*.** On raw lat/lon
   that is degrees, not metres. Use `ST_Distance_Sphere`, or transform to
   EPSG:3005 first. Same for the label `sjoin` — reproject first.
5. **DuckDB allows one writer.** The daily pipeline holds the write lock; FastAPI
   must open with `read_only=True` or read exported Parquet, or you get
   `Could not set lock on file` at the worst moment.
6. **Timezones.** `timezone=America/Vancouver` returns DST-shifted local time, so
   "noon" moves by an hour in March and the 24h precip window drifts with it. FWI
   is defined on local *standard* time. Fetch UTC and convert deliberately. Also
   check that BCWS discovery dates are local dates before joining them to
   weather dates.
7. **Wind units.** FWI expects km/h at 10m. Open-Meteo's default is km/h — assert
   it rather than assuming; a silent switch to m/s corrupts ISI with no error.
8. **Rolling windows need a 90-day warm-up.** Backfill from Feb 1; keep ≥90 days
   of live history. A gap silently produces a wrong `precip_sum_90d`.
9.  **Backfill must be idempotent and resumable.** Upsert on `(node_id, date)`.
   Detect gaps explicitly before feature build — a missing day corrupts every
   rolling feature spanning it, with no error raised.
10. **Out-of-sample fire counts must be recomputed per year** (§5.9). The easiest
    bug in the project, and it inflates results convincingly.
11. **Undo the negative downsampling when calibrating** (§7.2), or scores are
    biased high by roughly the downsampling ratio.
12. **NDVI fill values are −3000.** Screen on QA before any arithmetic.
13. **Join NDVI as-of its availability date, not its composite date** (§5.4). The
    natural join — "the composite whose 16-day window contains this row's date" —
    gives training rows vegetation data from the future, which inference can never
    have. Use `available_from` (composite end + measured processing lag) and an
    `ASOF JOIN` in both paths. Silent, and it costs you live accuracy rather than
    raising an error. Spot-check: a mid-season training row's NDVI should be 2–4
    weeks stale.
14. **Persist the feature-name list with the model artifact** and assert column
    order at inference. Silent column reordering between train and serve produces
    plausible-looking nonsense.
15. **FIRMS `confidence` is categorical for VIIRS** (`l`/`n`/`h`), not numeric as
    for MODIS. Schema accordingly.
16. **The 0–100 reference distribution is frozen at train time.** Do not recompute
    it against live data (§8).
17. **Airflow lives in Docker, not in the conda env** (§12.1). Installing
    `apache-airflow` alongside `geopandas`/`scikit-learn` produces a dependency
    resolution that appears to work and then breaks on the next upgrade.
18. **Never write to DuckDB from parallel mapped tasks.** DuckDB permits a single
    writer (§15.5), so four concurrent `fetch_chunk` tasks writing to
    `wildfire.duckdb` will deadlock or corrupt. Mapped tasks write **one Parquet
    file each**; a single downstream task loads them. This is why `backfill_era5`
    is shaped the way it is in §10.4.
19. **`catchup=False` on `daily_score`.** It defaults to `True`. With a
    `start_date` set in the past, enabling the DAG immediately schedules one run
    per missed day, every one of them competing for the same DuckDB write lock.
    Set `max_active_runs=1` as a second line of defence.
20. **Do not pass DataFrames through XCom.** XCom is for small values and is
    stored in the metadata DB. Tasks pass **file paths**; the data moves through
    Parquet.
21. **Every task must be idempotent**, because retries and manual clears will
    re-run them. Upsert on `(node_id, date)` and `(cell_id, run_date, valid_date)`,
    never blind insert. Airflow makes re-execution routine rather than
    exceptional, so this stops being a nicety.
22. **Airflow's rate-limit handling is retries, not sleeps.** Use
    `retries` + `retry_exponential_backoff` on the fetch task; do not
    `time.sleep()` inside a task to dodge 429s — it burns a worker slot and hides
    the failure from the UI.
23. **dbt tests that actually catch things:** cell coverage per run (did all
    unmasked cells get a score?), weather history continuity (no missing dates),
    label panel row count, NDVI staleness (assert every row's NDVI is *at least*
    the processing lag old — catches gotcha 13 directly, and flag anything older
    than ~32 days as stale), all-null soil moisture (catches the §4.1 failure),
    score range. `score BETWEEN 0 AND 100` alone passes trivially if you clamp.

---

## 16. Attribution and disclaimer

- **Open-Meteo** — CC-BY-4.0, non-commercial use, attribution required.
- **NASA FIRMS / MODIS** — attribution and citation required.
- **BC Open Data / BC Wildfire Service** — Open Government Licence – British Columbia.
- **CFFDRS / CWFIS** — Natural Resources Canada.

> The dashboard must carry a visible disclaimer: this is a student research
> project, not an operational fire-danger product, and must not be used for
> safety, planning or emergency decisions. Official BC fire danger ratings come
> from the BC Wildfire Service.