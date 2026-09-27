#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds, transform as window_transform
from rasterio.warp import Resampling, reproject
from shapely.geometry import Polygon

LOGGER = logging.getLogger("preprocess")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
DEFAULT_OUT_DIR = ROOT / "outputs"


@dataclass


class Config:
    study_area_path: Path = RAW_DIR / "study_area_boundary.geojson"
    stations_path: Path = RAW_DIR / "ovitrap_monitoring_stations.gpkg"
    traps_path: Path = RAW_DIR / "ovitrap_trap_records.csv"
    landcover_path: Path = RAW_DIR / "landcover_10m_2021.tif"
    population_path: Path = RAW_DIR / "population_100m.tif"
    era5_dir: Path = RAW_DIR / "era5_land_hourly"
    mspa_raster_path: Path = INTERIM_DIR / "mspa_greenspace_7class.tif"

    mspa_input_path: Path = INTERIM_DIR / "mspa_input_binary.tif"
    analysis_grid_path: Path = INTERIM_DIR / "analysis_grid_100m.tif"
    out_dir: Path = DEFAULT_OUT_DIR

    station_id_col: str = "station_id"
    station_lon_col: str = "longitude"
    station_lat_col: str = "latitude"
    trap_date_col: str = "survey_date"
    trap_positive_col: str = "n_positive"
    trap_deployed_col: str = "n_deployed"

    metric_crs: str = "EPSG:32649"
    geographic_crs: str = "EPSG:4326"

    period_start: str = "2024-05-01"
    period_end: str = "2024-09-30"
    buffer_radius_m: float = 1000.0
    hexagon_area_km2: float = 5.0
    analysis_resolution_m: float = 100.0
    min_stations_per_hexagon: int = 1

    greenspace_landcover_codes: Tuple[int, ...] = (10, 20)
    mspa_foreground_value: int = 2
    mspa_background_value: int = 1

    mspa_class_codes: Dict[int, str] = field(default_factory=lambda: {
        1: "Core", 2: "Islet", 3: "Perforation", 4: "Edge",
        5: "Loop", 6: "Bridge", 7: "Branch",
    })

    era5_t2m_var: str = "t2m"
    era5_d2m_var: str = "d2m"
    era5_unit_is_kelvin: bool = True
    era5_time_dim: str = "time"
    era5_lat_dim: str = "latitude"
    era5_lon_dim: str = "longitude"

    hi_unit: str = "F"

    moi_aggregation: str = "pooled"

    all_touched: bool = True


def minmax(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype="float64")
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr)
    lo, hi = np.nanmin(arr[finite]), np.nanmax(arr[finite])
    if not np.isfinite(hi - lo) or hi - lo == 0.0:
        return np.zeros_like(arr)
    out = np.zeros_like(arr)
    out[finite] = (arr[finite] - lo) / (hi - lo)
    return out


def window_for_bounds(
    bounds: Sequence[float],
    crs_transform,
    width: int,
    height: int,
) -> Optional[Window]:
    win = from_bounds(*bounds, transform=crs_transform).intersection(
        Window(0, 0, width, height)
    )
    win = win.round_offsets().round_lengths()
    if win.width <= 0 or win.height <= 0:
        return None
    return win


def read_window(
    src: rasterio.DatasetReader, bounds: Sequence[float]
) -> Tuple[Optional[np.ndarray], Optional[Window]]:
    win = window_for_bounds(bounds, src.transform, src.width, src.height)
    if win is None:
        return None, None
    return src.read(1, window=win), win


def read_vector(path: Path, crs: Optional[str] = None) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        if crs is None:
            raise ValueError(f"{path} has no CRS and none was supplied")
        gdf = gdf.set_crs(crs)
    return gdf


def dissolve_geometry(gdf: gpd.GeoDataFrame):
    if hasattr(gdf.geometry, "union_all"):
        return gdf.geometry.union_all()
    return gdf.geometry.unary_union


def compute_station_moi(cfg: Config) -> pd.DataFrame:
    traps = pd.read_csv(cfg.traps_path, parse_dates=[cfg.trap_date_col])
    start, end = pd.Timestamp(cfg.period_start), pd.Timestamp(cfg.period_end)
    traps = traps.loc[traps[cfg.trap_date_col].between(start, end)].copy()
    if traps.empty:
        raise ValueError("no ovitrap records fall inside the configured study period")

    id_col = cfg.station_id_col
    grouped = traps.groupby(id_col, as_index=False)

    if cfg.moi_aggregation == "pooled":
        out = grouped[[cfg.trap_positive_col, cfg.trap_deployed_col]].sum()
        deployed = out[cfg.trap_deployed_col].to_numpy(dtype="float64")
        positive = out[cfg.trap_positive_col].to_numpy(dtype="float64")
        out["MOI"] = np.where(deployed > 0, positive / deployed * 100.0, np.nan)
    elif cfg.moi_aggregation == "mean_of_rounds":
        deployed = traps[cfg.trap_deployed_col].to_numpy(dtype="float64")
        positive = traps[cfg.trap_positive_col].to_numpy(dtype="float64")
        traps["_round_moi"] = np.where(deployed > 0, positive / deployed * 100.0, np.nan)
        out = (
            traps.groupby(id_col, as_index=False)[["_round_moi"]]
            .mean()
            .rename(columns={"_round_moi": "MOI"})
        )
    else:
        raise ValueError(f"unknown moi_aggregation: {cfg.moi_aggregation!r}")

    out = out.rename(columns={id_col: "station_id"})
    LOGGER.info("S1   MOI computed for %d monitoring stations", len(out))
    return out[["station_id", "MOI"]]


def load_stations(cfg: Config) -> gpd.GeoDataFrame:
    path = cfg.stations_path
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if not {cfg.station_lon_col, cfg.station_lat_col}.issubset(df.columns):
            raise KeyError(
                "station CSV must contain "
                f"{cfg.station_lon_col!r} and {cfg.station_lat_col!r} columns"
            )
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df[cfg.station_lon_col], df[cfg.station_lat_col]),
            crs=cfg.geographic_crs,
        )
    else:
        gdf = read_vector(path, cfg.geographic_crs)

    gdf = gdf.rename(columns={cfg.station_id_col: "station_id"})
    if "station_id" not in gdf.columns:
        raise KeyError("station table must contain an identifier column")
    return gdf.to_crs(cfg.metric_crs)


def build_buffers(stations: gpd.GeoDataFrame, radius_m: float) -> gpd.GeoDataFrame:
    buffered = stations[["station_id", "geometry"]].copy()
    buffered["geometry"] = stations.geometry.buffer(radius_m)
    return buffered


def build_binary_greenspace(cfg: Config) -> Path:
    cfg.mspa_input_path.parent.mkdir(parents=True, exist_ok=True)
    aoi = read_vector(cfg.study_area_path, cfg.geographic_crs).to_crs(cfg.metric_crs)

    with rasterio.open(cfg.landcover_path) as src:
        aoi_in_src_crs = aoi.to_crs(src.crs) if src.crs is not None else aoi
        win = window_for_bounds(
            dissolve_geometry(aoi_in_src_crs).bounds,
            src.transform, src.width, src.height,
        )
        if win is None:
            raise ValueError("the study area does not overlap the land cover raster")
        data = src.read(1, window=win)
        transform = src.window_transform(win)
        raster_crs = src.crs
        nodata = src.nodata

    valid = np.ones(data.shape, dtype=bool) if nodata is None else data != nodata
    greenspace = np.isin(data, list(cfg.greenspace_landcover_codes))

    binary = np.zeros(data.shape, dtype="uint8")
    binary[valid] = cfg.mspa_background_value
    binary[valid & greenspace] = cfg.mspa_foreground_value

    with rasterio.open(
        cfg.mspa_input_path, "w", driver="GTiff",
        height=binary.shape[0], width=binary.shape[1], count=1, dtype="uint8",
        crs=raster_crs, transform=transform, nodata=0, compress="lzw",
    ) as dst:
        dst.write(binary, 1)

    share = float(greenspace[valid].mean()) if valid.any() else float("nan")
    LOGGER.info(
        "S3   MSPA input written to %s (greenspace = %.2f%% of valid pixels)",
        cfg.mspa_input_path, 100.0 * share,
    )
    return cfg.mspa_input_path


def zonal_class_counts(
    raster_path: Path,
    geometries: Sequence,
    class_values: Sequence[int],
    all_touched: bool = True,
) -> np.ndarray:
    counts = np.zeros((len(geometries), len(class_values)), dtype="int64")
    lookup = {int(v): i for i, v in enumerate(class_values)}

    with rasterio.open(raster_path) as src:
        nodata = src.nodata
        for row, geom in enumerate(geometries):
            data, win = read_window(src, geom.bounds)
            if data is None:
                continue
            inside = geometry_mask(
                [geom], data.shape, src.window_transform(win),
                invert=True, all_touched=all_touched,
            )
            if nodata is not None:
                inside &= data != nodata
            values = data[inside].ravel()
            if values.size == 0:
                continue
            uniq, freq = np.unique(values, return_counts=True)
            for value, count in zip(uniq.tolist(), freq.tolist()):
                index = lookup.get(int(value))
                if index is not None:
                    counts[row, index] += count
    return counts


def pixel_area_km2(raster_path: Path) -> float:
    with rasterio.open(raster_path) as src:
        if src.crs is not None and src.crs.is_geographic:
            raise ValueError(
                "pixel areas are only meaningful in a projected CRS; "
                "reproject the raster before extracting areas"
            )
        return abs(src.transform.a * src.transform.e) / 1e6


def greenspace_fraction(cfg: Config, buffers: gpd.GeoDataFrame) -> np.ndarray:
    fractions = np.full(len(buffers), np.nan, dtype="float64")

    with rasterio.open(cfg.landcover_path) as src:
        nodata = src.nodata
        for row, geom in enumerate(buffers.geometry):
            data, win = read_window(src, geom.bounds)
            if data is None:
                continue
            inside = geometry_mask(
                [geom], data.shape, src.window_transform(win),
                invert=True, all_touched=cfg.all_touched,
            )
            if nodata is not None:
                inside &= data != nodata
            values = data[inside]
            if values.size == 0:
                continue
            fractions[row] = float(
                np.isin(values, list(cfg.greenspace_landcover_codes)).mean()
            )

    LOGGER.info(
        "S4   greenspace fraction: mean = %.4f (n valid = %d)",
        float(np.nanmean(fractions)), int(np.isfinite(fractions).sum()),
    )
    return fractions


def relative_humidity(t_c: np.ndarray, td_c: np.ndarray) -> np.ndarray:
    t_c = np.asarray(t_c, dtype="float64")
    td_c = np.asarray(td_c, dtype="float64")
    es_t = 6.112 * np.exp(17.62 * t_c / (243.12 + t_c))
    es_td = 6.112 * np.exp(17.62 * td_c / (243.12 + td_c))
    with np.errstate(divide="ignore", invalid="ignore"):
        rh = 100.0 * es_td / es_t
    return np.clip(rh, 0.0, 100.0)


def heat_index_fahrenheit(t_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    t_f = np.asarray(t_c, dtype="float64") * 9.0 / 5.0 + 32.0
    rh = np.asarray(rh, dtype="float64")

    hi = (
        -42.379
        + 2.04901523 * t_f
        + 10.14333127 * rh
        - 0.22475541 * t_f * rh
        - 6.83783e-3 * t_f ** 2
        - 5.481717e-2 * rh ** 2
        + 1.22874e-3 * t_f ** 2 * rh
        + 8.5282e-4 * t_f * rh ** 2
        - 1.99e-6 * t_f ** 2 * rh ** 2
    )

    low = (rh < 13.0) & (t_f >= 80.0) & (t_f <= 112.0)
    low_delta = ((13.0 - rh) / 4.0) * np.sqrt(
        np.clip((17.0 - np.abs(t_f - 95.0)) / 17.0, 0.0, None)
    )
    hi = np.where(low, hi - low_delta, hi)

    high = (rh > 85.0) & (t_f >= 80.0) & (t_f <= 87.0)
    high_delta = ((rh - 85.0) / 10.0) * ((87.0 - t_f) / 5.0)
    hi = np.where(high, hi + high_delta, hi)

    simple = 0.5 * (t_f + 61.0 + (t_f - 68.0) * 1.2 + rh * 0.094)
    return np.where((simple + hi) / 2.0 < 80.0, simple, hi)


def build_wet_heat_field(
    cfg: Config,
) -> Tuple[np.ndarray, Tuple[float, float, float, float, float, float]]:
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError(
            "reading ERA5-Land NetCDF files requires xarray "
            "(pip install xarray netcdf4)"
        ) from exc

    files = sorted(cfg.era5_dir.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"no NetCDF files found in {cfg.era5_dir}")

    ds = xr.open_mfdataset(files, combine="by_coords")

    if "expver" in ds.dims:
        ds = ds.mean(dim="expver", skipna=True)

    time_name = cfg.era5_time_dim
    ds = ds.sel({time_name: slice(cfg.period_start, cfg.period_end)})
    if ds.sizes.get(time_name, 0) == 0:
        raise ValueError("no ERA5-Land time steps inside the configured period")

    t = ds[cfg.era5_t2m_var]
    td = ds[cfg.era5_d2m_var]
    if cfg.era5_unit_is_kelvin:
        t = t - 273.15
        td = td - 273.15

    rh = relative_humidity(t.values, td.values)
    hi = heat_index_fahrenheit(t.values, rh)
    if cfg.hi_unit.upper() == "C":
        hi = (hi - 32.0) * 5.0 / 9.0
    hi_mean = np.nanmean(hi, axis=0)

    lat = ds[cfg.era5_lat_dim].values
    lon = ds[cfg.era5_lon_dim].values
    if lat[0] > lat[-1]:
        hi_mean = np.flipud(hi_mean)
        lat = lat[::-1]

    d_lon = float(np.abs(np.diff(lon)).mean()) if lon.size > 1 else 0.25
    d_lat = float(np.abs(np.diff(lat)).mean()) if lat.size > 1 else 0.25
    transform = (
        float(lon.min() - d_lon / 2.0), d_lon, 0.0,
        float(lat.max() + d_lat / 2.0), 0.0, -d_lat,
    )

    LOGGER.info(
        "S5   Heat Index field %s, period mean = %.2f %s",
        hi_mean.shape, float(np.nanmean(hi_mean)), cfg.hi_unit.upper(),
    )
    return hi_mean, transform


def build_analysis_grid(cfg: Config) -> Tuple[np.ndarray, dict, np.ndarray]:
    with rasterio.open(cfg.population_path) as src:
        grid_transform = src.transform
        grid_crs = src.crs
        height, width = src.height, src.width
        pop_nodata = src.nodata
        pop = src.read(1).astype("float64")

    if pop_nodata is not None:
        pop = np.where(pop == pop_nodata, np.nan, pop)
    pop = np.nan_to_num(pop, nan=0.0)

    pixel_size = abs(grid_transform.a)
    if not np.isclose(pixel_size, cfg.analysis_resolution_m, rtol=0.05):
        LOGGER.warning(
            "population raster resolution (%.1f m) differs from the configured "
            "analysis resolution (%.1f m); the population grid governs",
            pixel_size, cfg.analysis_resolution_m,
        )

    hi, hi_transform = build_wet_heat_field(cfg)
    hi_on_grid = np.full((height, width), np.nan, dtype="float64")
    reproject(
        source=hi, destination=hi_on_grid,
        src_transform=hi_transform, src_crs=cfg.geographic_crs,
        dst_transform=grid_transform, dst_crs=grid_crs,
        resampling=Resampling.bilinear, init_dest_nodata=False,
    )

    cfg.analysis_grid_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        cfg.analysis_grid_path, "w", driver="GTiff",
        height=height, width=width, count=2, dtype="float32",
        crs=grid_crs, transform=grid_transform, nodata=np.nan, compress="lzw",
    ) as dst:
        dst.write(hi_on_grid.astype("float32"), 1)
        dst.write(pop.astype("float32"), 2)
        dst.set_band_description(1, "heat_index")
        dst.set_band_description(2, "population")

    grid_meta = {"transform": grid_transform, "crs": grid_crs,
                 "height": height, "width": width}
    LOGGER.info("S6   analysis grid written to %s (%d x %d px)",
                cfg.analysis_grid_path, width, height)
    return hi_on_grid, grid_meta, pop


def population_weighted_heat(
    hi_grid: np.ndarray,
    pop_grid: np.ndarray,
    grid_meta: dict,
    buffers: gpd.GeoDataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    transform = grid_meta["transform"]
    numerator = np.full(len(buffers), np.nan, dtype="float64")
    unit_population = np.full(len(buffers), np.nan, dtype="float64")

    for row, geom in enumerate(buffers.geometry):
        win = window_for_bounds(
            geom.bounds, transform, grid_meta["width"], grid_meta["height"]
        )
        if win is None:
            continue
        r0, r1 = int(win.row_off), int(win.row_off + win.height)
        c0, c1 = int(win.col_off), int(win.col_off + win.width)
        hi_win = hi_grid[r0:r1, c0:c1]
        pop_win = pop_grid[r0:r1, c0:c1]

        inside = geometry_mask(
            [geom], hi_win.shape, window_transform(win, transform),
            invert=True, all_touched=True,
        )
        mask = inside & np.isfinite(hi_win) & np.isfinite(pop_win)
        if not mask.any():
            continue

        numerator[row] = float((pop_win[mask] * hi_win[mask]).sum())
        unit_population[row] = float(pop_win[mask].sum())

    return numerator, unit_population


def compute_whpge(
    cfg: Config,
    buffers: gpd.GeoDataFrame,
    greenspace_frac: np.ndarray,
    hi_grid: np.ndarray,
    pop_grid: np.ndarray,
    grid_meta: dict,
) -> pd.DataFrame:
    numerator, unit_population = population_weighted_heat(
        hi_grid, pop_grid, grid_meta, buffers
    )
    denominator = float(np.nansum(unit_population))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("the study-wide population denominator is zero")

    i_raw = numerator / denominator

    valid = np.isfinite(i_raw) & np.isfinite(greenspace_frac)
    i_prime = np.full(len(buffers), np.nan, dtype="float64")
    i_dprime = np.full(len(buffers), np.nan, dtype="float64")
    e_j = np.full(len(buffers), np.nan, dtype="float64")
    whpge = np.full(len(buffers), np.nan, dtype="float64")

    if valid.any():
        i_prime[valid] = minmax(i_raw[valid])
        i_dprime[valid] = np.exp(i_prime[valid] - 1.0)
        e_j[valid] = greenspace_frac[valid] * i_dprime[valid]
        whpge[valid] = minmax(e_j[valid])

    out = pd.DataFrame({
        "station_id": buffers["station_id"].to_numpy(),
        "heat_index_pop_weighted_sum": numerator,
        "unit_population": unit_population,
        "heat_index_pop_weighted": i_raw,
        "heat_index_norm": i_prime,
        "heat_index_exp": i_dprime,
        "greenspace_fraction": greenspace_frac,
        "E_j": e_j,
        "WHPGE": whpge,
    })
    LOGGER.info(
        "S7   WHPGE: mean = %.4f, range = [%.4f, %.4f], n valid = %d "
        "(denominator = %.6g)",
        float(np.nanmean(whpge)), float(np.nanmin(whpge)),
        float(np.nanmax(whpge)), int(np.isfinite(whpge).sum()), denominator,
    )
    return out


def compute_mspa_features(cfg: Config, buffers: gpd.GeoDataFrame) -> pd.DataFrame:
    if not cfg.mspa_raster_path.exists():
        raise FileNotFoundError(
            f"{cfg.mspa_raster_path} not found. Run GuidosToolbox first: feed it "
            f"{cfg.mspa_input_path} with an edge width of 1 pixel and save the "
            "7-class output to that path."
        )

    codes = sorted(cfg.mspa_class_codes)
    counts = zonal_class_counts(
        cfg.mspa_raster_path, list(buffers.geometry), codes, cfg.all_touched
    )
    area_km2 = counts.astype("float64") * pixel_area_km2(cfg.mspa_raster_path)

    out = pd.DataFrame({"station_id": buffers["station_id"].to_numpy()})
    for index, code in enumerate(codes):
        name = cfg.mspa_class_codes[code]
        out[f"MSPA_{name}_area_km2"] = area_km2[:, index]
        out[f"MSPA_{name}"] = minmax(area_km2[:, index])

    LOGGER.info("S8   MSPA areas extracted for %d classes on %d stations",
                len(codes), len(buffers))
    return out


def make_hexagon_grid(aoi: gpd.GeoDataFrame, cell_area_km2: float) -> gpd.GeoDataFrame:
    side = float(np.sqrt(2.0 * cell_area_km2 * 1e6 / (3.0 * np.sqrt(3.0))))
    minx, miny, maxx, maxy = aoi.total_bounds
    dx, dy = 1.5 * side, np.sqrt(3.0) * side
    angles = np.deg2rad(np.arange(0.0, 360.0, 60.0))

    n_cols = int(np.ceil((maxx - minx) / dx)) + 2
    n_rows = int(np.ceil((maxy - miny) / dy)) + 2
    if n_cols * n_rows > 2_000_000:
        raise ValueError("hexagonal tessellation would create too many cells")

    polygons: List[Polygon] = []
    for col in range(n_cols):
        cx = minx - dx + col * dx
        y_offset = (dy / 2.0) if col % 2 else 0.0
        for row in range(n_rows):
            cy = miny - dy + y_offset + row * dy
            polygons.append(
                Polygon([(cx + side * np.cos(a), cy + side * np.sin(a)) for a in angles])
            )

    grid = gpd.GeoDataFrame({"geometry": polygons}, crs=aoi.crs)
    grid = grid.loc[grid.intersects(dissolve_geometry(aoi))].reset_index(drop=True)
    grid["hex_id"] = [f"hex_{i:05d}" for i in range(len(grid))]
    return grid


def aggregate_to_hexagons(
    cfg: Config, grid: gpd.GeoDataFrame, points: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    joined = gpd.sjoin(
        points, grid[["hex_id", "geometry"]], how="inner", predicate="within"
    )
    joined = joined.drop(columns=[c for c in ("index_right",) if c in joined.columns])

    value_cols = [c for c in joined.columns
                  if c not in ("station_id", "geometry", "hex_id")]
    agg = joined.groupby("hex_id", as_index=False)[value_cols].mean()
    counts = joined.groupby("hex_id", as_index=False).size().rename(
        columns={"size": "n_stations"}
    )
    agg = agg.merge(counts, on="hex_id", how="left")
    agg = agg.loc[agg["n_stations"] >= cfg.min_stations_per_hexagon]

    out = grid.merge(agg, on="hex_id", how="inner")
    LOGGER.info("S9   %d stations mapped onto %d hexagonal units", len(points), len(out))
    return out

PREDICTORS = [
    "WHPGE",
    "MSPA_Core", "MSPA_Islet", "MSPA_Edge", "MSPA_Perforation",
    "MSPA_Bridge", "MSPA_Loop", "MSPA_Branch",
]


def assemble_model_table(hex_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    missing = [c for c in PREDICTORS if c not in hex_gdf.columns]
    if missing:
        raise KeyError(f"missing predictors in the hexagonal table: {missing}")

    out = hex_gdf.copy()
    for column in PREDICTORS:
        out[f"{column}_01"] = minmax(out[column].to_numpy())
    out["MOI_01"] = minmax(out["MOI"].to_numpy())
    return out


def write_outputs(
    cfg: Config, station_table: pd.DataFrame, hex_gdf: gpd.GeoDataFrame
) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    station_table.to_csv(cfg.out_dir / "moi_station_level.csv", index=False)
    hex_gdf.drop(columns="geometry").to_csv(
        cfg.out_dir / "predictors_hexagon.csv", index=False
    )
    hex_gdf.to_file(cfg.out_dir / "predictors_hexagon.gpkg", driver="GPKG")

    metadata = {
        "config": {k: str(v) for k, v in asdict(cfg).items()},
        "n_stations": int(len(station_table)),
        "n_hexagons": int(len(hex_gdf)),
        "predictors": PREDICTORS,
        "outputs": [
            "moi_station_level.csv",
            "predictors_hexagon.csv",
            "predictors_hexagon.gpkg",
        ],
    }
    with open(cfg.out_dir / "preprocessing_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False, default=str)
    LOGGER.info("S10  model-ready table written to %s", cfg.out_dir)


def run(cfg: Config) -> gpd.GeoDataFrame:
    for directory in (RAW_DIR, INTERIM_DIR, cfg.out_dir):
        directory.mkdir(parents=True, exist_ok=True)

    moi = compute_station_moi(cfg)
    stations = load_stations(cfg)
    stations = stations.merge(moi, on="station_id", how="inner")
    if stations.empty:
        raise ValueError("no monitoring station could be matched to an MOI record")
    buffers = build_buffers(stations, cfg.buffer_radius_m)

    build_binary_greenspace(cfg)
    g_frac = greenspace_fraction(cfg, buffers)

    hi_grid, grid_meta, pop_grid = build_analysis_grid(cfg)
    whpge = compute_whpge(cfg, buffers, g_frac, hi_grid, pop_grid, grid_meta)

    mspa = compute_mspa_features(cfg, buffers)

    station_table = (
        stations.drop(columns="geometry")
        .merge(whpge, on="station_id", how="left")
        .merge(mspa, on="station_id", how="left")
    )

    aoi = read_vector(cfg.study_area_path, cfg.geographic_crs).to_crs(cfg.metric_crs)
    grid = make_hexagon_grid(aoi, cfg.hexagon_area_km2)
    points = stations[["station_id", "geometry"]].merge(
        station_table.drop(columns=[c for c in ("geometry",) if c in station_table.columns]),
        on="station_id", how="inner",
    )
    points = gpd.GeoDataFrame(points, geometry="geometry", crs=cfg.metric_crs)
    hex_gdf = aggregate_to_hexagons(cfg, grid, points)
    hex_gdf = assemble_model_table(hex_gdf)

    write_outputs(cfg, station_table, hex_gdf)
    return hex_gdf


def parse_args(argv: Optional[Sequence[str]] = None) -> Config:
    parser = argparse.ArgumentParser(
        description="Data pre-processing for the Guangzhou MOI / greenspace "
                    "morphology / WHPGE study (Sections 2.2-2.3 of the manuscript)."
    )
    parser.add_argument("--buffer-radius", type=float, default=Config.buffer_radius_m,
                        help="station buffer radius in metres (default: 1000)")
    parser.add_argument("--hexagon-area", type=float, default=Config.hexagon_area_km2,
                        help="hexagonal cell area in km^2 (default: 5.0)")
    parser.add_argument("--period-start", type=str, default=Config.period_start)
    parser.add_argument("--period-end", type=str, default=Config.period_end)
    parser.add_argument("--hi-unit", choices=("F", "C"), default=Config.hi_unit,
                        help="unit of the Heat Index field (default: F)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    cfg = Config()
    cfg.buffer_radius_m = args.buffer_radius
    cfg.hexagon_area_km2 = args.hexagon_area
    cfg.period_start = args.period_start
    cfg.period_end = args.period_end
    cfg.hi_unit = args.hi_unit
    cfg.out_dir = args.out_dir
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    cfg = parse_args(argv)
    LOGGER.info("starting pre-processing (metric CRS %s)", cfg.metric_crs)
    hex_gdf = run(cfg)
    LOGGER.info("finished: %d hexagonal units, %d columns",
                len(hex_gdf), len(hex_gdf.columns))

if __name__ == "__main__":
    main()
