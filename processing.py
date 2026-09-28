from pathlib import Path
import hashlib
import json
import gc
import os

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
import h3


PERIODS = {
    "24_07": "h3_cl_24_07.parquet",
    "25_07": "h3_cl_25_07.parquet",
    "26_07": "h3_cl_26_07.parquet",
}


# Cache is process-local. With one Render worker this means the expensive
# H3 dictionary is loaded only once and is reused by subsequent uploads.
_H3_CACHE = None


def _progress(cb, n, msg):
    if cb:
        cb(n, msg)


def _find_col(df, candidates):
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def _load_h3_lookup(data_dir):
    global _H3_CACHE
    if _H3_CACHE is not None:
        return _H3_CACHE

    # Preferred format: GeoParquet prepared once during Render build.
    geo_path = Path(data_dir) / "h3_lookup.parquet"
    if geo_path.exists():
        g = gpd.read_parquet(geo_path, columns=["h3_id", "geometry"])
        if g.crs is None:
            g = g.set_crs(4326)
        elif g.crs.to_epsg() != 4326:
            g = g.to_crs(4326)
        _H3_CACHE = g
        return g

    # Fallback for local testing: original WKT parquet.
    src = Path(data_dir) / "h3_hexagons_12_600_street_tr.parquet"
    if not src.exists():
        raise FileNotFoundError(
            f"Не найден H3 dictionary: {src}. "
            "Положите его в data/ или подготовьте h3_lookup.parquet."
        )

    df = pd.read_parquet(src, columns=["h3_id", "wkt"])
    g = gpd.GeoDataFrame(
        df[["h3_id"]].copy(),
        geometry=gpd.GeoSeries.from_wkt(df["wkt"]),
        crs=4326,
    )
    _H3_CACHE = g
    return g


def _read_h3_period(path):
    df = pd.read_parquet(path)

    h3_col = _find_col(df, ["h3_id", "h3", "hexagon", "index"])
    trip_col = _find_col(df, ["sum_trip", "trips", "trip_count", "count"])
    class_col = _find_col(df, ["trip_class", "class", "cluster"])

    if not h3_col or not trip_col or not class_col:
        raise ValueError(
            f"Не удалось определить колонки H3 в {path.name}. "
            f"Найдены: {list(df.columns)}"
        )

    out = df[[h3_col, trip_col, class_col]].copy()
    out.columns = ["h3_id", "sum_trip", "trip_class"]
    out["sum_trip"] = pd.to_numeric(out["sum_trip"], errors="coerce")
    out["trip_class"] = pd.to_numeric(out["trip_class"], errors="coerce")
    out = out.dropna(subset=["h3_id", "sum_trip"])
    out["h3_id"] = out["h3_id"].astype(str)

    # If a source has duplicate H3 rows, consolidate them before joining.
    out = (
        out.groupby("h3_id", as_index=False)
        .agg(sum_trip=("sum_trip", "sum"),
             trip_class=("trip_class", "mean"))
    )
    return out


def _prepare_lanes(input_path):
    g = gpd.read_file(input_path)

    if g.empty:
        raise ValueError("GeoJSON не содержит объектов.")

    # Normalize common source field names.
    cols = {str(c).lower(): c for c in g.columns}

    def pick(*names):
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    name_col = pick("name", "NAME")
    type_col = pick("type", "TYPE")
    status_col = pick("status", "STATUS")
    zone_col = pick("zone", "ZONE")

    g = g[g.geometry.notna() & ~g.geometry.is_empty & g.geometry.is_valid].copy()
    g = g[g.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()

    if g.empty:
        raise ValueError("После очистки не осталось корректных линий.")

    g = g.to_crs(4326)

    g["velo_id"] = g.geometry.apply(
        lambda geom: hashlib.md5(geom.wkb).hexdigest()[:16]
    )
    g = g.drop_duplicates(subset=["velo_id"]).copy()

    g["name"] = g[name_col].fillna("").astype(str) if name_col else ""
    g["type"] = g[type_col].fillna("").astype(str) if type_col else ""
    g["status"] = g[status_col].fillna("").astype(str) if status_col else ""
    g["zone"] = g[zone_col].fillna("").astype(str) if zone_col else ""

    return g[["velo_id", "name", "type", "status", "zone", "geometry"]]


def _candidate_h3(h3_lookup, lanes):
    # Important Render optimization:
    # do not spatial-join against all 600k cells when the uploaded file
    # occupies only a fraction of the dictionary extent.
    lanes_utm = lanes.to_crs(32637)
    buffered = lanes_utm.copy()
    buffered["geometry"] = buffered.geometry.buffer(5)

    minx, miny, maxx, maxy = buffered.total_bounds

    h3_utm = h3_lookup.to_crs(32637)
    s = h3_utm.sindex
    idx = list(s.intersection((minx, miny, maxx, maxy)))
    cand = h3_utm.iloc[idx].copy()

    return buffered, cand


def _make_intersections(lanes_buffered, h3_candidates):
    # One spatial join for all uploaded lanes.
    joined = gpd.sjoin(
        lanes_buffered[["velo_id", "geometry"]],
        h3_candidates[["h3_id", "geometry"]],
        how="inner",
        predicate="intersects",
    )
    return joined[["velo_id", "h3_id"]].drop_duplicates()


def _percent_rank(s):
    n = len(s)
    if n <= 1:
        return pd.Series(np.ones(n), index=s.index)
    return (s.rank(method="min") - 1) / (n - 1)


def _score_period(intersections, h3_data, period):
    x = intersections.merge(h3_data, on="h3_id", how="inner")
    if x.empty:
        return pd.DataFrame()

    g = (
        x.groupby("velo_id", as_index=False)
        .agg(
            h3_count=("h3_id", "nunique"),
            total_trips_reference=("sum_trip", "sum"),
            avg_trips=("sum_trip", "mean"),
            median_trips=("sum_trip", "median"),
            max_trips=("sum_trip", "max"),
            avg_class=("trip_class", "mean"),
            std_trips=("sum_trip", "std"),
        )
    )

    g["std_trips"] = g["std_trips"].fillna(0)
    g["cv"] = np.where(g["avg_trips"] > 0, g["std_trips"] / g["avg_trips"], 0)
    g["homogeneity"] = 1 - np.minimum(g["cv"], 1)

    g["rank_avg"] = _percent_rank(g["avg_trips"])
    g["rank_median"] = _percent_rank(g["median_trips"])
    g["rank_max"] = _percent_rank(g["max_trips"])
    g["rank_class"] = _percent_rank(g["avg_class"].fillna(0))

    g["demand_score"] = 100 * (
        0.35 * g["rank_avg"]
        + 0.25 * g["rank_median"]
        + 0.15 * g["rank_max"]
        + 0.15 * g["rank_class"]
        + 0.10 * g["homogeneity"]
    )

    g["period"] = period
    return g


def process_job(input_path, job_dir, data_dir, progress=None):
    job_dir = Path(job_dir)
    data_dir = Path(data_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    _progress(progress, 5, "Читаем GeoJSON")
    lanes = _prepare_lanes(input_path)

    _progress(progress, 15, f"Подготовлено велодорожек: {len(lanes):,}")

    _progress(progress, 20, "Загружаем H3-слой")
    h3_lookup = _load_h3_lookup(data_dir)

    _progress(progress, 30, "Отбираем H3-кандидатов по bbox")
    lanes_buffered, candidates = _candidate_h3(h3_lookup, lanes)

    _progress(progress, 40, f"H3-кандидатов: {len(candidates):,}")

    intersections = _make_intersections(lanes_buffered, candidates)
    _progress(progress, 48, f"Получено пересечений: {len(intersections):,}")

    del lanes_buffered, candidates
    gc.collect()

    period_results = []
    for i, (period, filename) in enumerate(PERIODS.items()):
        _progress(progress, 50 + i * 14, f"Считаем {period}")

        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Не найден файл H3: {path}")

        h3_data = _read_h3_period(path)
        r = _score_period(intersections, h3_data, period)

        if not r.empty:
            period_results.append(r)

        del h3_data, r
        gc.collect()

    if not period_results:
        raise ValueError("Не найдено ни одного пересечения велодорог с H3.")

    all_periods = pd.concat(period_results, ignore_index=True)

    # Attach source attributes and keep geometry.
    summary = lanes[["velo_id", "name", "type", "status", "zone"]].merge(
        all_periods,
        on="velo_id",
        how="left",
    )

    # Period CSVs.
    for period in PERIODS:
        p = summary[summary["period"] == period].copy()
        p.to_csv(job_dir / f"velo_demand_{period}.csv", index=False, encoding="utf-8-sig")

    all_periods.to_csv(
        job_dir / "velo_demand_all_periods.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pivot = all_periods.pivot_table(
        index="velo_id",
        columns="period",
        values="demand_score",
        aggfunc="first",
    ).reset_index()

    for p in PERIODS:
        if p not in pivot.columns:
            pivot[p] = np.nan

    pivot["score_3_periods"] = pivot[list(PERIODS)].mean(axis=1)

    meta = lanes[["velo_id", "name", "type", "status", "zone"]]
    summary_csv = meta.merge(pivot, on="velo_id", how="left")
    summary_csv.to_csv(
        job_dir / "velo_demand_summary_3_periods.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # GeoJSON: one feature per unique uploaded geometry with all period scores.
    geo = lanes.merge(summary_csv, on=["velo_id", "name", "type", "status", "zone"], how="left")
    geo = geo.replace({np.nan: None})

    # Avoid NaN/Inf in JSON.
    numeric_cols = [
        c for c in geo.columns
        if c not in {"velo_id", "name", "type", "status", "zone", "geometry"}
    ]
    for c in numeric_cols:
        geo[c] = pd.to_numeric(geo[c], errors="coerce")

    geo = geo.replace([np.inf, -np.inf], np.nan)
    geo.to_file(job_dir / "result.geojson", driver="GeoJSON")

    _progress(progress, 100, "Результат готов")

    return {
        "lanes": int(len(lanes)),
        "h3_cells": int(len(h3_lookup)),
        "h3_candidates": int(len(set(intersections["h3_id"]))),
        "intersections": int(len(intersections)),
        "periods": list(PERIODS.keys()),
        "downloads": {
            "geojson": f"/api/download/{job_dir.name}/geojson",
            "summary": f"/api/download/{job_dir.name}/summary",
            "all": f"/api/download/{job_dir.name}/all",
            "24_07": f"/api/download/{job_dir.name}/24_07",
            "25_07": f"/api/download/{job_dir.name}/25_07",
            "26_07": f"/api/download/{job_dir.name}/26_07",
        },
    }
