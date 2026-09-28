from pathlib import Path
import gc
import hashlib

import h3
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping
from shapely.ops import transform
from pyproj import Transformer


PERIODS = {
    "24_07": "h3_cl_24_07.parquet",
    "25_07": "h3_cl_25_07.parquet",
    "26_07": "h3_cl_26_07.parquet",
}

H3_RESOLUTION = 12

# Реальный буфер вокруг велодорожки.
# Буфер строится в метрах (EPSG:32637), а не через градусы.
BUFFER_M = 5

# geo_to_cells() возвращает H3, чьи центры попали в буфер.
# Добавляем соседей 1-го кольца, чтобы не потерять H3,
# которые геометрически пересекают буфер, но их центр оказался за границей.
NEIGHBOR_RING = 1

WGS84 = "EPSG:4326"
UTM37N = "EPSG:32637"

_TO_UTM = Transformer.from_crs(WGS84, UTM37N, always_xy=True).transform
_TO_WGS84 = Transformer.from_crs(UTM37N, WGS84, always_xy=True).transform


def _progress(cb, n, msg):
    if cb:
        cb(int(n), msg)


def _find_col(df, candidates):
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def _prepare_lanes(input_path):
    print("[1/6] Reading GeoJSON...", flush=True)

    g = gpd.read_file(input_path)

    if g.empty:
        raise ValueError("GeoJSON не содержит объектов.")

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

    g = g[g.geometry.notna() & ~g.geometry.is_empty].copy()
    g = g[g.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()

    if g.empty:
        raise ValueError("После очистки не осталось корректных линий.")

    # GeoJSON обычно WGS84. Если CRS отсутствует, считаем его WGS84.
    if g.crs is None:
        g = g.set_crs(WGS84)
    else:
        g = g.to_crs(WGS84)

    # Для линий buffer(0) обычно не нужен. Оставляем только валидные геометрии.
    invalid = ~g.geometry.is_valid
    if invalid.any():
        print(f"[1/6] Repairing invalid geometries: {int(invalid.sum())}", flush=True)
        repaired = g.loc[invalid, "geometry"].buffer(0)
        g.loc[invalid, "geometry"] = repaired

    g = g[
        g.geometry.notna()
        & ~g.geometry.is_empty
        & g.geometry.geom_type.isin(["LineString", "MultiLineString"])
    ].copy()

    if g.empty:
        raise ValueError("После исправления геометрии не осталось корректных линий.")

    g["velo_id"] = g.geometry.apply(
        lambda geom: hashlib.md5(geom.wkb).hexdigest()[:16]
    )
    g = g.drop_duplicates(subset=["velo_id"]).copy()

    g["name"] = g[name_col].fillna("").astype(str) if name_col else ""
    g["type"] = g[type_col].fillna("").astype(str) if type_col else ""
    g["status"] = g[status_col].fillna("").astype(str) if status_col else ""
    g["zone"] = g[zone_col].fillna("").astype(str) if zone_col else ""

    g = g[["velo_id", "name", "type", "status", "zone", "geometry"]].copy()

    print(f"[1/6] Lanes after cleanup: {len(g):,}", flush=True)
    return g


def _h3_cells_for_geometry(geometry):
    """
    Превращает одну велодорожку непосредственно в H3 ID.

    Важное отличие от старой версии:
    - не загружаем H3 dictionary;
    - не строим Polygon для 600k H3;
    - не делаем GeoPandas spatial join по справочнику.

    Буфер строится точно в метрах через EPSG:32637.
    Затем H3 определяет ячейки resolution 12.
    Соседи первого кольца добавляются для консервативного покрытия границы.
    """
    geom_utm = transform(_TO_UTM, geometry)
    buffered_utm = geom_utm.buffer(BUFFER_M)
    buffered_wgs84 = transform(_TO_WGS84, buffered_utm)

    if buffered_wgs84.is_empty:
        return set()

    geo = mapping(buffered_wgs84)

    # h3-py 4.x принимает GeoJSON-like geometry.
    base_cells = set(h3.geo_to_cells(geo, H3_RESOLUTION))

    if not base_cells:
        return set()

    if NEIGHBOR_RING <= 0:
        return base_cells

    cells = set()
    for cell in base_cells:
        cells.update(h3.grid_disk(cell, NEIGHBOR_RING))

    return cells


def _build_h3_intersections(lanes):
    """
    Создаёт связь:
        velo_id -> h3_id

    Используется только H3 как пространственный индекс.
    Никаких 600k WKT-полигонов здесь нет.
    """
    rows = []
    all_cells = set()

    total = len(lanes)

    for i, row in enumerate(lanes.itertuples(index=False), start=1):
        cells = _h3_cells_for_geometry(row.geometry)

        if cells:
            all_cells.update(cells)
            rows.extend((row.velo_id, cell) for cell in cells)

        if i == 1 or i % 25 == 0 or i == total:
            print(
                f"[2/6] H3 index: {i:,}/{total:,} lanes; "
                f"unique H3={len(all_cells):,}; "
                f"links={len(rows):,}",
                flush=True,
            )

    if not rows:
        return pd.DataFrame(columns=["velo_id", "h3_id"])

    result = pd.DataFrame(rows, columns=["velo_id", "h3_id"])
    result = result.drop_duplicates(ignore_index=True)

    print(
        f"[2/6] H3 links: {len(result):,}; "
        f"unique H3: {result['h3_id'].nunique():,}",
        flush=True,
    )

    return result


def _read_h3_period(path, needed_h3):
    """
    Читает только нужные колонки одного периода и сразу оставляет
    только H3, связанные с велодорожками.
    """
    # Сначала читаем только заголовок через PyArrow/Parquet metadata.
    try:
        import pyarrow.parquet as pq

        schema = pq.ParquetFile(path).schema_arrow
        names = schema.names

        h3_col = _find_col(pd.DataFrame(columns=names), [
            "h3_id", "h3", "hexagon", "index"
        ])
        trip_col = _find_col(pd.DataFrame(columns=names), [
            "sum_trip", "trips", "trip_count", "count"
        ])
        class_col = _find_col(pd.DataFrame(columns=names), [
            "trip_class", "class", "cluster"
        ])
    except Exception as e:
        raise RuntimeError(f"Не удалось прочитать схему Parquet {path}: {e}")

    if not h3_col or not trip_col or not class_col:
        raise ValueError(
            f"Не удалось определить колонки H3 в {path.name}. "
            f"Колонки файла: {names}"
        )

    df = pd.read_parquet(
        path,
        columns=[h3_col, trip_col, class_col],
        engine="pyarrow",
    )
    df.columns = ["h3_id", "sum_trip", "trip_class"]

    df["h3_id"] = df["h3_id"].astype(str)
    df["sum_trip"] = pd.to_numeric(df["sum_trip"], errors="coerce")
    df["trip_class"] = pd.to_numeric(df["trip_class"], errors="coerce")

    df = df.dropna(subset=["h3_id", "sum_trip"])

    # Самое важное сокращение перед дальнейшим расчётом.
    df = df[df["h3_id"].isin(needed_h3)].copy()

    if df.empty:
        return df

    # На случай дублей одного H3 в исходном Parquet.
    df = (
        df.groupby("h3_id", as_index=False)
        .agg(
            sum_trip=("sum_trip", "sum"),
            trip_class=("trip_class", "mean"),
        )
    )

    return df


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

    g["cv"] = np.where(
        g["avg_trips"] > 0,
        g["std_trips"] / g["avg_trips"],
        0,
    )

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


def _write_outputs(lanes, all_periods, job_dir):
    print("[6/6] Writing results...", flush=True)

    meta = lanes[["velo_id", "name", "type", "status", "zone"]]

    # CSV по каждому периоду.
    summary = meta.merge(all_periods, on="velo_id", how="left")

    for period in PERIODS:
        p = summary[summary["period"] == period].copy()

        p.to_csv(
            job_dir / f"velo_demand_{period}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # Все периоды одной таблицей.
    all_periods.to_csv(
        job_dir / "velo_demand_all_periods.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Итоговый score по трём периодам.
    pivot = all_periods.pivot_table(
        index="velo_id",
        columns="period",
        values="demand_score",
        aggfunc="first",
    ).reset_index()

    for period in PERIODS:
        if period not in pivot.columns:
            pivot[period] = np.nan

    pivot["score_3_periods"] = pivot[list(PERIODS)].mean(axis=1)

    summary_csv = meta.merge(pivot, on="velo_id", how="left")

    summary_csv.to_csv(
        job_dir / "velo_demand_summary_3_periods.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Геометрия остаётся геометрией исходных велодорожек.
    geo = lanes.merge(
        summary_csv,
        on=["velo_id", "name", "type", "status", "zone"],
        how="left",
    )

    geo = geo.replace([np.inf, -np.inf], np.nan)

    geo.to_file(
        job_dir / "result.geojson",
        driver="GeoJSON",
    )

    return summary_csv


def process_job(input_path, job_dir, data_dir, progress=None):
    """
    Основной pipeline.

    Архитектура:
        GeoJSON велодорожек
          -> H3 IDs напрямую
          -> join с тремя Parquet
          -> demand score
          -> CSV + GeoJSON

    H3 dictionary / WKT не используется во время пользовательского расчёта.
    """
    job_dir = Path(job_dir)
    data_dir = Path(data_dir)

    job_dir.mkdir(parents=True, exist_ok=True)

    print("=== VELO DEMAND: H3 ID MODE ===", flush=True)

    # ------------------------------------------------------------
    # 1. Велодорожки
    # ------------------------------------------------------------
    _progress(progress, 5, "Читаем GeoJSON")
    lanes = _prepare_lanes(input_path)

    _progress(
        progress,
        15,
        f"Подготовлено велодорожек: {len(lanes):,}",
    )

    # ------------------------------------------------------------
    # 2. H3 IDs
    # ------------------------------------------------------------
    _progress(progress, 20, "Получаем H3 ID для велодорожек")
    print("[2/6] Generating H3 IDs directly from geometry...", flush=True)

    intersections = _build_h3_intersections(lanes)

    unique_h3 = (
        intersections["h3_id"].nunique()
        if not intersections.empty
        else 0
    )

    print(
        f"[2/6] Unique H3 candidates: {unique_h3:,}",
        flush=True,
    )

    _progress(
        progress,
        45,
        f"Получено H3: {unique_h3:,}",
    )

    if intersections.empty:
        raise ValueError(
            "Не найдено ни одного H3 для велодорожек. "
            "Проверьте CRS и геометрию GeoJSON."
        )

    # Здесь intersections уже является готовой связью velo_id -> h3_id.
    # Никакого второго spatial join не требуется.
    needed_h3 = set(intersections["h3_id"].astype(str))

    print(
        f"[2/6] H3 links ready: {len(intersections):,}",
        flush=True,
    )

    # ------------------------------------------------------------
    # 3-5. Периоды
    # ------------------------------------------------------------
    period_results = []

    for i, (period, filename) in enumerate(PERIODS.items()):
        progress_value = 50 + i * 14

        _progress(
            progress,
            progress_value,
            f"Считаем {period}",
        )

        path = data_dir / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Не найден файл H3: {path}"
            )

        print(
            f"[{period}] Reading relevant H3 rows...",
            flush=True,
        )

        h3_data = _read_h3_period(
            path,
            needed_h3=needed_h3,
        )

        print(
            f"[{period}] Relevant H3 rows: {len(h3_data):,}",
            flush=True,
        )

        r = _score_period(
            intersections,
            h3_data,
            period,
        )

        if not r.empty:
            period_results.append(r)

        del h3_data
        del r
        gc.collect()

    if not period_results:
        raise ValueError(
            "Не найдено данных H3 для велодорожек "
            "ни в одном из трёх периодов."
        )

    all_periods = pd.concat(
        period_results,
        ignore_index=True,
    )

    # ------------------------------------------------------------
    # 6. Результаты
    # ------------------------------------------------------------
    _progress(
        progress,
        95,
        "Формируем CSV и GeoJSON",
    )

    summary_csv = _write_outputs(
        lanes,
        all_periods,
        job_dir,
    )

    _progress(
        progress,
        100,
        "Результат готов",
    )

    print("=== DONE ===", flush=True)

    return {
        "lanes": int(len(lanes)),
        "h3_cells": int(unique_h3),
        "h3_candidates": int(unique_h3),
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
