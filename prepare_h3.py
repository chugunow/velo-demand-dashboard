from pathlib import Path
import pandas as pd
import geopandas as gpd

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
SRC = DATA / "h3_hexagons_12_600_street_tr.parquet"
DST = DATA / "h3_lookup.parquet"

if DST.exists():
    print(f"{DST} already exists")
    raise SystemExit(0)

print("Reading H3 dictionary...")
df = pd.read_parquet(SRC, columns=["h3_id", "wkt"])

print("Building geometry...")
g = gpd.GeoDataFrame(
    df[["h3_id"]].copy(),
    geometry=gpd.GeoSeries.from_wkt(df["wkt"]),
    crs=4326,
)

print("Writing GeoParquet...")
g.to_parquet(DST, index=False, compression="zstd")
print(f"Created {DST}")
