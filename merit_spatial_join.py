import geopandas as gpd
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from shapely.geometry import Point
import time
import os

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
COMID_PATH     = 'E:/CANADA/merit.gpkg'
AREAPERIL_PATH = 'E:/CANADA/areaperil_geom_parquet/geometry.parquet'
OUTPUT_PATH    = 'E:/CANADA/joined_2/areaperil_MERIT_joined.parquet'
BATCH_SIZE     = 5_000_000  # tune down to 2_000_000 if RAM is tight

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print('=' * 60)
print('STEP 1 — Loading input data')
print('=' * 60)

t0 = time.time()
print(' Loading comids from GeoPackage...')
comid_gdf = gpd.read_file(COMID_PATH)
print(f' ✓ Comids loaded: {len(comid_gdf):,} features ({time.time()-t0:.1f}s)')

t1 = time.time()
print(' Loading areaperils from Parquet...')
df = pd.read_parquet(AREAPERIL_PATH)
df.rename(columns={'x': 'longitude', 'y': 'latitude'}, inplace=True)
print(f' ✓ Areaperils loaded: {len(df):,} rows ({time.time()-t1:.1f}s)')

# ══════════════════════════════════════════════════════════════════════════════
# PRE-PROJECT COMID ONCE (outside loop — critical for performance & correctness)
# ══════════════════════════════════════════════════════════════════════════════
print()
print('=' * 60)
print('STEP 2 — Pre-projecting comid layer')
print('=' * 60)

sample_point = gpd.GeoDataFrame(
    geometry=[Point(df['longitude'].iloc[0], df['latitude'].iloc[0])],
    crs='EPSG:4326'
)
target_crs = sample_point.estimate_utm_crs()
comid_projected = comid_gdf.to_crs(target_crs)
print(f' ✓ Comid projected to: {target_crs}')

# ══════════════════════════════════════════════════════════════════════════════
# BATCH SPATIAL JOIN WITH PYARROW STREAMING WRITER
# ══════════════════════════════════════════════════════════════════════════════
print()
print('=' * 60)
print('STEP 3 — Batch spatial join')
print('=' * 60)

total_rows = len(df)
num_batches = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE
print(f' Total rows : {total_rows:,}')
print(f' Batch size : {BATCH_SIZE:,}')
print(f' Num batches: {num_batches}')

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

writer = None
total_written = 0

for batch_num in range(num_batches):
    t_batch = time.time()
    start_idx = batch_num * BATCH_SIZE
    end_idx   = min(start_idx + BATCH_SIZE, total_rows)

    print(f'\n Batch {batch_num+1}/{num_batches} (rows {start_idx:,} → {end_idx:,})')

    # ── Extract & build GeoDataFrame ──────────────────────────────────────────
    batch_df = df.iloc[start_idx:end_idx].copy()
    geometry = [Point(xy) for xy in zip(batch_df['longitude'], batch_df['latitude'])]
    areaperil_gdf = gpd.GeoDataFrame(batch_df, geometry=geometry, crs='EPSG:4326')
    areaperil_gdf = areaperil_gdf.to_crs(target_crs)
    print(f' - GeoDataFrame built')

    # ── Spatial join (nearest) ────────────────────────────────────────────────
    join_gdf = gpd.sjoin_nearest(
        areaperil_gdf,
        comid_projected,
        how='left',
        distance_col='dist'
    )
    join_gdf.drop(columns='geometry', inplace=True)
    print(f' - Spatial join done: {len(join_gdf):,} rows')

    # ── Deduplicate: one nearest MERIT section per areaperil_id ───────────────
    before = len(join_gdf)
    join_gdf = join_gdf.sort_values('dist').drop_duplicates(subset='areaperil_id', keep='first')
    print(f' - Deduplicated: {before:,} → {len(join_gdf):,} rows ({before - len(join_gdf):,} duplicates removed)')

    # ── Stream-append via PyArrow (NO read-back, NO concat) ───────────────────
    table = pa.Table.from_pandas(join_gdf, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(OUTPUT_PATH, table.schema)
    writer.write_table(table)

    total_written += len(join_gdf)
    elapsed = time.time() - t_batch
    print(f' - Saved ✓ ({elapsed:.1f}s, total written: {total_written:,})')

if writer:
    writer.close()

print()
print('=' * 60)
print(f'ALL DONE — {total_written:,} rows written to:')
print(f' {OUTPUT_PATH}')
print('=' * 60)
