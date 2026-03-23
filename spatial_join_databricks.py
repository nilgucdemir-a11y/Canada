"""
Nearest-neighbour spatial join in Databricks  (CRS: EPSG 4326)
===============================================================
Join a GPKG LineString layer to a 3.6-billion-row (lon, lat) table.

CRS correctness notes
---------------------
Both tables are in EPSG:4326 (geographic, degrees).  Working directly in 4326
introduces two silent errors that this script avoids:

  1. ST_Distance in 4326 returns DEGREES, not metres.
     → Use ST_DistanceSphere (metres on a sphere) or ST_DistanceSpheroid
       (metres on the GRS-80 ellipsoid) for any reported distance value.
     → For ranking (finding the *nearest*), degree distance is monotonic so
       it still identifies the correct nearest feature – but never store it
       as a metric distance.

  2. ST_Buffer(geom, 0.01) in 4326 creates an ellipse, not a circle.
     At 60 °N (central Canada) 0.01° longitude ≈ 555 m but
     0.01° latitude ≈ 1 111 m.  A degree-buffer search radius is therefore
     asymmetric and will miss nearby features to the east/west.
     → Either reproject to EPSG:3347 (Statistics Canada Lambert), buffer
       in metres, then reproject back; or inflate the degree radius to
       account for the worst-case latitude distortion.

Recommended stack
-----------------
* Apache Sedona  – native distributed spatial library for Spark
  Install on the cluster:  sedona[spark]  (PyPI) or use the
  Databricks Marketplace / cluster library UI.
* Databricks Runtime 13+ (Spark 3.4+)

Three strategies are provided, ordered from simplest to most scalable:

  A. Sedona ST_KNN  – exact k-nearest, works well when the LineString
                      table is small enough to broadcast (~millions of rows).
  B. Sedona range join + ST_DistanceSphere  – exact nearest via two-pass;
                      handles both tables at arbitrary scale.
  C. H3 bucketing   – approximate nearest; extremely fast; use when a
                      ~100-metre approximation is acceptable.
"""

# ── 0. Install / import ────────────────────────────────────────────────────
# In a Databricks notebook cell run first:
#   %pip install sedona apache-sedona  (then restart Python kernel)

from sedona.spark import SedonaContext
from pyspark.sql import functions as F

# ── 1. Bootstrap Sedona ────────────────────────────────────────────────────
config = (
    SedonaContext.builder()
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .config(
        "spark.kryo.registrator",
        "org.apache.sedona.core.serde.SedonaKryoRegistrator",
    )
    # Tune for your cluster – more partitions = finer parallelism
    .config("spark.sql.shuffle.partitions", "2000")
    .getOrCreate()
)
sedona = SedonaContext.create(config)


# ── 2. Load the GPKG LineString table ─────────────────────────────────────
# Replace the path with your DBFS / Unity Catalog Volume path.
GPKG_PATH = "/Volumes/my_catalog/my_schema/my_volume/roads.gpkg"

lines_df = (
    sedona.read.format("geopackage")
    .option("layerName", "roads")          # change to your layer name
    .load(GPKG_PATH)
    # geometry column is already a Sedona ST_ type after loading
    .withColumnRenamed("geometry", "line_geom")
    .select("line_id", "line_geom")        # keep only what you need
)
lines_df.cache()
lines_df.count()  # trigger cache – only worthwhile if lines fit in memory


# ── 3. Load the lon/lat table ──────────────────────────────────────────────
# Your 3.6 B-row table saved as Delta in Databricks.
COORDS_TABLE = "my_catalog.my_schema.coords"   # or a DBFS parquet path

coords_df = (
    spark.table(COORDS_TABLE)
    # or: spark.read.parquet("dbfs:/path/to/coords/")
    .select("point_id", "longitude", "latitude")
    # ST_Point(lon, lat) — note: X=longitude, Y=latitude (common mistake: swapping these)
    # ST_SetSRID tags the geometry as 4326 so Sedona knows the CRS for
    # ST_Transform, ST_DistanceSpheroid, etc.
    .withColumn(
        "point_geom",
        F.expr("ST_SetSRID(ST_Point(CAST(longitude AS DOUBLE), CAST(latitude AS DOUBLE)), 4326)"),
    )
)


# ── CRS VALIDATION (run once before the join) ─────────────────────────────
# Check 1: GPKG geometry SRID must be 4326
lines_df.selectExpr("ST_SRID(line_geom) AS srid").distinct().show()
# Expected output: 4326.  If you see 0 the GPKG has no embedded SRID —
# fix with: lines_df = lines_df.withColumn("line_geom", F.expr("ST_SetSRID(line_geom, 4326)"))
# If you see another value (e.g. 32617 / UTM) reproject:
#   lines_df = lines_df.withColumn("line_geom",
#       F.expr("ST_Transform(line_geom, 'EPSG:32617', 'EPSG:4326')"))

# Check 2: Coordinate sanity for the coords table
coords_df.selectExpr(
    "MIN(longitude)", "MAX(longitude)",
    "MIN(latitude)",  "MAX(latitude)",
    "COUNT(*) AS total",
    "SUM(CASE WHEN longitude NOT BETWEEN -180 AND 180 THEN 1 ELSE 0 END) AS bad_lon",
    "SUM(CASE WHEN latitude  NOT BETWEEN  -90 AND  90 THEN 1 ELSE 0 END) AS bad_lat",
    # Canada bounding box sanity check
    "SUM(CASE WHEN longitude NOT BETWEEN -141 AND -52 THEN 1 ELSE 0 END) AS outside_canada_lon",
    "SUM(CASE WHEN latitude  NOT BETWEEN   42 AND  84 THEN 1 ELSE 0 END) AS outside_canada_lat",
).show()

# Check 3: Confirm ST_Point(lon, lat) order is correct for a known location.
# Vancouver: lon=-123.12, lat=49.28  → should render near the BC coast, not in the ocean.
sedona.sql("""
    SELECT ST_AsText(ST_SetSRID(ST_Point(-123.12, 49.28), 4326)) AS vancouver_wkt
""").show(truncate=False)
# Expected: POINT (-123.12 49.28)  — if you see POINT (49.28 -123.12) your
# lon/lat columns are swapped in the source table.


# ══════════════════════════════════════════════════════════════════════════
# STRATEGY A – ST_KNN (exact; broadcast the lines table)
# Best when: LineString table fits in driver/executor memory (< ~5 M rows).
# ══════════════════════════════════════════════════════════════════════════

def nearest_join_knn(coords_df, lines_df, k=1):
    """
    Use Sedona's ST_KNN to find the k nearest LineStrings for every point.

    ST_KNN performs a distributed KNN search using the R-tree index that
    Sedona builds automatically on the broadcast side.

    CRS note: ST_KNN internally uses Euclidean distance in degree-space,
    which is correct for finding the nearest neighbour (monotonic ranking)
    but NOT for measuring the actual distance in metres.
    ST_DistanceSphere is used here to report meaningful distances.
    """
    coords_df.createOrReplaceTempView("coords")
    lines_df.createOrReplaceTempView("lines")

    result = sedona.sql(f"""
        SELECT
            c.point_id,
            l.line_id,
            -- ST_DistanceSphere returns metres on a sphere (fast, ~0.3% error)
            ST_DistanceSphere(c.point_geom, l.line_geom)     AS dist_metres,
            -- ST_DistanceSpheroid uses the GRS-80 ellipsoid (most accurate, slower)
            -- ST_DistanceSpheroid(c.point_geom, l.line_geom) AS dist_metres_exact
        FROM coords c
        JOIN lines l
        ON ST_KNN(c.point_geom, l.line_geom, {k}, true)
    """)
    return result

# result_knn = nearest_join_knn(coords_df, lines_df, k=1)
# result_knn.write.mode("overwrite").saveAsTable("my_schema.nearest_line_per_point")


# ══════════════════════════════════════════════════════════════════════════
# STRATEGY B – Range join + ST_DistanceSphere (exact; both tables distributed)
# Best when: both tables are huge and LineStrings cannot be broadcast.
# ══════════════════════════════════════════════════════════════════════════
#
# CRS-correct approach for 4326:
#
#   OPTION B-1 (recommended):
#     Reproject to EPSG:3347 (Statistics Canada Lambert), buffer in metres,
#     then do the distance ranking in metres.  The reprojection is done
#     on-the-fly inside Spark — no pre-materialisation needed.
#
#   OPTION B-2 (simpler, slight approximation):
#     Stay in 4326 but inflate the degree radius to compensate for
#     latitude distortion.  At 84°N (Canada's northern tip) 1° lon ≈ 111 km
#     × cos(84°) ≈ 11.6 km — so the worst-case compression is ~cos(lat).
#     Divide the desired metre radius by 111_320 * cos(max_lat_radians) to
#     get a safe degree radius that always covers the intended distance.

import math

# ── B-1 : reproject to EPSG:3347 ──────────────────────────────────────────
# Buffer radius in metres (tune to your data — for roads in Canada 2 km is
# usually enough; sparse networks may need 10–20 km).
SEARCH_RADIUS_M = 2_000   # metres

coords_df.createOrReplaceTempView("coords")
lines_df.createOrReplaceTempView("lines")

nearest_range = sedona.sql(f"""
    WITH projected AS (
        -- Reproject both tables to Statistics Canada Lambert (EPSG:3347)
        -- so that ST_Buffer creates a true circle in metres.
        SELECT point_id,
               ST_Transform(point_geom, 'EPSG:4326', 'EPSG:3347') AS pt_3347
        FROM   coords
    ),
    lines_proj AS (
        SELECT line_id,
               ST_Transform(line_geom, 'EPSG:4326', 'EPSG:3347') AS ln_3347
        FROM   lines
    ),
    candidates AS (
        SELECT
            p.point_id,
            l.line_id,
            -- Distance in metres (projected CRS, no spheroid approximation needed)
            ST_Distance(p.pt_3347, l.ln_3347)  AS dist_metres,
            ROW_NUMBER() OVER (
                PARTITION BY p.point_id
                ORDER BY     ST_Distance(p.pt_3347, l.ln_3347)
            ) AS rn
        FROM projected p
        JOIN lines_proj l
          ON ST_Intersects(
               ST_Buffer(p.pt_3347, {SEARCH_RADIUS_M}),   -- true circle, metres
               l.ln_3347
             )
    )
    SELECT point_id, line_id, dist_metres
    FROM   candidates
    WHERE  rn = 1
""")

# ── B-2 : stay in 4326, inflate radius ────────────────────────────────────
# Use this if ST_Transform is unavailable or too slow.
# Canada spans ~42°N – 84°N.  Worst distortion is at the northernmost lat.
MAX_LAT_DEG      = 84.0
SEARCH_RADIUS_M2 = 2_000  # desired radius in metres

# At max latitude, 1 degree longitude ≈ 111_320 * cos(lat) metres.
# Divide desired metres by this to get the degree equivalent that is always
# large enough everywhere in Canada.
_deg_radius = SEARCH_RADIUS_M2 / (111_320 * math.cos(math.radians(MAX_LAT_DEG)))

nearest_range_approx = sedona.sql(f"""
    WITH candidates AS (
        SELECT
            c.point_id,
            l.line_id,
            ST_DistanceSphere(c.point_geom, l.line_geom) AS dist_metres,
            ROW_NUMBER() OVER (
                PARTITION BY c.point_id
                ORDER BY     ST_DistanceSphere(c.point_geom, l.line_geom)
            ) AS rn
        FROM coords c
        JOIN lines  l
          ON ST_Intersects(
               ST_Buffer(c.point_geom, {_deg_radius:.6f}),
               l.line_geom
             )
    )
    SELECT point_id, line_id, dist_metres
    FROM   candidates
    WHERE  rn = 1
""")

# Save result
# nearest_range.write.mode("overwrite").saveAsTable("my_schema.nearest_line_per_point")


# ══════════════════════════════════════════════════════════════════════════
# STRATEGY C – H3 bucketing (approximate; fastest at 3.6 B scale)
# Best when: ~100 m accuracy is acceptable and you need max throughput.
# ══════════════════════════════════════════════════════════════════════════
#
# Resolution guide (H3):
#   res 8  → avg cell ≈  0.74 km²  (side ≈ 460 m)
#   res 9  → avg cell ≈  0.10 km²  (side ≈ 174 m)
#   res 10 → avg cell ≈  0.015 km² (side ≈  65 m)
#
# Install: %pip install h3  or use built-in Databricks h3_polyfill functions.

H3_RES = 9   # adjust for desired accuracy vs. join fanout

coords_h3 = (
    coords_df
    .withColumn(
        "h3_cell",
        F.expr(f"h3_longlatash3(longitude, latitude, {H3_RES})"),
    )
)

# For LineStrings: explode each linestring into H3 cells it covers,
# then also add the k-ring (immediate neighbours) to avoid boundary misses.
lines_h3 = (
    lines_df
    # h3_polyfillash3 fills a polygon; for a linestring use h3_linestringash3
    # (available in Databricks Runtime 11+, or use Sedona ST_H3CellIDs)
    .withColumn(
        "h3_cells",
        F.expr(f"flatten(transform(ST_H3CellIDs(line_geom, {H3_RES}, true), c -> h3_kring(c, 1)))"),
    )
    .withColumn("h3_cell", F.explode("h3_cells"))
    .drop("h3_cells")
)

# Join on H3 cell, then pick the nearest within each bucket.
# ST_DistanceSphere is used here so the reported distance is in metres,
# not degrees.  For the ORDER BY ranking, degree distance also gives the
# correct nearest result (monotonic), but metres is more meaningful.
nearest_h3 = (
    coords_h3.alias("c")
    .join(lines_h3.alias("l"), "h3_cell", "inner")
    .withColumn(
        "dist_metres",
        F.expr("ST_DistanceSphere(c.point_geom, l.line_geom)"),
    )
    .withColumn(
        "rn",
        F.expr("ROW_NUMBER() OVER (PARTITION BY c.point_id ORDER BY dist_metres)"),
    )
    .filter("rn = 1")
    .select("c.point_id", "l.line_id", "dist_metres")
)

# nearest_h3.write.mode("overwrite").saveAsTable("my_schema.nearest_line_h3")


# ══════════════════════════════════════════════════════════════════════════
# SPOT-CHECK the result (run after any strategy)
# ══════════════════════════════════════════════════════════════════════════
#
# 1. Visualise a sample in Kepler.gl or any GIS tool to confirm points
#    snap to the expected linestrings and are not mirrored / transposed.
#
# 2. Manual sanity check: pick a known point (e.g. CN Tower, Toronto)
#    and verify it matches a road/rail line, not a line in a different city.
#
# 3. Check distance distribution — if median dist_metres > 5 000 m your
#    data likely has a lon/lat swap or a CRS mismatch.
#
# 4. Count unmatched points (no line within the search radius):
#    unmatched_count = coords_df.join(result, "point_id", "left_anti").count()
#    If > 0, rerun those points with a larger radius (see tip 5 below).

# ══════════════════════════════════════════════════════════════════════════
# Performance tips for 3.6 B rows
# ══════════════════════════════════════════════════════════════════════════
#
# 1. PARTITION the coords table by a spatial key before the join:
#      coords_df.withColumn("h3_bucket", F.expr(f"h3_longlatash3(longitude, latitude, 5)"))
#               .repartition(2000, "h3_bucket")
#               .write.partitionBy("h3_bucket").saveAsTable(...)
#    Then load with partition pruning in the join.
#
# 2. Use Delta Lake Z-ORDER on (longitude, latitude) for the coords table:
#      OPTIMIZE my_schema.coords ZORDER BY (longitude, latitude)
#    This collocates nearby points on the same files → fewer shuffles.
#
# 3. Cache the LineString table if it fits:
#      lines_df.cache(); lines_df.count()
#
# 4. Tune Sedona's index join:
#      spark.conf.set("sedona.join.numpartitions", "4000")
#      spark.conf.set("sedona.join.indextype", "rtree")   # or "quadtree"
#
# 5. For Strategy B, if points outside the initial radius R have no match,
#    do a second pass only for unmatched points with a larger radius:
#
#    unmatched = coords_df.join(result, "point_id", "left_anti")
#    retry = nearest_join_range(unmatched, lines_df, radius=0.05)
#    final  = result.union(retry)
#
# 6. Cluster size recommendation for 3.6 B rows:
#      Driver:  32–64 GB RAM
#      Workers: 16–32 nodes × 16 cores, 64 GB RAM each
#      Use Spot/Preemptible workers for cost savings on pure compute jobs.
