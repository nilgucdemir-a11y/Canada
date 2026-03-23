"""
Nearest-neighbour spatial join in Databricks
=============================================
Join a GPKG LineString layer to a 3.6-billion-row (lon, lat) table.

Recommended stack
-----------------
* Apache Sedona  – native distributed spatial library for Spark
  Install on the cluster:  sedona[spark]  (PyPI) or use the
  Databricks Marketplace / cluster library UI.
* Databricks Runtime 13+ (Spark 3.4+)

Three strategies are provided, ordered from simplest to most scalable:

  A. Sedona ST_KNN  – exact k-nearest, works well when the LineString
                      table is small enough to broadcast (~millions of rows).
  B. Sedona range join + ST_Distance  – exact nearest via two-pass approach;
                      handles both tables at arbitrary scale.
  C. H3 bucketing   – approximate nearest; extremely fast; use when a
                      ~100-metre approximation is acceptable.
"""

# ── 0. Install / import ────────────────────────────────────────────────────
# In a Databricks notebook cell run first:
#   %pip install sedona apache-sedona keplergl  (then restart Python kernel)

from sedona.spark import SedonaContext
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

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
    # Create a Sedona geometry column from lon/lat
    .withColumn(
        "point_geom",
        F.expr("ST_Point(CAST(longitude AS DOUBLE), CAST(latitude AS DOUBLE))"),
    )
)


# ══════════════════════════════════════════════════════════════════════════
# STRATEGY A – ST_KNN (exact; broadcast the lines table)
# Best when: LineString table fits in driver/executor memory (< ~5 M rows).
# ══════════════════════════════════════════════════════════════════════════

def nearest_join_knn(coords_df, lines_df, k=1):
    """
    Use Sedona's ST_KNN to find the k nearest LineStrings for every point.

    ST_KNN performs a distributed KNN search using the R-tree index that
    Sedona builds automatically on the broadcast side.
    """
    coords_df.createOrReplaceTempView("coords")
    lines_df.createOrReplaceTempView("lines")

    result = sedona.sql(f"""
        SELECT
            c.point_id,
            l.line_id,
            ST_Distance(c.point_geom, l.line_geom) AS dist_degrees
        FROM coords c
        JOIN lines l
        ON ST_KNN(c.point_geom, l.line_geom, {k}, true)
    """)
    return result

# result_knn = nearest_join_knn(coords_df, lines_df, k=1)
# result_knn.write.mode("overwrite").saveAsTable("my_schema.nearest_line_per_point")


# ══════════════════════════════════════════════════════════════════════════
# STRATEGY B – Range join + ST_Distance (exact; both tables distributed)
# Best when: both tables are huge and LineStrings cannot be broadcast.
# ══════════════════════════════════════════════════════════════════════════
#
# Idea:
#   1. Pick a search radius R (degrees) that is guaranteed to contain at
#      least one LineString for every point.
#   2. Use ST_Intersects(ST_Buffer(point, R), line_geom) as the join
#      predicate – Sedona optimises this with a spatial partition index.
#   3. Rank results by distance and keep rank = 1.
#
# Finding a good R:
#   * 0.01° ≈ 1 km at mid-latitudes  (good default for dense road networks)
#   * If some points get no match, double R and rerun for those points only.

SEARCH_RADIUS_DEG = 0.01   # ≈ 1 km

coords_df.createOrReplaceTempView("coords")
lines_df.createOrReplaceTempView("lines")

nearest_range = sedona.sql(f"""
    WITH candidates AS (
        SELECT
            c.point_id,
            c.point_geom,
            l.line_id,
            l.line_geom,
            ST_Distance(c.point_geom, l.line_geom)  AS dist_degrees,
            -- Haversine-accurate distance in metres (optional, slower)
            -- ST_DistanceSphere(c.point_geom, l.line_geom) AS dist_metres,
            ROW_NUMBER() OVER (
                PARTITION BY c.point_id
                ORDER BY ST_Distance(c.point_geom, l.line_geom)
            ) AS rn
        FROM coords      c
        JOIN lines        l
          ON ST_Intersects(
               ST_Buffer(c.point_geom, {SEARCH_RADIUS_DEG}),
               l.line_geom
             )
    )
    SELECT point_id, line_id, dist_degrees
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

# Join on H3 cell, then pick the nearest within each bucket
nearest_h3 = (
    coords_h3.alias("c")
    .join(lines_h3.alias("l"), "h3_cell", "inner")
    .withColumn(
        "dist",
        F.expr("ST_Distance(c.point_geom, l.line_geom)"),
    )
    .withColumn(
        "rn",
        F.expr("ROW_NUMBER() OVER (PARTITION BY c.point_id ORDER BY dist)"),
    )
    .filter("rn = 1")
    .select("c.point_id", "l.line_id", "dist")
)

# nearest_h3.write.mode("overwrite").saveAsTable("my_schema.nearest_line_h3")


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
