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

import math

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
    .config("spark.sql.shuffle.partitions", "2000")
    .getOrCreate()
)
sedona = SedonaContext.create(config)

# ── 2. Config ──────────────────────────────────────────────────────────────
GPKG_PATH       = "/Volumes/my_catalog/my_schema/my_volume/roads.gpkg"
GPKG_LAYER      = "roads"
COORDS_TABLE    = "my_catalog.my_schema.coords"
OUTPUT_TABLE    = "my_catalog.my_schema.nearest_line_per_point"
SEARCH_RADIUS_M = 2_000   # metres; increase for sparse networks


# ══════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════

def load_lines(gpkg_path: str, layer_name: str):
    lines_df = (
        sedona.read.format("geopackage")
        .option("layerName", layer_name)
        .load(gpkg_path)
        .withColumnRenamed("geometry", "line_geom")
        .select("line_id", "line_geom")
    )
    lines_df.cache()
    lines_df.count()
    return lines_df


def load_coords(table: str):
    return (
        spark.table(table)
        .select("point_id", "longitude", "latitude")
        # X=longitude, Y=latitude — swapping these is a common mistake.
        # ST_SetSRID tags the geometry as 4326 so Sedona can use
        # ST_Transform, ST_DistanceSpheroid, etc. correctly.
        .withColumn(
            "point_geom",
            F.expr(
                "ST_SetSRID("
                "  ST_Point(CAST(longitude AS DOUBLE), CAST(latitude AS DOUBLE)),"
                "  4326"
                ")"
            ),
        )
    )


# ══════════════════════════════════════════════════════════════════════════
# CRS validation  (call before any join)
# ══════════════════════════════════════════════════════════════════════════

def validate_crs(lines_df, coords_df):
    """
    Run three quick checks that catch the most common CRS mistakes.
    Raises RuntimeError if a problem is found so the pipeline fails fast.
    """
    # Check 1: GPKG SRID must be 4326.
    srid_rows = lines_df.selectExpr("ST_SRID(line_geom) AS srid").distinct().collect()
    srids = {r.srid for r in srid_rows}
    if srids != {4326}:
        raise RuntimeError(
            f"lines_df SRID is {srids}, expected {{4326}}. "
            "Reproject with ST_Transform or fix with ST_SetSRID."
        )

    # Check 2: Coordinate range + Canada bounding-box sanity.
    stats = coords_df.selectExpr(
        "MIN(longitude) AS min_lon", "MAX(longitude) AS max_lon",
        "MIN(latitude)  AS min_lat", "MAX(latitude)  AS max_lat",
        "COUNT(*)       AS total_rows",
        "SUM(CASE WHEN longitude NOT BETWEEN -180 AND 180 THEN 1 ELSE 0 END) AS bad_lon",
        "SUM(CASE WHEN latitude  NOT BETWEEN  -90 AND  90 THEN 1 ELSE 0 END) AS bad_lat",
        "SUM(CASE WHEN longitude NOT BETWEEN -141 AND  -52 THEN 1 ELSE 0 END) AS outside_canada_lon",
        "SUM(CASE WHEN latitude  NOT BETWEEN   42 AND   84 THEN 1 ELSE 0 END) AS outside_canada_lat",
    ).first()

    print(
        f"coords: {stats.total_rows:,} rows  "
        f"lon=[{stats.min_lon:.4f}, {stats.max_lon:.4f}]  "
        f"lat=[{stats.min_lat:.4f}, {stats.max_lat:.4f}]"
    )
    if stats.bad_lon > 0 or stats.bad_lat > 0:
        raise RuntimeError(
            f"Invalid coordinates: {stats.bad_lon} bad lon, {stats.bad_lat} bad lat."
        )
    if stats.outside_canada_lon > 0 or stats.outside_canada_lat > 0:
        print(
            f"WARNING: {stats.outside_canada_lon} points outside Canada lon range, "
            f"{stats.outside_canada_lat} outside lat range — check your data."
        )

    # Check 3: Smoke-test ST_Point(lon, lat) axis order.
    # Vancouver at (-123.12, 49.28) must NOT appear as (49.28, -123.12).
    wkt = sedona.sql(
        "SELECT ST_AsText(ST_SetSRID(ST_Point(-123.12, 49.28), 4326)) AS wkt"
    ).first().wkt
    if wkt != "POINT (-123.12 49.28)":
        raise RuntimeError(
            f"ST_Point axis order wrong: got {wkt!r}. "
            "Your lon/lat columns may be swapped."
        )

    print("CRS validation passed.")


# ══════════════════════════════════════════════════════════════════════════
# STRATEGY A – ST_KNN  (exact; broadcast the lines table)
# Best when: LineString table fits in executor memory (< ~5 M rows).
# ══════════════════════════════════════════════════════════════════════════

def nearest_join_knn(coords_df, lines_df, k: int = 1):
    """
    Use Sedona's ST_KNN to find the k nearest LineStrings for every point.

    ST_KNN builds an R-tree on the broadcast side automatically.

    CRS note: ST_KNN ranks by Euclidean degree-distance (correct for finding
    the nearest; monotonic), but the output distance is in metres via
    ST_DistanceSphere so the value is meaningful.
    """
    coords_df.createOrReplaceTempView("coords")
    lines_df.createOrReplaceTempView("lines")

    return sedona.sql(f"""
        SELECT
            c.point_id,
            l.line_id,
            ST_DistanceSphere(c.point_geom, l.line_geom) AS dist_metres
        FROM coords c
        JOIN lines  l
          ON ST_KNN(c.point_geom, l.line_geom, {k}, true)
    """)


# ══════════════════════════════════════════════════════════════════════════
# STRATEGY B – Range join + ST_DistanceSphere  (exact; both tables huge)
# Best when: both tables are too large to broadcast.
# ══════════════════════════════════════════════════════════════════════════
#
# Two CRS-correct sub-options:
#
#   B-1 (recommended): reproject to EPSG:3347 so ST_Buffer is a true circle
#       in metres.  Most accurate; works for all of Canada.
#
#   B-2 (fallback): stay in 4326 but inflate the degree radius to compensate
#       for latitude distortion at the northernmost point of your data.
#       Slightly over-fetches candidates but avoids ST_Transform cost.

def nearest_join_range_b1(coords_df, lines_df, radius_m: int = SEARCH_RADIUS_M):
    """
    B-1: reproject to EPSG:3347 (Statistics Canada Lambert), buffer in metres.
    ST_Buffer on a projected CRS creates a true circle.
    """
    coords_df.createOrReplaceTempView("coords")
    lines_df.createOrReplaceTempView("lines")

    return sedona.sql(f"""
        WITH projected AS (
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
                ST_Distance(p.pt_3347, l.ln_3347) AS dist_metres,
                ROW_NUMBER() OVER (
                    PARTITION BY p.point_id
                    ORDER BY     ST_Distance(p.pt_3347, l.ln_3347)
                ) AS rn
            FROM projected  p
            JOIN lines_proj l
              ON ST_Intersects(
                   ST_Buffer(p.pt_3347, {radius_m}),
                   l.ln_3347
                 )
        )
        SELECT point_id, line_id, dist_metres
        FROM   candidates
        WHERE  rn = 1
    """)


def nearest_join_range_b2(
    coords_df,
    lines_df,
    radius_m: int = SEARCH_RADIUS_M,
    max_lat_deg: float = 84.0,
):
    """
    B-2: stay in 4326 but inflate the degree radius to account for latitude
    distortion.  At max_lat_deg, 1° longitude ≈ 111_320 × cos(lat) metres.
    Dividing radius_m by that value gives a degree radius that is always
    large enough across all of Canada.
    """
    deg_radius = radius_m / (111_320 * math.cos(math.radians(max_lat_deg)))

    coords_df.createOrReplaceTempView("coords")
    lines_df.createOrReplaceTempView("lines")

    return sedona.sql(f"""
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
                   ST_Buffer(c.point_geom, {deg_radius:.6f}),
                   l.line_geom
                 )
        )
        SELECT point_id, line_id, dist_metres
        FROM   candidates
        WHERE  rn = 1
    """)


# ══════════════════════════════════════════════════════════════════════════
# STRATEGY C – H3 bucketing  (approximate; fastest at 3.6 B scale)
# Best when: ~100 m accuracy is acceptable and you need max throughput.
#
# H3 resolution guide:
#   res 8  → avg cell ≈  0.74 km²  (side ≈ 460 m)
#   res 9  → avg cell ≈  0.10 km²  (side ≈ 174 m)
#   res 10 → avg cell ≈  0.015 km² (side ≈  65 m)
# ══════════════════════════════════════════════════════════════════════════

def nearest_join_h3(coords_df, lines_df, h3_res: int = 9):
    """
    Bucket points and linestring cells by H3 index, then pick nearest per point.
    k-ring=1 is added to the linestring cells to avoid missing features that
    straddle a cell boundary.
    """
    coords_h3 = coords_df.withColumn(
        "h3_cell",
        F.expr(f"h3_longlatash3(longitude, latitude, {h3_res})"),
    )

    lines_h3 = (
        lines_df
        .withColumn(
            "h3_cells",
            F.expr(
                f"flatten(transform("
                f"  ST_H3CellIDs(line_geom, {h3_res}, true),"
                f"  c -> h3_kring(c, 1)"
                f"))"
            ),
        )
        .withColumn("h3_cell", F.explode("h3_cells"))
        .drop("h3_cells")
    )

    return (
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


# ══════════════════════════════════════════════════════════════════════════
# Spot-check helpers  (call on the result DataFrame before writing)
# ══════════════════════════════════════════════════════════════════════════

def spot_check(result_df, coords_df, label: str = "result"):
    """
    Print a distance distribution summary and the unmatched-point count.
    A median dist_metres > 5 000 m usually signals a lon/lat swap or CRS mismatch.
    """
    result_df.createOrReplaceTempView("_result")

    print(f"\n── {label} distance distribution (metres) ──")
    sedona.sql("""
        SELECT
            COUNT(*)                                              AS matched_points,
            ROUND(MIN(dist_metres),    1)                        AS min_m,
            ROUND(PERCENTILE(dist_metres, 0.50), 1)              AS p50_m,
            ROUND(PERCENTILE(dist_metres, 0.95), 1)              AS p95_m,
            ROUND(MAX(dist_metres),    1)                        AS max_m
        FROM _result
    """).show()

    unmatched = coords_df.join(result_df.select("point_id"), "point_id", "left_anti").count()
    print(f"Unmatched points (no line within radius): {unmatched:,}")
    if unmatched > 0:
        print("  → Re-run those points with a larger SEARCH_RADIUS_M.")


def second_pass(coords_df, lines_df, first_result_df, radius_m: int):
    """
    Re-run Strategy B-1 on unmatched points with a larger radius,
    then union with the first-pass result.
    """
    unmatched = coords_df.join(first_result_df.select("point_id"), "point_id", "left_anti")
    retry = nearest_join_range_b1(unmatched, lines_df, radius_m=radius_m)
    return first_result_df.union(retry)


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    lines_df  = load_lines(GPKG_PATH, GPKG_LAYER)
    coords_df = load_coords(COORDS_TABLE)

    validate_crs(lines_df, coords_df)

    # Pick one strategy.  B-1 is recommended for Canada at scale.
    result = nearest_join_range_b1(coords_df, lines_df, radius_m=SEARCH_RADIUS_M)

    spot_check(result, coords_df, label="B-1 (EPSG:3347)")

    # Widen the search for any unmatched points (sparse areas, e.g. far north).
    result = second_pass(coords_df, lines_df, result, radius_m=SEARCH_RADIUS_M * 5)

    spot_check(result, coords_df, label="after second pass")

    result.write.mode("overwrite").saveAsTable(OUTPUT_TABLE)
    print(f"Saved → {OUTPUT_TABLE}")


# ══════════════════════════════════════════════════════════════════════════
# Performance tips for 3.6 B rows
# ══════════════════════════════════════════════════════════════════════════
#
# 1. PARTITION coords by a coarse H3 bucket before the join:
#      coords_df
#        .withColumn("h3_bucket", F.expr("h3_longlatash3(longitude, latitude, 5)"))
#        .repartition(2000, "h3_bucket")
#        .write.partitionBy("h3_bucket").saveAsTable(...)
#
# 2. Delta Z-ORDER on (longitude, latitude):
#      OPTIMIZE my_schema.coords ZORDER BY (longitude, latitude)
#
# 3. Tune Sedona's index join:
#      spark.conf.set("sedona.join.numpartitions", "4000")
#      spark.conf.set("sedona.join.indextype", "rtree")
#
# 6. Cluster size recommendation for 3.6 B rows:
#      Driver:  32–64 GB RAM
#      Workers: 16–32 nodes × 16 cores, 64 GB RAM each
