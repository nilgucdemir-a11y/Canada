"""
Fetch daily precipitation historical data for all Canadian climate stations (1950-2025)
using the MSC GeoMet OGC API - Features (OAFeat) service.

Based on the pattern from:
https://eccc-msc.github.io/open-data/usage/use-case_oafeat/use-case_oafeat-script_en/

API base: https://api.weather.gc.ca/
Collections used:
  - climate-stations  : metadata for all climate stations
  - climate-daily     : daily climate observations (incl. TOTAL_PRECIPITATION)
"""

import json
import math
import time
import logging
from pathlib import Path

import pandas as pd
from owslib.ogcapi.features import Features

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL = "https://api.weather.gc.ca/"
STATIONS_COLLECTION = "climate-stations"
DAILY_COLLECTION = "climate-daily"

START_DATE = "1950-01-01"
END_DATE = "2025-12-31"
DATETIME_RANGE = f"{START_DATE}/{END_DATE}"

# OAFeat hard limit is 10 000 features per request
PAGE_LIMIT = 10_000

OUTPUT_DIR = Path("data")
STATIONS_FILE = OUTPUT_DIR / "stations.csv"
PRECIPITATION_FILE = OUTPUT_DIR / "precipitation_daily_1950_2025.csv"
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.json"

# Retry settings for transient network errors
MAX_RETRIES = 4
BACKOFF_BASE = 2  # seconds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _retry(func, *args, **kwargs):
    """Call *func* with exponential back-off on failure."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise
            wait = BACKOFF_BASE ** attempt
            log.warning("Attempt %d failed (%s). Retrying in %ds …", attempt, exc, wait)
            time.sleep(wait)


def _collection_items_page(oafeat, collection, offset=0, **kwargs):
    """Fetch one page of items from *collection*, starting at *offset*."""
    response = _retry(
        oafeat.collection_items,
        collection,
        limit=PAGE_LIMIT,
        offset=offset,
        **kwargs,
    )
    return response


def features_to_df(geojson_response):
    """Convert a GeoJSON FeatureCollection response to a flat DataFrame."""
    records = []
    for feat in geojson_response.get("features", []):
        row = feat.get("properties", {})
        geom = feat.get("geometry")
        if geom and geom.get("type") == "Point":
            coords = geom.get("coordinates", [None, None, None])
            row["LONGITUDE"] = coords[0] if len(coords) > 0 else None
            row["LATITUDE"] = coords[1] if len(coords) > 1 else None
            row["ELEVATION"] = coords[2] if len(coords) > 2 else None
        records.append(row)
    return pd.DataFrame(records)


def fetch_all_pages(oafeat, collection, **filter_kwargs):
    """
    Iterate through all pages of *collection* for the given filters and return
    a single concatenated DataFrame.

    The OAFeat response contains ``numberMatched`` which tells us how many
    total features match the query; we use it to compute the number of pages.
    """
    # --- first page (also gives us numberMatched) ---
    log.info("Fetching first page of '%s' …", collection)
    first_page = _collection_items_page(oafeat, collection, offset=0, **filter_kwargs)

    number_matched = first_page.get("numberMatched", None)
    frames = [features_to_df(first_page)]

    if number_matched is None or number_matched <= PAGE_LIMIT:
        return pd.concat(frames, ignore_index=True)

    # --- remaining pages ---
    n_pages = math.ceil(number_matched / PAGE_LIMIT)
    log.info("  %d features matched → fetching %d page(s) …", number_matched, n_pages)

    for page_idx in range(1, n_pages):
        offset = page_idx * PAGE_LIMIT
        log.info("  page %d/%d (offset=%d) …", page_idx + 1, n_pages, offset)
        page = _collection_items_page(oafeat, collection, offset=offset, **filter_kwargs)
        frames.append(features_to_df(page))

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Step 1 – fetch all climate stations
# ---------------------------------------------------------------------------

def fetch_stations(oafeat):
    """Return a DataFrame of all climate stations."""
    if STATIONS_FILE.exists():
        log.info("Loading cached stations from %s", STATIONS_FILE)
        return pd.read_csv(STATIONS_FILE, dtype=str)

    log.info("=== Fetching all climate stations ===")
    df = fetch_all_pages(oafeat, STATIONS_COLLECTION)
    log.info("  Retrieved %d stations.", len(df))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(STATIONS_FILE, index=False)
    log.info("  Saved to %s", STATIONS_FILE)
    return df


# ---------------------------------------------------------------------------
# Step 2 – fetch daily precipitation for every station, 1950-2025
# ---------------------------------------------------------------------------

def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"completed": []}


def save_checkpoint(completed_ids):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"completed": list(completed_ids)}, f)


def fetch_precipitation_all_stations(oafeat, stations_df):
    """
    For every station in *stations_df* query the ``climate-daily`` collection
    for TOTAL_PRECIPITATION between START_DATE and END_DATE.

    Results are appended incrementally to PRECIPITATION_FILE so that the
    script can be safely interrupted and resumed.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint()
    completed = set(checkpoint["completed"])

    # Determine the station identifier column
    id_col = None
    for candidate in ("CLIMATE_IDENTIFIER", "STN_ID", "STATION_ID", "ID"):
        if candidate in stations_df.columns:
            id_col = candidate
            break
    if id_col is None:
        raise ValueError(
            f"Cannot find a station ID column. Available columns: {list(stations_df.columns)}"
        )

    station_ids = stations_df[id_col].dropna().unique().tolist()
    total = len(station_ids)
    log.info("=== Fetching precipitation for %d stations (%s – %s) ===",
             total, START_DATE, END_DATE)

    # Open output file; write header only on first creation
    write_header = not PRECIPITATION_FILE.exists()

    for idx, station_id in enumerate(station_ids, start=1):
        station_id = str(station_id).strip()
        if station_id in completed:
            log.debug("[%d/%d] %s already done, skipping.", idx, total, station_id)
            continue

        log.info("[%d/%d] Station %s …", idx, total, station_id)

        try:
            df = fetch_all_pages(
                oafeat,
                DAILY_COLLECTION,
                datetime=DATETIME_RANGE,
                CLIMATE_IDENTIFIER=station_id,
            )
        except Exception as exc:
            log.error("  Failed for station %s: %s — skipping.", station_id, exc)
            continue

        if df.empty:
            log.info("  No data for station %s in the requested period.", station_id)
        else:
            # Keep only precipitation-relevant columns (plus identifiers)
            keep_cols = [
                c for c in [
                    "CLIMATE_IDENTIFIER",
                    "STATION_NAME",
                    "PROVINCE_CODE",
                    "LOCAL_DATE",
                    "TOTAL_PRECIPITATION",
                    "TOTAL_RAIN",
                    "TOTAL_SNOW",
                    "SNOW_ON_GROUND",
                    "LATITUDE",
                    "LONGITUDE",
                    "ELEVATION",
                ] if c in df.columns
            ]
            df = df[keep_cols]
            df.to_csv(
                PRECIPITATION_FILE,
                mode="a",
                header=write_header,
                index=False,
            )
            write_header = False
            log.info("  Wrote %d rows.", len(df))

        completed.add(station_id)
        save_checkpoint(completed)

    log.info("=== Done. Output: %s ===", PRECIPITATION_FILE)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    oafeat = Features(API_BASE_URL)

    stations_df = fetch_stations(oafeat)
    fetch_precipitation_all_stations(oafeat, stations_df)

    # Final summary
    if PRECIPITATION_FILE.exists():
        result_df = pd.read_csv(PRECIPITATION_FILE, dtype={"CLIMATE_IDENTIFIER": str})
        log.info(
            "Final dataset: %d rows, %d stations, columns: %s",
            len(result_df),
            result_df["CLIMATE_IDENTIFIER"].nunique() if "CLIMATE_IDENTIFIER" in result_df.columns else "?",
            list(result_df.columns),
        )


if __name__ == "__main__":
    main()
