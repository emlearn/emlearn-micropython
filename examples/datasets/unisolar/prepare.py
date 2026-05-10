#!/usr/bin/env python3
"""Download and preprocess UniSolar solar generation dataset.

Downloads the UniSolar dataset from Kaggle (42 PV sites across 5 Australian
campuses), saves per-site generation data and per-campus weather data as
.npy files with timestamps embedded, plus a metadata file for site-to-campus
mapping.

Output structure:
    prepare.py              -- this script
    metadata.json           -- structured site↔campus mapping + column names
    site_01/                -- one dir per PV site (42 total)
        solar_generation.npy -- structured array: float32 columns
            epoch_sec   : datetime as float32 seconds since Unix epoch
            generation  : PV power output in kW
    campus_01/              -- one dir per campus/weather source (5 total)
        weather_features.npy -- structured array: float32 columns
            epoch_sec       : datetime as float32 seconds since Unix epoch
            apparent_temp   : ApparentTemperature (°C)
            air_temp        : AirTemperature (°C)
            dew_point_temp  : DewPointTemperature (°C)
            rel_humidity    : RelativeHumidity (%)
            wind_speed      : WindSpeed (m/s)
            wind_direction  : WindDirection (degrees)

Each site's solar_generation array has a 'timestamp' field and a 'generation'
field. Each campus's weather_features array has a 'timestamp' field and six
weather feature columns.

Run with CPython: python3 examples/datasets/unisolar/prepare.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import kagglehub


# Solar generation column spec (epoch_sec + data)
SOLAR_FIELDS = [
    ("timestamp", np.float32),  # epoch seconds since Unix epoch
    ("generation", np.float32),
]
SOLAR_DTYPE = np.dtype(SOLAR_FIELDS)

# Weather features column spec (epoch_sec + 6 feature columns)
WEATHER_FIELDS = [
    ("timestamp", np.float32),  # epoch seconds since Unix epoch
    ("apparent_temp", np.float32),
    ("air_temp", np.float32),
    ("dew_point_temp", np.float32),
    ("rel_humidity", np.float32),
    ("wind_speed", np.float32),
    ("wind_direction", np.float32),
]
WEATHER_DTYPE = np.dtype(WEATHER_FIELDS)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_dataset() -> dict:
    """Download UniSolar dataset using kagglehub and return file paths."""
    print("Downloading UniSolar dataset from Kaggle...")
    path = kagglehub.dataset_download("cdaclab/unisolar")

    # kagglehub may place files directly under `path/unisolar/` or in a
    # versioned subdirectory.  Walk to find the three CSVs.
    gen_file = None
    weather_file = None
    sites_file = None
    for root, _dirs, files in os.walk(path):
        if "Solar_Energy_Generation.csv" in files:
            gen_file = os.path.join(root, "Solar_Energy_Generation.csv")
        if "Weather_Data_reordered_all.csv" in files:
            weather_file = os.path.join(root, "Weather_Data_reordered_all.csv")
        if "Solar_Site_Details.csv" in files:
            sites_file = os.path.join(root, "Solar_Site_Details.csv")

    assert gen_file, f"Solar_Energy_Generation.csv not found under {path}"
    assert weather_file, f"Weather_Data_reordered_all.csv not found under {path}"
    assert sites_file, f"Solar_Site_Details.csv not found under {path}"

    return {"generation": gen_file, "weather": weather_file, "sites": sites_file}


# ---------------------------------------------------------------------------
# Load & parse CSVs
# ---------------------------------------------------------------------------

def load_csvs(file_paths: dict) -> tuple:
    """Load the three CSV files and parse Timestamp → datetime."""
    print("Loading CSV files...")

    df_gen = pd.read_csv(file_paths["generation"])
    df_weather = pd.read_csv(file_paths["weather"])
    df_sites = pd.read_csv(file_paths["sites"])

    df_gen["Timestamp"] = pd.to_datetime(df_gen["Timestamp"])
    df_weather["Timestamp"] = pd.to_datetime(df_weather["Timestamp"])

    print(f"  Generation: {df_gen.shape} ({df_gen['SiteKey'].nunique()} sites)")
    print(f"  Weather:    {df_weather.shape} ({df_weather['CampusKey'].nunique()} campuses)")
    print(f"  Sites:      {df_sites.shape}")

    return df_gen, df_weather, df_sites


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def build_metadata(df_sites: pd.DataFrame) -> dict:
    """Build a JSON-serialisable metadata dict with site↔campus mapping.

    Also records the column names used in each .npy file so consumers know
    how to interpret the structured arrays without reading source code.
    """
    campus_to_sites = {}
    site_to_campus = {}
    for ck, group in df_sites.groupby("CampusKey"):
        ck = int(ck)
        sk_list = sorted(int(sk) for sk in group["SiteKey"])
        campus_to_sites[ck] = sk_list
        for sk in sk_list:
            site_to_campus[int(sk)] = ck

    def _norm_fields(fields):
        """Convert dtype objects to their numpy type strings for JSON."""
        return [(name, np.dtype(t).name) for name, t in fields]

    return {
        "campus_to_sites": {str(k): v for k, v in campus_to_sites.items()},
        "site_to_campus": {str(k): int(v) for k, v in site_to_campus.items()},
        "n_sites": int(df_sites["SiteKey"].nunique()),
        "n_campuses": int(df_sites["CampusKey"].nunique()),
        "solar_columns": _norm_fields(SOLAR_FIELDS),
        "weather_columns": _norm_fields(WEATHER_FIELDS),
    }


# ---------------------------------------------------------------------------
# Timestamp helper — float32 epoch seconds
# ---------------------------------------------------------------------------

def _ts_to_epoch_sec(timestamps: np.ndarray) -> np.ndarray:
    """Convert datetime64 array to float32 seconds since Unix epoch."""
    # Convert to int64 nanoseconds, then divide by 1e9 → float32 seconds
    ns = timestamps.astype("datetime64[ns]").astype(np.int64)
    return (ns / 1e9).astype(np.float32)


# ---------------------------------------------------------------------------
# Save helpers — structured arrays with embedded timestamps
# ---------------------------------------------------------------------------

def save_site_generation(
    df_gen: pd.DataFrame, site_key: int, output_dir: Path
) -> None:
    """Save generation data for one site as a structured .npy array.

    The structured dtype contains two columns: 'timestamp' (float32 epoch sec)
    and 'generation' (float32 kW).
    """
    site_df = df_gen[df_gen["SiteKey"] == site_key].sort_values("Timestamp")
    if site_df.empty:
        return

    timestamps_sec = _ts_to_epoch_sec(site_df["Timestamp"].values)
    gen_values = pd.to_numeric(
        site_df["SolarGeneration"], errors="coerce"
    ).astype(np.float32).values

    # Build structured array
    n = len(timestamps_sec)
    arr = np.empty(n, dtype=SOLAR_DTYPE)
    arr["timestamp"] = timestamps_sec
    arr["generation"] = gen_values

    save_path = output_dir / f"site_{site_key:02d}"
    os.makedirs(save_path, exist_ok=True)
    np.save(save_path / "solar_generation.npy", arr)

    nan_pct = np.isnan(gen_values).mean() * 100
    print(
        f"  site_{site_key:02d}: {n:>7} rows, NaN={nan_pct:.1f}%, "
        f"ts range=({timestamps_sec[0]}) to ({timestamps_sec[-1]})"
    )


def save_campus_weather(
    df_weather: pd.DataFrame, campus_key: int, output_dir: Path
) -> None:
    """Save weather data for one campus as a structured .npy array.

    The structured dtype contains seven columns: 'timestamp' (float32 epoch sec)
    plus six weather feature columns.
    """
    campus_df = (
        df_weather[df_weather["CampusKey"] == campus_key]
        .sort_values("Timestamp")
        .copy()
    )
    if campus_df.empty:
        return

    timestamps_sec = _ts_to_epoch_sec(campus_df["Timestamp"].values)

    csv_to_arr = {
        "ApparentTemperature": "apparent_temp",
        "AirTemperature": "air_temp",
        "DewPointTemperature": "dew_point_temp",
        "RelativeHumidity": "rel_humidity",
        "WindSpeed": "wind_speed",
        "WindDirection": "wind_direction",
    }
    feature_values = campus_df[list(csv_to_arr.keys())].astype(np.float32).values

    # Build structured array
    n = len(timestamps_sec)
    arr = np.empty(n, dtype=WEATHER_DTYPE)
    arr["timestamp"] = timestamps_sec
    for csv_name, arr_name in csv_to_arr.items():
        idx = list(csv_to_arr.keys()).index(csv_name)
        arr[arr_name] = feature_values[:, idx]

    save_path = output_dir / f"campus_{campus_key:02d}"
    os.makedirs(save_path, exist_ok=True)
    np.save(save_path / "weather_features.npy", arr)

    nan_pct = np.isnan(feature_values).mean() * 100
    print(
        f"  campus_{campus_key:02d}: {n:>7} rows, NaN={nan_pct:.1f}%, "
        f"ts range=({timestamps_sec[0]}) to ({timestamps_sec[-1]})"
    )


# ---------------------------------------------------------------------------
# Check / verification — load everything back and assemble DataFrames
# ---------------------------------------------------------------------------

def check_data(output_dir: Path, metadata: dict) -> tuple:
    """Load all prepared .npy files into memory and build DataFrames.

    Returns (site_dfs, campus_dfs):
        site_dfs   — list of per-site DataFrames
        campus_dfs — dict keyed by int(campus_key) → DataFrame
    """
    print("\n" + "=" * 70)
    print("Checking prepared data")
    print("=" * 70)

    # ---- Per-site generation DataFrames ----
    site_dfs = []
    for sk in sorted(int(k) for k in metadata["site_to_campus"]):
        sp = output_dir / f"site_{sk:02d}"
        arr = np.load(sp / "solar_generation.npy")

        ck = metadata["site_to_campus"][str(sk)]
        df = pd.DataFrame({
            "site_key": sk,
            "campus_key": ck,
            "timestamp": arr["timestamp"].astype(np.float64),  # convert back to float64 for display
            "generation": arr["generation"],
        })
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="s")
        site_dfs.append(df)

    print(f"\n  Loaded {len(site_dfs)} site DataFrames:")
    for df in site_dfs[:3]:
        ck = int(df["campus_key"].iloc[0])
        nan_pct = df["generation"].isna().mean() * 100
        print(
            f"    site={int(df['site_key'].iloc[0])} campus={ck}: "
            f"{len(df)} rows, ts dtype={df['timestamp'].dtype}, "
            f"gen dtype={df['generation'].dtype}, NaN={nan_pct:.1f}%"
        )
    if len(site_dfs) > 3:
        print(f"    ... and {len(site_dfs) - 3} more sites")

    total_site_rows = sum(len(d) for d in site_dfs)
    print(f"\n  Total site rows across all sites: {total_site_rows}")

    # ---- Per-campus weather DataFrames ----
    campus_dfs = {}
    feature_cols = ["apparent_temp", "air_temp", "dew_point_temp",
                    "rel_humidity", "wind_speed", "wind_direction"]
    for ck in sorted(metadata["campus_to_sites"], key=int):
        cp = output_dir / f"campus_{int(ck):02d}"
        arr = np.load(cp / "weather_features.npy")

        df_cols = {
            "campus_key": int(ck),
            "timestamp": arr["timestamp"].astype(np.float64),
        }
        for fname in feature_cols:
            df_cols[fname] = arr[fname]

        df = pd.DataFrame(df_cols)
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="s")
        campus_dfs[int(ck)] = df

    print(f"\n  Loaded {len(campus_dfs)} campus DataFrames:")
    for idx, ck in list(enumerate(sorted(campus_dfs.keys())))[:3]:
        df = campus_dfs[ck]
        nan_pct = df.isna().mean().mean() * 100
        shape = str(df.shape) if idx == 0 else "…"
        print(
            f"    campus={ck}: {len(df)} rows, ts dtype={df['timestamp'].dtype}, "
            f"weather shape={shape}, NaN={nan_pct:.1f}%"
        )
    if len(campus_dfs) > 3:
        print(f"    ... and {len(campus_dfs) - 3} more campuses")

    total_weather_rows = sum(len(d) for d in campus_dfs.values())
    print(f"\n  Total weather rows across all campuses: {total_weather_rows}")

    # ---- Spot-check timestamp alignment between a site and its campus ----
    print("\n  Timestamp alignment check (site_01 vs campus_02):")
    s_arr = np.load(output_dir / "site_01" / "solar_generation.npy")
    c_arr = np.load(output_dir / "campus_02" / "weather_features.npy")
    overlap = int(np.isin(s_arr["timestamp"], c_arr["timestamp"]).sum())
    print(
        f"    Site 01 has {len(s_arr)} timestamps, "
        f"Campus 02 has {len(c_arr)} timestamps, "
        f"{overlap}/{len(s_arr)} match ({overlap / len(s_arr):.1%})"
    )

    return site_dfs, campus_dfs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = Path(here)
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("UniSolar Dataset Preparation")
    print("=" * 70)

    # 1. Download
    file_paths = download_dataset()

    # 2. Load & parse
    df_gen, df_weather, df_sites = load_csvs(file_paths)

    # 3. Build metadata
    metadata = build_metadata(df_sites)
    meta_path = OUTPUT_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved metadata -> {meta_path}")

    # Show campus structure
    all_site_keys = sorted(df_gen["SiteKey"].unique())
    all_campus_keys = sorted(df_weather["CampusKey"].unique())

    print(f"\n{len(all_site_keys)} sites across {len(all_campus_keys)} campuses:")
    for ck_str in sorted(metadata["campus_to_sites"]):
        sk_list = metadata["campus_to_sites"][ck_str]
        print(
            f"  Campus {ck_str}: sites {sk_list[0]:3d}–{sk_list[-1]:3d} "
            f"({len(sk_list)} sites)"
        )

    # 4. Save per-site generation data (structured arrays with timestamps)
    print("\nSaving per-site generation data...")
    for sk in all_site_keys:
        save_site_generation(df_gen, int(sk), OUTPUT_DIR)

    # 5. Save per-campus weather data (structured arrays with timestamps)
    print("\nSaving per-campus weather data...")
    for ck in all_campus_keys:
        save_campus_weather(df_weather, int(ck), OUTPUT_DIR)

    # 6. Check / verify — load everything back into DataFrames
    site_dfs, campus_dfs = check_data(OUTPUT_DIR, metadata)

    print(f"\nDone! Saved {len(all_site_keys)} site files + {len(all_campus_keys)} campus files")
    print("Timestamps embedded as float32 epoch seconds in structured arrays.")


if __name__ == "__main__":
    main()
