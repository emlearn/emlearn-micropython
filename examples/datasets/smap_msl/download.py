#!/usr/bin/env python3
"""Download SMAP and MSL NASA anomaly detection datasets.

Downloads the raw SMAP (Soil Moisture Active Passive) and MSL (Mars Science Laboratory)
telemetry datasets from Kaggle, which contains pre-split train/test sets of multivariate
sensor data with anomalies for spacecraft health monitoring.

Data source: https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl
Original paper: "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding"
    (KDD 2018) by Hundman et al.

The dataset contains anonymized spacecraft telemetry data where each channel is a separate .npy file
with shape (n_timesteps, n_features). Features include sensor readings and one-hot encoded commands.
All values are pre-scaled to (-1, 1).

Run with CPython: python3 examples/datasets/smap_msl/download.py
"""

import os
import shutil
import urllib.request
import zipfile


# Kaggle dataset download URL (public API, no auth needed)
DATA_URL = "https://www.kaggle.com/api/v1/datasets/download/patrickfleith/nasa-anomaly-detection-dataset-smap-msl"

# Anomaly labels from the original telemanom repository
LABELS_URL = "https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv"


def download_file(url, path):
    """Download a file with progress indication."""
    if os.path.exists(path):
        print(f"  {path} already exists, skipping")
        return
    print(f"  Downloading {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        total = int(resp.headers.get("Content-Length", "unknown"))
        downloaded = 0
        buffer_size = 8192
        with open(path, "wb") as f:
            while True:
                chunk = resp.read(buffer_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total != "unknown":
                    pct = downloaded / total * 100
                    print(f"    {downloaded}/{total} ({pct:.1f}%)\r", end="")
        print()


def download(data_dir):
    """Download SMAP and MSL datasets."""

    # Download the main data archive (~86 MB)
    zip_path = os.path.join(data_dir, "data.zip")
    download_file(DATA_URL, zip_path)

    # Download anomaly labels
    labels_path = os.path.join(data_dir, "labeled_anomalies.csv")
    download_file(LABELS_URL, labels_path)

    # Extract the archive if not already extracted
    extracted_marker = os.path.join(data_dir, ".extracted")
    if not os.path.exists(extracted_marker):
        print(f"\nExtracting data.zip ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir)

        # Move the nested 'data/data' folder up one level to just 'data/'
        src = os.path.join(data_dir, "data", "data")
        dst = os.path.join(data_dir, "data_raw")
        if os.path.exists(src):
            shutil.move(src, dst)

        # Create marker file
        with open(extracted_marker, "w") as f:
            f.write("done\n")

        print(f"  Extracted to {data_dir}/data_raw/")


def list_channels(data_dir):
    """Print channel information for available data."""

    data_dir = os.path.join(data_dir, "data_raw")

    # List train/test channels
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
        print("Data not yet downloaded. Run: python3 download.py")
        return

    train_files = sorted(
        f.replace(".npy", "") for f in os.listdir(train_dir) if f.endswith(".npy")
    )
    test_files = sorted(
        f.replace(".npy", "") for f in os.listdir(test_dir) if f.endswith(".npy")
    )

    print(f"\nAvailable channels ({len(train_files)} train, {len(test_files)} test):")

    # Separate SMAP vs MSL from labels
    labels = {}
    labels_path = os.path.join(data_dir, "labeled_anomalies.csv")
    if os.path.exists(labels_path):
        import csv
        with open(labels_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                chan_id = row["chan_id"].strip()
                spacecraft = row["spacecraft"].strip()
                labels[chan_id] = spacecraft

    for ch in sorted(set(train_files)):
        sc = labels.get(ch, "?")
        print(f"  {ch} ({sc})")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    download(here)


if __name__ == "__main__":
    main()
