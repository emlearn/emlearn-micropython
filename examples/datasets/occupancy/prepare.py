#!/usr/bin/env python3
"""
Download and preprocess the UCI Occupancy Detection dataset.

Binary classification: predict room occupancy from environmental sensors.
Features: Temperature, Humidity, Light, CO2, HumidityRatio.

Dataset: https://archive.ics.uci.edu/dataset/357/occupancy+detection
"""

import os
import csv
import urllib.request
import zipfile
import io

import numpy as np
from sklearn.preprocessing import StandardScaler


def download_and_extract(data_dir):
    """Download the UCI Occupancy Detection zip and return paths to CSV files."""
    url = "https://archive.ics.uci.edu/static/public/357/occupancy+detection.zip"
    zip_path = os.path.join(data_dir, "occupancy+detection.zip")

    if not os.path.exists(zip_path):
        print(f"Downloading from {url} ...")
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete.")
    else:
        print("Zip already exists, reusing.")

    # Extract CSV files
    csv_files = {}
    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            if name.endswith(".txt"):
                out = os.path.join(data_dir, name)
                if not os.path.exists(out):
                    z.extract(name, data_dir)
                csv_files[name] = out
                print(f"  {name} -> {out}")

    return csv_files


def load_csv(filepath):
    """Load a UCI occupancy CSV, returning X (features) and y (labels).

    The CSV has 7 header columns but 8 data values per row.
    The extra value is an index followed by a timestamp; both are dropped.
    Feature columns: Temperature, Humidity, Light, CO2, HumidityRatio
    Target column: Occupancy (0=not occupied, 1=occupied)
    """
    X_rows = []
    y_rows = []
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
        for row in reader:
            # row[0] = row index (int), row[1] = timestamp (str)
            # row[2..6] = features (Temperature, Humidity, Light, CO2, HumidityRatio)
            # row[7] = Occupancy
            feats = [float(v) for v in row[2:7]]
            label = int(row[7])
            X_rows.append(feats)
            y_rows.append(label)

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.float32)
    return X, y


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = here

    # Download and extract
    csv_files = download_and_extract(data_dir)

    # Load predefined train/test splits
    train_file = csv_files["datatraining.txt"]
    test_file = csv_files["datatest.txt"]

    print(f"\nLoading training data from {train_file} ...")
    X_train, y_train = load_csv(train_file)
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  Target distribution: {np.bincount(y_train.astype(int))}")

    print(f"\nLoading test data from {test_file} ...")
    X_test, y_test = load_csv(test_file)
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
    print(f"  Target distribution: {np.bincount(y_test.astype(int))}")

    # Scale features using training statistics
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    print(f"\nScaling:")
    print(f"  Feature means: {scaler.mean_}")
    print(f"  Feature stds:  {scaler.scale_}")

    # Save as .npy files
    np.save(os.path.join(data_dir, "X_train.npy"), X_train)
    np.save(os.path.join(data_dir, "X_test.npy"), X_test)
    np.save(os.path.join(data_dir, "y_train.npy"), y_train)
    np.save(os.path.join(data_dir, "y_test.npy"), y_test)

    print(f"\nSaved files:")
    print(f"  X_train.npy: {X_train.shape} float32")
    print(f"  X_test.npy:  {X_test.shape} float32")
    print(f"  y_train.npy: {y_train.shape} float32")
    print(f"  y_test.npy:  {y_test.shape} float32")

    print(f"\nData statistics:")
    print(f"  X_train range: [{X_train.min():.4f}, {X_train.max():.4f}]")
    print(f"  y_train  0/1: {np.bincount(y_train.astype(int))}")
    print(f"  y_test   0/1: {np.bincount(y_test.astype(int))}")


if __name__ == "__main__":
    main()
