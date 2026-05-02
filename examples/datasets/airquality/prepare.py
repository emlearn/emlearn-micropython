#!/usr/bin/env python3
"""Download and preprocess the Air Quality UCI dataset for PLS regression.

Also computes sklearn PLSR reference results for comparison.
Run with CPython: python3 examples/datasets/airquality/prepare.py
"""

from pathlib import Path
import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error


def main():

    here = os.path.dirname(__file__)
    OUTPUT_DIR = Path(here)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Download
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip"
    zip_path = OUTPUT_DIR / "AirQualityUCI.zip"
    if not zip_path.exists():
        print("Downloading Air Quality UCI dataset...")
        urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(OUTPUT_DIR)

    # Load and preprocess
    csv_file = OUTPUT_DIR / "AirQualityUCI.csv"
    df = pd.read_csv(csv_file, sep=';', decimal=',')
    df = df.iloc[:, :-2]  # drop last two empty columns
    df.replace(-200, np.nan, inplace=True)
    df.dropna(inplace=True)

    X = df.iloc[:, 2:].values.astype(np.float32)  # sensor columns
    y = df["CO(GT)"].values.astype(np.float32)

    scaler_X = StandardScaler()
    X = scaler_X.fit_transform(X).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    FILENAMES = {
        'X_train': OUTPUT_DIR / 'X_train.npy',
        'X_test': OUTPUT_DIR / 'X_test.npy',
        'y_train': OUTPUT_DIR / 'y_train.npy',
        'y_test': OUTPUT_DIR / 'y_test.npy',
    }

    np.save(FILENAMES['X_train'], X_train)
    np.save(FILENAMES['X_test'], X_test)
    np.save(FILENAMES['y_train'], y_train)
    np.save(FILENAMES['y_test'], y_test)

    print('Saved datasets:')
    print(f"  X_train: {X_train.shape} -> {FILENAMES['X_train']}")
    print(f"  X_test : {X_test.shape} -> {FILENAMES['X_test']}")
    print(f"  y_train: {y_train.shape} -> {FILENAMES['y_train']}")
    print(f"  y_test : {y_test.shape} -> {FILENAMES['y_test']}")

    # Sklearn PLSR reference results
    print('\nSklearn PLSR reference:')
    for nc in [3, 5]:
        pls = PLSRegression(n_components=nc)
        pls.fit(X_train, y_train)
        y_pred = pls.predict(X_test).ravel()
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"  n_components={nc}: MSE={mse:.5f}, R^2={r2:.5f}")


if __name__ == '__main__':
    main()
