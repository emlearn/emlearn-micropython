#!/usr/bin/env python3
"""Download and preprocess the SpectroFood dataset for PLS regression.

Also computes sklearn PLSR reference results for comparison.
Run with CPython: python3 examples/datasets/spectrofood/prepare.py
"""

from pathlib import Path
import os
import urllib.request
from io import StringIO

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error


DATA_URL = "https://zenodo.org/records/8362947/files/SpectroFood_dataset.csv?download=1"


def load_spectrofood_chunks(csv_file, target_col="DRY MATTER", food_col="food"):
    """
    Splits CSV into chunks using empty lines as separators.
    Returns list of tuples: (food_name, DataFrame)
    """
    chunks = []
    with open(csv_file, 'r') as f:
        content = f.read()

    raw_chunks = [c.strip() for c in content.split("\n\n") if c.strip()]

    for chunk_text in raw_chunks:
        chunk_io = StringIO(chunk_text)
        try:
            df_chunk = pd.read_csv(chunk_io, dtype=str, keep_default_na=False)
        except pd.errors.EmptyDataError:
            continue

        if food_col in df_chunk.columns:
            food_name = df_chunk[food_col].iloc[0].strip().replace(" ", "_")
        else:
            food_name = str(df_chunk.iloc[0, 0]).strip().replace(" ", "_")

        df_chunk = df_chunk.apply(pd.to_numeric, errors='coerce')
        chunks.append((food_name, df_chunk))

    return chunks


def preprocess_chunk(df_chunk, target_col="DRY MATTER"):
    """Convert DataFrame to X and y numpy arrays."""
    df_chunk = df_chunk[pd.to_numeric(df_chunk[target_col], errors='coerce').notna()].copy()
    df_chunk = df_chunk.dropna(axis=1, how='all')
    df_chunk = df_chunk.dropna(axis=0, how='any')

    exclude_cols = [c for c in df_chunk.columns if c == target_col or df_chunk[c].dtype == object]
    X = df_chunk.drop(columns=exclude_cols).values.astype(np.float32)
    y = df_chunk[target_col].values.astype(np.float32)

    scaler_X = StandardScaler()
    X = np.ascontiguousarray(scaler_X.fit_transform(X))

    return X, y


def main():

    here = os.path.dirname(__file__)
    OUTPUT_DIR = Path(here)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Download
    csv_path = OUTPUT_DIR / "SpectroFood_dataset.csv"
    if not csv_path.exists():
        print("Downloading SpectroFood dataset...")
        urllib.request.urlretrieve(DATA_URL, csv_path)

    # Load and preprocess
    chunks = load_spectrofood_chunks(str(csv_path))
    print(f"Found {len(chunks)} food types")

    for food_name, df in chunks:
        X, y = preprocess_chunk(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0
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
        for nc in [5, 10, 15]:
            if nc > min(X_train.shape):
                continue
            pls = PLSRegression(n_components=nc)
            pls.fit(X_train, y_train)
            y_pred = pls.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            print(f"  n_components={nc}: MSE={mse:.5f}, R^2={r2:.5f}")


if __name__ == '__main__':
    main()
