#!/usr/bin/env python3
"""Download and preprocess the Breast Cancer Wisconsin dataset."""

from pathlib import Path
import os

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():

    here = os.path.dirname(__file__)

    OUTPUT_DIR = Path(here)
    OUTPUT_DIR.mkdir(exist_ok=True)

    FILENAMES = {
        'X_train': OUTPUT_DIR / 'X_train.npy',
        'X_test': OUTPUT_DIR / 'X_test.npy',
        'y_train': OUTPUT_DIR / 'y_train.npy',
        'y_test': OUTPUT_DIR / 'y_test.npy',
    }

    X, y = load_breast_cancer(return_X_y=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype('float32')
    y = y.astype('float32')

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    np.save(FILENAMES['X_train'], X_train)
    np.save(FILENAMES['X_test'], X_test)
    np.save(FILENAMES['y_train'], y_train)
    np.save(FILENAMES['y_test'], y_test)

    print('Saved datasets:')
    print(f"  X_train: {X_train.shape} -> {FILENAMES['X_train']}")
    print(f"  X_test : {X_test.shape} -> {FILENAMES['X_test']}")
    print(f"  y_train: {y_train.shape} -> {FILENAMES['y_train']}")
    print(f"  y_test : {y_test.shape} -> {FILENAMES['y_test']}")


if __name__ == '__main__':
    main()
