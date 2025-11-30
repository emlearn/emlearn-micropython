import os
import pandas as pd
import numpy as np
import urllib.request
from io import StringIO

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score


DATA_URL = "https://zenodo.org/records/8362947/files/SpectroFood_dataset.csv?download=1"

def download_dataset(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    csv_file = os.path.join(data_dir, "SpectroFood_dataset.csv")
    if not os.path.exists(csv_file):
        print("Downloading SpectroFood CSV...")
        urllib.request.urlretrieve(DATA_URL, csv_file)
    return csv_file


def load_spectrofood_chunks(csv_file, target_col="dry_matter", food_col="food"):
    """
    Splits CSV into chunks using empty lines (newlines) as separators.
    Each chunk is loaded with pandas.read_csv separately.
    Returns list of tuples: (food_name, DataFrame)
    """
    chunks = []
    with open(csv_file, 'r') as f:
        content = f.read()

    # Split into raw text blocks on empty lines
    raw_chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
    # FIXME: only returns 1 chunk right now
    print(len(raw_chunks))

    for chunk_text in raw_chunks:
        # Use StringIO to read the chunk as CSV
        chunk_io = StringIO(chunk_text)
        try:
            df_chunk = pd.read_csv(chunk_io, dtype=str, keep_default_na=False)
        except pd.errors.EmptyDataError:
            continue  # skip empty chunks

        # Determine food name: use the first column of the first row
        if food_col in df_chunk.columns:
            food_name = df_chunk[food_col].iloc[0].strip().replace(" ", "_")
        else:
            food_name = str(df_chunk.iloc[0, 0]).strip().replace(" ", "_")

        # Convert numeric columns to float, ignore errors
        df_chunk = df_chunk.apply(pd.to_numeric, errors='coerce')
        chunks.append((food_name, df_chunk))

    return chunks

def preprocess_chunk(df_chunk, target_col="DRY MATTER"):
    """
    Converts DataFrame to C-contiguous X and y numpy arrays
    """

    #print(df_chunk.columns)

    # Keep only rows where the target column is numeric
    df_chunk = df_chunk[pd.to_numeric(df_chunk[target_col], errors='coerce').notna()].copy()

    # Drop columns that are entirely NaN
    df_chunk = df_chunk.dropna(axis=1, how='all')

    # Drop rows that are entirely NaN
    df_chunk = df_chunk.dropna(axis=0, how='any')

    exclude_cols = [c for c in df_chunk.columns if c == target_col or df_chunk[c].dtype == object]
    X = df_chunk.drop(columns=exclude_cols).values.astype(np.float32)
    y = df_chunk[target_col].values.astype(np.float32).reshape(-1, 1)

    # Standardize
    scaler_X = StandardScaler()
    #scaler_y = StandardScaler()
    X = scaler_X.fit_transform(X)
    #y = scaler_y.fit_transform(y)

    X = np.ascontiguousarray(X)
    y = np.ascontiguousarray(y)
    return X, y

def save_all_chunks(chunks, data_dir):
    """
    Saves all chunks as numpy files
    """
    for food_name, df in chunks:
        X, y = preprocess_chunk(df)
        dataset_dir = data_dir+f'_{food_name}'
        os.makedirs(dataset_dir, exist_ok=True)
        np.save(os.path.join(dataset_dir, f"X.npy"), X)
        np.save(os.path.join(dataset_dir, f"y.npy"), y)
        print(f"Saved chunk for {food_name}: {dataset_dir}")

def train_pls_for_chunks(chunks, n_components=10):
    """
    Trains a scikit-learn PLSRegression model for each chunk
    and prints MSE and R2
    """
    for food_name, df in chunks:
        X, y = preprocess_chunk(df)
        # Split 80/20
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Train PLS
        pls = PLSRegression(n_components=n_components)
        pls.fit(X_train, y_train)

        # Predict and inverse scale
        y_pred = pls.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"{food_name}: PLSRegression n_components={n_components} | MSE={np.sqrt(mse):.4f} | R2={r2:.4f}")

def main(data_dir="spectrofood_data"):
    csv_file = download_dataset(data_dir)
    chunks = load_spectrofood_chunks(csv_file)
    print(f"Found {len(chunks)} chunks (food types)")
    save_all_chunks(chunks, data_dir)
    train_pls_for_chunks(chunks, n_components=5)

if __name__ == "__main__":
    main(data_dir="my_spectrofood_data")

