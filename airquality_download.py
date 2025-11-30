
import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score


def download_dataset(url="https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip",
                     data_dir="data",
                     zip_name="AirQualityUCI.zip"):
    os.makedirs(data_dir, exist_ok=True)
    zip_file = os.path.join(data_dir, zip_name)
    if not os.path.exists(zip_file):
        print("Downloading dataset...")
        urllib.request.urlretrieve(url, zip_file)
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    return data_dir


def load_and_preprocess(csv_file=None, data_dir="data", feature_cols=None, target_col="CO(GT)"):
    if csv_file is None:
        csv_file = os.path.join(data_dir, "AirQualityUCI.csv")
    df = pd.read_csv(csv_file, sep=';', decimal=',')
    df = df.iloc[:, :-2]  # drop last two empty columns
    df.replace(-200, np.nan, inplace=True)
    df.dropna(inplace=True)

    if feature_cols is None:
        X = df.iloc[:, 2:].values.astype(np.float32)  # default all sensor columns
    else:
        X = df[feature_cols].values.astype(np.float32)

    y = df[target_col].values.astype(np.float32).reshape(-1, 1)

    scaler_X = StandardScaler()
    X_scaled = np.ascontiguousarray(scaler_X.fit_transform(X))

    scaler_y = StandardScaler()
    y_scaled = np.ascontiguousarray(scaler_y.fit_transform(y))

    np.save(os.path.join(data_dir, "X.npy"), X_scaled, allow_pickle=False)
    np.save(os.path.join(data_dir, "y.npy"), y_scaled, allow_pickle=False)
    return X_scaled, y_scaled, scaler_X, scaler_y


def train_and_evaluate(X, y, n_components=5, test_size=0.2, random_state=42, data_dir="data"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    pls = PLSRegression(n_components=n_components)
    pls.fit(X_train, y_train)
    y_pred = pls.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    np.save(os.path.join(data_dir, "pls_coef.npy"), pls.coef_)

    return mse, r2, pls


def load_numpy_data(data_dir="data"):
    X_loaded = np.load(os.path.join(data_dir, "X.npy"))
    y_loaded = np.load(os.path.join(data_dir, "y.npy"))
    return X_loaded, y_loaded


def main():
    n_components = 3
    data_dir = download_dataset()
    X, y, _, _ = load_and_preprocess(data_dir=data_dir)
    mse, r2, _ = train_and_evaluate(X, y, n_components=n_components, data_dir=data_dir)
    print(f"PLSR Reference Results (n_components={n_components}):")
    print(f"  MSE: {mse:.5f}")
    print(f"  R^2: {r2:.5f}")


if __name__ == "__main__":
    main()
