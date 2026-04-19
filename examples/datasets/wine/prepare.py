
import os.path

import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():

    here = os.path.dirname(__file__)
    out_dir = here

    wine = load_wine()
    X = wine.data.astype(np.float32)
    y = wine.target.astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    np.save(os.path.join(out_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(out_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(out_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(out_dir, 'y_test.npy'), y_test)

    print('Wrote to', out_dir)

if __name__ == '__main__':
    main()
