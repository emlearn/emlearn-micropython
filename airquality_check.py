
import array
import emlearn_plsr
import npyfile
import os
import os.path

def mean_squared_error(y_true, y_pred):
    n = len(y_true)
    return sum((yi - yi_hat) ** 2 for yi, yi_hat in zip(y_true, y_pred)) / n


def r2_score(y_true, y_pred):
    n = len(y_true)
    y_mean = sum(y_true) / n
    ss_tot = sum((yi - y_mean) ** 2 for yi in y_true)
    ss_res = sum((yi - yi_hat) ** 2 for yi, yi_hat in zip(y_true, y_pred))
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0


def load_data(data_dir):
    x_file = os.path.join(data_dir, "X.npy")
    y_file = os.path.join(data_dir, "y.npy")
    shape_X, X_array = npyfile.load(x_file)  # X_array is array.array('f')
    shape_y, y_array = npyfile.load(y_file)  # y_array is array.array('f')
    return shape_X, X_array, shape_y, y_array


def run_plsr_reference(data_dir, n_components=5):
    # -----------------------------
    # Load data
    # -----------------------------
    shape_X, X_array, shape_y, y_array = load_data(data_dir)
    n_samples = shape_y[0]
    n_features = shape_X[1]

    # -----------------------------
    # Create and train model
    # -----------------------------
    model = emlearn_plsr.new(n_samples, n_features, n_components)
    success = emlearn_plsr.fit(
        model, X_array, y_array,
        max_iterations=1000,
        tolerance=1e-5,
        verbose=0
    )

    print(success, model.is_complete())

    # -----------------------------
    # Compute predictions
    # -----------------------------
    y_pred = array.array('f')
    for i in range(n_samples):
        x_row = X_array[i * n_features:(i + 1) * n_features]
        y_val = model.predict(x_row)
        y_pred.append(y_val)

    # -----------------------------
    # Compute metrics
    # -----------------------------
    mse = mean_squared_error(y_array, y_pred)
    r2 = r2_score(y_array, y_pred)

    print(f"PLSR Reference Results (n_components={n_components}):")
    print(f"  MSE: {mse:.5f}")
    print(f"  R^2 score: {r2:.5f}")


if __name__ == "__main__":
    # Example usage: adjust data_dir as needed
    run_plsr_reference(data_dir="data", n_components=3)

    run_plsr_reference(data_dir="my_spectrofood_data_L1", n_components=10)


