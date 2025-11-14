
import emlearn_linreg
import array

def test_linreg_train_trivial():

    model = emlearn_linreg.new(4, 0.1, 0.5, 0.01)

    # Training data (float32 arrays)
    X = array.array('f', [1,2,3,4, 2,3,4,5])  # flattened
    y = array.array('f', [10, 15])

    # Train
    emlearn_linreg.train(model, X, y, max_iterations=100, tolerance=1e-6, verbose=0)

    # Predict
    prediction = model.predict(array.array('f', [1,2,3,4]))

    error = prediction - 10.0
    assert abs(error) < 0.80, error

if __name__ == '__main__':
    test_linreg_train_trivial()
