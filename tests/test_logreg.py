import array
import emlearn_logreg


def make_dataset():
    # simple AND gate dataset
    X = array.array('f', [
        0, 0,
        0, 1,
        1, 0,
        1, 1,
    ])
    y = array.array('f', [0, 0, 0, 1])
    return X, y


def test_logreg_train_and_predict():
    X, y = make_dataset()
    model = emlearn_logreg.new(2, 0.5, 0.0, 0.0)

    emlearn_logreg.train(model, X, y, max_iterations=400, tolerance=1e-5,
                         check_interval=5, batch_size=2)

    pred = model.predict(array.array('f', [1, 1]))
    assert pred > 0.8, pred

    zero_pred = model.predict(array.array('f', [0, 0]))
    assert zero_pred < 0.2, zero_pred


if __name__ == '__main__':
    test_logreg_train_and_predict()
