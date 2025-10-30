import os

import pandas
import numpy
import emlearn
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from plotting import plot_timeline, find_runs, configure_xaxis
from plotting import make_label_colors, make_timeline_plot
from processing import process_data, make_label_track, convert_to_raw


colors = {
    'Nordic_walking': 'rgb(251,128,114)',
    'cycling': 'rgb(179,222,105)',
    'walking': 'rgb(253,180,98)',
    'transient': 'rgb(217,217,217)',
    'running': 'rgb(255,237,111)',
    'rope_jumping': '#2E91E5',
    'lying': 'rgb(255,255,179)',
    'sitting': 'rgb(190,186,218)',
    'standing': 'rgb(251,128,114)',
    'ironing': 'rgb(128,177,211)',
    'vacuum_cleaning': 'rgb(253,180,98)',
    'ascending_stairs': 'rgb(179,222,105)',
    'descending_stairs': 'rgb(252,205,229)',
}

sensor_columns = [
    'hand_acceleration_6g_x',
    'hand_acceleration_6g_y',
    'hand_acceleration_6g_z',
]

def f1_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')

def train_and_run(data : pandas.DataFrame, class_columns, label_column='activity'):
    """
    Super simplified training for illustration purposes.

    For a real example, see har_train.py in har_trees
    """

    events = find_runs(data[label_column])
    sensor_data = convert_to_raw(data[sensor_columns])
    features = process_data(sensor_data)

    labels = make_label_track(features.index, events)
    combined = pandas.merge(features, sensor_data, left_index=True, right_index=True)
    combined['activity'] = labels

    estimator = RandomForestClassifier(n_estimators=10, min_samples_leaf=10)
    feature_columns = features.columns
    X = combined[feature_columns]
    Y = combined[label_column]

    # Train on entire dataset (no test set or cross-validation)
    # XXX: Not acceptable in real model, but here we are just looking to illustrate
    estimator.fit(X, Y)

    # Convert model with emlearn
    model_path = 'model.csv'
    converted = emlearn.convert(estimator)
    converted.save(file=model_path, format='csv')
    print('Exported', model_path)

    class_names = estimator.classes_
    ref_preds = pandas.DataFrame(estimator.predict_proba(X), index=X.index, columns=estimator.classes_)
    ref_score = f1_macro(Y, class_names[numpy.argmax(ref_preds, axis=1)])

    # Run model predictions with emlearn
    out = process_data(sensor_data, model_path=model_path, classes=list(estimator.classes_))
    preds = out[estimator.classes_]
    c_score = f1_macro(Y, class_names[numpy.argmax(preds, axis=1)])

    print('test scores', ref_score, c_score)

    # Apply a bit of temporal smoothing on predictions
    rolling_window = 5
    smooth = preds[class_columns].rolling(rolling_window).median()
    smooth['activity'] =  make_label_track(smooth.index, events)

    return combined, smooth, estimator


def main():

    data_path = 'pamap2_subject102_exerpt.parquet'
    plot_path = 'physical_activity_recognition_pipeline.png'

    sub = pandas.read_parquet(data_path)

    class_columns = [
        'Nordic_walking',
        'cycling',
        'walking', 
        #'rope_jumping',
        #'running',
        'transient',
    ]

    # Fit model onto this data and run pipeline
    combined, smooth, estimator = train_and_run(sub, class_columns)

    # Make a plot
    width = 1600
    aspect = 2.0
    height = width/aspect
    fig = make_timeline_plot(sub, combined, smooth, colors, class_names=class_columns, width=width, aspect=aspect)

    fig.write_image(plot_path, scale=1.5, width=width, height=height)
    print('Wrote plot', plot_path)


if __name__  == '__main__':
    main()


