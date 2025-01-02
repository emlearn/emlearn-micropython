
import os
import time
import uuid
import pickle
import tempfile
import subprocess
import json

import pandas
import numpy
import structlog

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer, get_scorer, PrecisionRecallDisplay
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit
from matplotlib import pyplot as plt

import emlearn
from emlearn.preprocessing.quantizer import Quantizer
from sklearn.preprocessing import LabelEncoder

log = structlog.get_logger()


def train_test_split_grouped(X, y, groups, test_size=0.25, random_state=0):

    gs = GroupShuffleSplit(n_splits=2, test_size=test_size, random_state=random_state)
    train_ix, test_ix = next(gs.split(X, y, groups=groups))

    X_train = X.iloc[train_ix]
    X_test = X.iloc[test_ix]
    Y_train = y.iloc[train_ix]
    Y_test = y.iloc[test_ix]

    return X_train, X_test, Y_train, Y_test

def evaluate(windows : pandas.DataFrame, groupby : str, hyperparameters : dict,
    random_state=1, n_splits=5, label_column='activity'):

    # Setup subject-based cross validation
    splitter = GroupShuffleSplit(n_splits=n_splits, test_size=0.25, random_state=random_state)

    clf = RandomForestClassifier(random_state = random_state, n_jobs=1, class_weight = "balanced")

    f1_micro = 'f1_micro'

    search = GridSearchCV(
        clf,
        param_grid=hyperparameters,
        scoring={
            # our predictive model metric
            'f1_micro': f1_micro,
        },
        refit='f1_micro',
        cv=splitter,
        return_train_score=True,
        n_jobs=4,
        verbose=2,
    )

    feature_columns = sorted(set(windows.columns) - set([label_column, groupby]))
    assert 'subject' not in feature_columns
    assert 'activity' not in feature_columns
    assert 'file' not in feature_columns
    X = windows[feature_columns]
    Y = windows[label_column]
    groups = windows.index.get_level_values(groupby)

    # Split out test set
    X_train, X_test, Y_train, Y_test = train_test_split_grouped(X, Y, groups=groups, random_state=random_state)
    test_groups = list(X_test.index.get_level_values(groupby).unique())
    groups_train = X_train.index.get_level_values(groupby)
    train_groups = list(groups_train.unique())

    # Run hyperparameter search using cross-validation
    search.fit(X_train, Y_train, groups=groups_train)
    cv_results = pandas.DataFrame(search.cv_results_)
    estimator = search.best_estimator_

    # Evaluate final estimator on train/test sets
    scorer = get_scorer(f1_micro)
    test_score = scorer(estimator, X_test, Y_test)
    train_score = scorer(estimator, X_train, Y_train)

    figures = dict()

    if len(estimator.classes_) == 2:
        # Binary classification, compute precision-recall curves
        fig, ax = plt.subplots(1, figsize=(10, 10))
        PrecisionRecallDisplay.from_estimator(estimator, X_train, Y_train, ax=ax, name='train')
        PrecisionRecallDisplay.from_estimator(estimator, X_test, Y_test, ax=ax, name='test')
        figures['precision_recall'] = fig

    splits = (train_groups, test_groups)
    scores = (train_score, test_score)
    return cv_results, estimator, splits, scores, figures


def extract_windows(sensordata : pandas.DataFrame,
    window_length : int,
    window_hop : int,
    groupby : list[str],
    time_column = 'time',
    ):

    groups = sensordata.groupby(groupby, observed=True)


    for group_idx, group_df in groups:

        windows = []

        # make sure order is correct
        group_df = group_df.reset_index().set_index(time_column).sort_index()

        # create windows
        win_start = 0
        length = len(group_df)
        while win_start < length:
            win_end = win_start + window_length
            # ignore partial window at the end
            if win_end > length:
                break
            
            win = group_df.iloc[win_start:win_end].copy()
            win['window'] = win.index[0]
            assert len(win) == window_length, (len(win), window_length)
   
            windows.append(win)

            win_start += window_hop

        yield windows

def assign_window_label(labels, majority=0.66):
    """
    Assign the most common label to window, if it is above the @majority threshold
    Otherwise return NA
    """
    counts = labels.value_counts(dropna=True)
    threshold = majority * len(labels)
    if counts.iloc[0] > threshold:
        return counts.index[0]
    else:
        return None


def timebased_features(windows : list[pandas.DataFrame],
        columns : list[str],
        micropython_bin='micropython') -> pandas.DataFrame:

    #print('w', len(windows), columns)

    here = os.path.dirname(__file__)
    feature_extraction_script = os.path.join(here, 'compute_features.py')

    data = numpy.stack([ d[columns] for d in windows ])
    assert data.dtype == numpy.int16
    assert data.shape[2] == 3, data.shape

    #log.debug('data-range',
    #    upper=numpy.quantile(data, 0.99),
    #    lower=numpy.quantile(data, 0.01),
    #)

    with tempfile.TemporaryDirectory() as tempdir:
        data_path = os.path.join(tempdir, 'data.npy')
        features_path = os.path.join(tempdir, 'features.npy')

        # Persist output
        numpy.save(data_path, data)

        # Run MicroPython program
        args = [
            micropython_bin,
            feature_extraction_script,
            data_path,
            features_path,
        ]
        cmd = ' '.join(args)
        #log.debug('run-micropython', cmd=cmd)
        try:
            out = subprocess.check_output(args)
        except subprocess.CalledProcessError as e:
            log.error('micropython-error', out=e.stdout, err=e.stderr)
            raise e

        # Load output
        out = numpy.load(features_path)
        assert len(out) == len(data)

        # TODO: add feature names
        df = pandas.DataFrame(out)

    return df

def extract_features(sensordata : pandas.DataFrame,
    columns : list[str],
    groupby,
    window_length = 128,
    window_hop = 64,
    features='timebased',
    quant_div = 4,
    quant_depth = 6,
    sensitivity = 2.0, # how many g range the int16 sensor data has
    label_column='activity',
    time_column='time',
    ) -> pandas.DataFrame:
    """
    Convert sensor data into fixed-sized time windows and extact features
    """

    if features == 'quant':
        raise NotImplementedError
    elif features == 'timebased':
        feature_extractor = lambda w: timebased_features(w, columns=columns)
    else:
        raise ValueError(f"Unsupported features: {features}")

    # Split into fixed-length windows
    features_values = []
    generator = extract_windows(sensordata, window_length, window_hop, groupby=groupby, time_column=time_column)
    for windows in generator:
    
        # drop invalid data
        windows = [ w for w in windows if not w[columns].isnull().values.any() ]

        # Convert from floats in "g" to the sensor scaling in int16
        data_windows = [ ((w[columns] / sensitivity) * (2**15-1)).astype(numpy.int16) for w in windows ]

        # Extract features
        df = feature_extractor(data_windows)

        # Convert features to 16-bit integers
        # XXX: Assuming that they are already in resonable scale
        # TODO: consider moving the quantization to inside timebased
        quant = df.values.astype(numpy.int16)
        df.loc[:,:] = quant

        # Attach labels
        df[label_column] = [ assign_window_label(w[label_column]) for w in windows ]

        # Combine with identifying information
        index_columns = list(groupby + ['window'])
        for idx_column in index_columns:
            df[idx_column] = [w[idx_column].iloc[0] for w in windows]
        df = df.set_index(index_columns)

        features_values.append(df)

    out = pandas.concat(features_values)
    return out

def export_model(path, out):

    with open(path, "rb") as f:
        classifier = pickle.load(f)

        classes = classifier.classes_
        class_mapping = dict(zip(classes, range(len(classifier.classes_))))

        cmodel = emlearn.convert(classifier)
        cmodel.save(name='harmodel', format='csv', file=out)


def run_pipeline(run, hyperparameters, dataset,
        data_dir,
        out_dir,
        model_settings=dict(),
        n_splits=5,
        features='timebased',
    ):

    dataset_config = {
        'uci_har': dict(
            groups=['subject', 'experiment'],
            data_columns = ['acc_x', 'acc_y', 'acc_z'],
            classes = [
                #'STAND_TO_LIE',
                #'SIT_TO_LIE',
                #'LIE_TO_SIT',
                #'STAND_TO_SIT',
                #'LIE_TO_STAND',
                #'SIT_TO_STAND',
                'STANDING', 'LAYING', 'SITTING',
                'WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS',
            ],
        ),
        'pamap2': dict(
            groups=['subject'],
            data_columns = ['hand_acceleration_16g_x', 'hand_acceleration_16g_y', 'hand_acceleration_16g_z'],
            classes = [
                #'transient',
                'walking', 'ironing', 'lying', 'standing',
                'Nordic_walking', 'sitting', 'vacuum_cleaning',
                'cycling', 'ascending_stairs', 'descending_stairs',
                'running', 'rope_jumping',
            ],
        ),
        'har_exercise_1': dict(
            groups=['file'],
            data_columns = ['x', 'y', 'z'],
            classes = [
                #'mixed',
                'squat', 'jumpingjack', 'lunge', 'other',
            ],
        ),
        'toothbrush_hussain2021': dict(
            groups=['subject'],
            label_column = 'is_brushing',
            time_column = 'elapsed',
            data_columns = ['acc_x', 'acc_y', 'acc_z'],
            #data_columns = ['gravity_x', 'gravity_y', 'gravity_z'],
            #data_columns = ['motion_x', 'motion_y', 'motion_z'],
            classes = [
                #'mixed',
                'True', 'False',
            ],
        ),
    }

    if not dataset in dataset_config.keys():
        raise ValueError(f"Unknown dataset {dataset}")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    data_path = os.path.join(data_dir, f'{dataset}.parquet')

    data_load_start = time.time()
    data = pandas.read_parquet(data_path)

    #print(data.index.names)
    #print(data.columns)

    groups = dataset_config[dataset]['groups']
    data_columns = dataset_config[dataset]['data_columns']
    enabled_classes = dataset_config[dataset]['classes']
    label_column = dataset_config[dataset].get('label_column', 'activity')
    time_column = dataset_config[dataset].get('time_column', 'time')
    sensitivity = dataset_config[dataset].get('sensitivity', 2.0)

    data[label_column] = data[label_column].astype(str)

    data_load_duration = time.time() - data_load_start
    log.info('data-loaded', dataset=dataset, samples=len(data), duration=data_load_duration)

    feature_extraction_start = time.time()
    features = extract_features(data,
        columns=data_columns,
        groupby=groups,             
        features=features,
        sensitivity=sensitivity,
        window_length=model_settings['window_length'],
        window_hop=model_settings['window_hop'],
        label_column=label_column,
        time_column=time_column,
    )
    labeled = numpy.count_nonzero(features[label_column].notna())

    feature_extraction_duration = time.time() - feature_extraction_start
    log.info('feature-extraction-done',
        dataset=dataset,
        total_instances=len(features),
        labeled_instances=labeled,
        duration=feature_extraction_duration,
    )

    # Drop windows without labels
    features = features[features[label_column].notna()]

    # Keep only windows with enabled classes
    features = features[features[label_column].isin(enabled_classes)]

    print('Class distribution\n', features[label_column].value_counts(dropna=False))

    # Run train-evaluate
    evaluate_groupby = groups[0]
    results, estimator, splits, scores, figures = evaluate(features,
        hyperparameters=hyperparameters,
        groupby=evaluate_groupby,
        n_splits=n_splits,
        label_column=label_column,
    )

    print('Train-test splits')
    for split, score, groups in zip(['train', 'test'], scores, splits):
        print(split, score, groups)
    
    # Save eval plots
    for name, fig in figures.items():
        p = os.path.join(out_dir, f'{dataset}.{name}.png')
        fig.savefig(p)
        print('Figure:', p)

    # Save a model
    estimator_path = os.path.join(out_dir, f'{dataset}.estimator.pickle')
    with open(estimator_path, 'wb') as f:
        pickle.dump(estimator, file=f)

    # Export model with emlearn
    model_path = os.path.join(out_dir, f'{dataset}_trees.csv')
    export_model(estimator_path, model_path)

    # Save testdata
    classes = estimator.classes_
    class_mapping = dict(zip(classes, range(len(classes))))
    meta_path = os.path.join(out_dir, f'{dataset}.meta.json')    
    with open(meta_path, 'w') as f:
        f.write(json.dumps(class_mapping))

    testdata_path = os.path.join(out_dir, f'{dataset}.testdata.npz')
    testdata = features.groupby(label_column, as_index=False).sample(n=10)
    # convert to class number/index
    testdata['class'] = testdata[label_column].map(class_mapping)
    feature_columns = sorted(set(testdata.columns) - set([label_column, 'class']))
    numpy.savez(testdata_path,
        X=numpy.ascontiguousarray(testdata[feature_columns].astype(numpy.int16)),
        Y=numpy.ascontiguousarray(testdata['class'].astype(numpy.int8)),
    )

    # Save results
    results['dataset'] = dataset
    results['run'] = run
    results_path = os.path.join(out_dir, f'{dataset}.results.parquet')
    results.to_parquet(results_path)
    print('Results written to', results_path)

    return results

def autoparse_number(s):
    if '.' in s:
        return float(s)
    else:
        return int(s)

def config_number_list(var : str, default : str, delim=',') -> list[int]:

    s = os.environ.get(var, default)
    tok = s.split(delim)
    values = [ autoparse_number(v.strip()) for v in tok if v.strip() ] 

    return values

def parse():
    import argparse
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--dataset', type=str, default='uci_har',
                        help='Which dataset to use')
    parser.add_argument('--data-dir', metavar='DIRECTORY', type=str, default='./data/processed',
                        help='Where the input data is stored')
    parser.add_argument('--out-dir', metavar='DIRECTORY', type=str, default='./',
                        help='Where to store results')

    parser.add_argument('--features', type=str, default='timebased',
                        help='Which feature-set to use')
    parser.add_argument('--window-length', type=int, default=128,
                        help='Length of each window to classify (in samples)')
    parser.add_argument('--window-hop', type=int, default=64,
                        help='How far to hop for next window to classify (in samples)')

    args = parser.parse_args()

    return args


def main():

    args = parse()
    dataset = args.dataset
    out_dir = args.out_dir
    data_dir = args.data_dir

    run_id = uuid.uuid4().hex.upper()[0:6]

    min_samples_leaf = config_number_list('MIN_SAMPLES_LEAF', '1,4,16,64,256')
    max_leaf_nodes = config_number_list('MAX_LEAF_NODES', '4,8,16,32,64,128')
    trees = config_number_list('TREES', '10')

    hyperparameters = {
        "max_features": [ 0.30 ],
        'n_estimators': trees,
        'min_samples_leaf': min_samples_leaf,
        #'max_leaf_nodes': max_leaf_nodes,
    }

    results = run_pipeline(dataset=args.dataset,
        out_dir=args.out_dir,
        data_dir=args.data_dir,
        run=run_id,
        hyperparameters=hyperparameters,
        model_settings=dict(
            window_hop=args.window_hop,
            window_length=args.window_length,
        ),
        n_splits=int(os.environ.get('FOLDS', '5')),
        features=args.features,
    )

    df = results.rename(columns=lambda c: c.replace('param_', ''))
    display_columns = [
        'n_estimators',
        'min_samples_leaf',
        'mean_train_f1_micro',
        'mean_test_f1_micro',
    ]

    print('Results\n', df[display_columns])

if __name__ == '__main__':
    main()
