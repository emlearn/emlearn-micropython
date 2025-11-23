
import os
import time
import uuid
import pickle
import tempfile
import subprocess
import json
import itertools
from typing import Optional

import yaml
import pandas
import numpy
import structlog
import joblib

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



class DataProcessorProgram():

    def __init__(self, program : list[str],
            options : dict = {},
            input_option : str = '--input',
            output_option : str = '--output',
            serialization : str = 'csv',
            timeout : float = 10.0,
            column_order : Optional[list[str]] = None,
        ):

        self.program = program
        self.options = options
        self.input_option = input_option
        self.output_option = output_option
        self.serialization = serialization
        self.timeout = timeout
        self.column_order = column_order

        supported = set(['npy', 'csv'])
        if not serialization in supported:
            raise ValueError(f'Unsupported serialization {serialization}. Valid: {supported}')


    def process(self, data : pandas.DataFrame) -> pandas.DataFrame:
        """
        Run a program (executable) to compute features

        Takes a DataFrame with sensor data as input.
        The sensor data should be continous and regular in time.
   
        Returns windows with features.
        The windows are usually overlapping in time.
        """


        extension = self.serialization

        mod = data.copy()
        if self.column_order is not None:
            mod = mod.reset_index()
            mod = mod[self.column_order]

        with tempfile.TemporaryDirectory() as tempdir:
            #tempdir = 'temp' # XXX, debug

            data_path = os.path.join(tempdir, f'data.{extension}')
            features_path = os.path.join(tempdir, f'features.{extension}')

            # Persist the data
            if self.serialization == 'npy':
                # make sure C order, and non-sparse
                arr = numpy.ascontiguousarray(mod)
                arr = arr.astype(numpy.int16)
                numpy.save(data_path, arr)
            elif self.serialization == 'csv':
                mod.to_csv(data_path, index=False)
            else:
                raise NotImplementedError(self.serialization)

            # Build arguments
            args = [] + self.program

            # Input and output
            if self.input_option:
                 args += [ self.input_option, data_path ]
            else:
                 args += [ data_path ]

            if self.output_option:
                args += [ self.output_option, features_path ]
            else:
                args += [ features_path ]

            # Other options
            for k, v in self.options.items():
                args += [ f'--{k}', str(v) ]

            cmd = ' '.join(args)
            try:
                out = subprocess.check_output(args, timeout=self.timeout)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                code = getattr(e, 'returncode', None)
                log.error('preprocessor-error',
                    cmd=cmd, out=e.stdout, err=e.stderr, code=code)
                raise e

            log.debug('preprocessor-run',
                cmd=cmd,
                #out=out,
            )

            # Load output
            if self.serialization == 'npy':
                # TODO: support feature names. Separat output file, with --features
                out = numpy.load(features_path)
                windows = pandas.DataFrame(out)
                # FIXME: support reading times, not infer
                span = (data.index.max() - data.index.min()).total_seconds()
                dt = span / len(windows)
                log.debug('preprocess', windows=len(windows), dt=dt)
                windows['time'] = dt * numpy.arange(len(windows))
            elif self.serialization == 'csv':
                windows = pandas.read_csv(features_path)
                span = (data.index.max() - data.index.min()).total_seconds()
                dt = span / len(windows)
                log.debug('preprocess', windows=len(windows), dt=dt)
            else:
                raise NotImplementedError(self.serialization)

        windows['time'] = pandas.to_timedelta(windows['time'], unit='s')

        # post-conditions
        time_in = data.index
        time_out = windows['time']

        window_duration = pandas.Timedelta(4.0, unit='s') # XXX: hardcoded
        start_delta = time_out.min() - time_in.min()
        assert abs(start_delta) <= window_duration, (start_delta, time_out.min(), time_in.min())
        end_delta = time_out.max() - time_in.max()
        assert abs(end_delta) <= window_duration, (end_delta, time_out.max(), time_in.max())

        return windows

class TimebasedFeatureExtractor(DataProcessorProgram):

    def __init__(self, sensitivity=4.0, python_bin='python', **kwargs):
        super().__init__(self, serialization='npy', **kwargs)

        here = os.path.dirname(__file__)
        feature_extraction_script = os.path.join(here, 'compute_features.py')
        self.program = [ python_bin, feature_extraction_script ]

        self.sensitivity = sensitivity

    def process(self, data):

        data = data.copy()
        columns = self.column_order
    
        # Convert from floats in "g" to the sensor scaling in int16
        data.loc[:, columns] = \
            ((data.loc[:, columns] / self.sensitivity) * (2**15-1)).astype(numpy.int16)

        return super().process(data)



def extract_features(sensordata : pandas.DataFrame,
    columns : list[str],
    groupby : list[str],
    extractor,
    samplerate = 50,
    label_column='activity',
    time_column='time',
    parallel_jobs : int = 4 ,
    verbose = 1,
    ) -> pandas.DataFrame:
    """
    Convert sensor data into fixed-sized time windows and extact features
    """

    # Process one whole stream of sensor data at a time
    # the feature extraction process might have time/history dependent logic,
    # such as filters estimating gravity, background levels etc
    def process_one(idx, stream : pandas.DataFrame) -> pandas.DataFrame:
    
        if verbose >= 4:
            log.debug('process-one-start',
                samples=len(stream),
                idx=idx,
            )

        # drop invalid data
        stream = stream.dropna(subset=columns)

        # Extract features
        windows = extractor.process(stream)

        # Combine with identifying information
        # time should come from data processing
        assert time_column in windows

        # the group is our job to manage
        index_columns = list(groupby) + [time_column]
        for idx_column, idx_value in zip(groupby, idx):
            windows[idx_column] = idx_value

        windows = windows.set_index(index_columns)
        if verbose >= 4:

            log.debug('process-one-done',
                columns=list(windows.columns),
                index_columns=list(windows.index.names),
                windows=len(windows),
                samples=len(stream),
            )
        return windows
    
    #win['window'] = win.index[0]
    # PERF: may be possible to parellize within a sensordata stream,
    # but then need each section to have a run-in period that is long enough
    # for any time-dependent logic to stabilize, and to merge while ignoring the run-in
    def split_sections(data, groupby : list[str], time_column='time'):
        groups = sensordata.groupby(groupby, observed=True)
        for group_idx, df in groups:

            # ensure sorted by time
            df = df.reset_index()

            # convert to time-delta, if neeeded
            #if pandas.api.types.is_datetime64_dtype(df[time_column]):
            #    df[time_column] = df[time_column] - df[time_column].min() 

            df = df.set_index(time_column).sort_index()

            expected_freq = pandas.Timedelta(1/samplerate, unit='s')
            diff = df.index.to_series().diff()
            holes = diff[diff > expected_freq]
            irregular = diff[diff != expected_freq].dropna()

            # Convert to regular time-series
            times = pandas.timedelta_range(df.index.min(), df.index.max(), freq=expected_freq)
            df = df.reindex(times)

            missing = df[columns].isna().any(axis=1)
            missing_ratio = numpy.count_nonzero(missing) / len(df)
            if missing_ratio > 0.01:
                log.debug('section-missing-data',
                    idx=group_idx,
                    rows=len(df[missing]),
                    ratio=missing_ratio,
                    irregular=len(irregular),
                )

            # Fill holes (if any)
            df = df.ffill()

            assert pandas.api.types.is_timedelta64_dtype(df.index)

            df[time_column] = df.index

            yield group_idx, df

    sections = split_sections(sensordata, groupby=groupby, time_column=time_column)
    jobs = [ joblib.delayed(process_one)(idx, df) for idx, df in sections]

    log.debug('process-parallel', jobs=len(jobs))

    start_time = time.time()
    features_values = joblib.Parallel(n_jobs=parallel_jobs)(jobs)
    out = pandas.concat(features_values)
    duration = round(time.time() - start_time, 3)

    log.debug('process-parallel-done', rows=len(out), duration=duration)


    return out

def export_model(path, out, name='motion'):

    with open(path, "rb") as f:
        classifier = pickle.load(f)

        classes = classifier.classes_
        class_mapping = dict(zip(classes, range(len(classifier.classes_))))

        cmodel = emlearn.convert(classifier)
        cmodel.save(name=name, format='csv', file=out)


def load_config(file_path):

    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    return data

def label_windows(sensordata,
        windows,
        groupby,
        label_column,
        window_duration : pandas.Timedelta,
        majority=0.66) -> pandas.DataFrame:

    windows = windows.copy()
    # default to unknown=NA
    windows[label_column] = None

    print(sensordata.head())

    sensor_groups = {idx: df for idx, df in sensordata.groupby(groupby, group_keys=False, as_index=False) }

    log.debug('label-windows', groups=groupby, g=list(sensor_groups.keys()))

    for idx, ww in windows.groupby(groupby):
        data = sensor_groups[idx]
        #log.debug('label-window', idx=idx, index_dtype=data.index.dtype)        
        data = data.reset_index().set_index('time') # XXX: Why is this needed?

        # convert to time-delta, if neeeded
        #if pandas.api.types.is_datetime64_dtype(data.index):
        #    data.index = data.index - data.index.min() 

        for idx, w in ww.iterrows():
            window_end = idx[-1] # XXX: assuming this is time
            window_start = window_end - window_duration

            labels = data.loc[window_start:window_end, label_column]
            threshold = majority * len(labels)
            counts = labels.value_counts(dropna=True)
            label = counts.index[0] if counts.iloc[0] > threshold else None
            windows.loc[idx, label_column] = label

    return windows

def plot_timelines(sensordata, windows, groupby, sensor_columns, label_column):

    # Plot
    from plotting import make_timeline_plot

    sensor_columns = [ c for c in sensor_columns if c.startswith('gyro_')] # XXX

    sensor_groups = {idx: df for idx, df in sensordata.groupby(groupby, group_keys=False, as_index=False) }

    log.debug('label-windows', groups=groupby, g=list(sensor_groups.keys()))

    for idx, ww in windows.groupby(groupby, group_keys=False, as_index=False):
        data = sensor_groups[idx]
        #log.debug('label-window', idx=idx, index_dtype=data.index.dtype) 

        # XXX: Why is this needed?  
        data = data.reset_index().set_index('time')
        ww = ww.reset_index().set_index('time')

        # convert to seconds
        #data.index = data.index / pandas.Timedelta(seconds=1)
        #ww.index = ww.index / pandas.Timedelta(seconds=1)

        feature_columns = list(ww.columns)

        feature_columns = [
            'motion_mag_rms', 'motion_mag_p2p', 'motion_x_rms', 'motion_y_rms', 'motion_z_rms',
            'fft_0_4hz', 'fft_0_8hz', 'fft_1_2hz', 'fft_1_6hz', 'fft_1_10hz', 'fft_2_3hz', 'fft_2_7hz', 'fft_3_1hz', 'fft_3_5hz']

        line_features  = [
            #'orientation_x', 'orientation_y', 'orientation_z',
            'motion_mag_rms'
        ]

        #print('pp', ww[o])

        idx_name = '_'.join([str(s) for s in idx]   )
        plot_path  = f'plot_{idx_name}.png'
        # Make a plot
        width = 1600
        aspect = 2.0
        height = width/aspect
        fig = make_timeline_plot(data, ww,
            sensor_columns=sensor_columns,
            label_column=label_column,
            line_feature_columns=line_features,
            heatmap_feature_columns=feature_columns,
            colors=None,
            class_names=['class_0', 'class_1'], # FIXME: pass
            predictions=None, # FIXME: pass separate
            width=width, aspect=aspect)

        fig.write_image(plot_path, scale=1.5, width=width, height=height)
        print('Wrote plot', plot_path)

def run_pipeline(run, hyperparameters, dataset,
        config,
        data_dir,
        out_dir,
        model_settings=dict(),
        n_splits=5,
        features='timebased',
    ):

    dataset_config = load_config(config)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    data_path = os.path.join(data_dir, f'{dataset}.parquet')

    data_load_start = time.time()
    log.info('data-load-start', dataset=dataset)
    data = pandas.read_parquet(data_path)

    groups = dataset_config['groups']
    data_columns = dataset_config['data_columns']
    enabled_classes = dataset_config['classes']
    label_column = dataset_config.get('label_column', 'activity')
    time_column = dataset_config.get('time_column', 'time')
    sensitivity = dataset_config.get('sensitivity', 4.0)

    print('dd', sorted(data.columns))
    print('dt', data.dtypes)

    data[label_column] = data[label_column].astype(str)

    data_load_duration = time.time() - data_load_start
    log.info('data-load-end', dataset=dataset, samples=len(data), duration=data_load_duration)

    feature_extraction_start = time.time()
    log.info('feature-extraction-start',
        dataset=dataset,
        features=features,
    )
    window_length = model_settings['window_length']
    samplerate = dataset_config.get('samplerate', 100)
    window_hop = model_settings['window_hop']
    
    window_duration = (window_length / samplerate)

    remap = {
        'x': 'acc_x',
        'y': 'acc_y',
        'z': 'acc_z',
    }
    data = data.rename(columns=remap)


    # convert to time-delta, if neeeded
    def convert_time(data):
        if pandas.api.types.is_datetime64_dtype(data.index):
            data.index = data.index - data.index.min()
        return data

    data = data.groupby(groups, as_index=False, group_keys=False).apply(convert_time)


    # Setup feature extraction
    extract_options = dict(
        window_length=window_length,
        hop_length=window_hop,
        samplerate=samplerate,
    )
    if features == 'timebased':
        #columns = ['x', 'y', 'z']
        columns = ['acc_x', 'acc_y', 'acc_z']
        extractor = TimebasedFeatureExtractor(sensitivity=sensitivity, column_order=columns, options=extract_options)

    elif features == 'custom':
        # FIXME: unhardcode path
        executable = ['/home/jon/projects/emlearn/examples/motion_recognition/build/motion_preprocess']

        columns = ['time', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
        data_columns = [ c for c in columns if not c == 'time' ]
        extractor = DataProcessorProgram(program=executable,
            options=extract_options, column_order=columns)

        # Feature extractor expects these to be set
        data['gyro_x'] = 0.0
        data['gyro_y'] = 0.0
        data['gyro_z'] = 0.0
    else:
        raise ValueError(f"Unsupported features: {features}")

    features = extract_features(data,
        extractor=extractor,
        columns=data_columns,
        groupby=groups,
        label_column=label_column,
        time_column=time_column,
        samplerate=samplerate,
        #parallel_jobs=1,
    )

    # Attach labels
    features = label_windows(data, features,
        groupby=groups,
        label_column=label_column,
        window_duration=pandas.Timedelta(window_duration, unit='s'),
    )

    labeled = numpy.count_nonzero(features[label_column].notna())

    feature_extraction_duration = time.time() - feature_extraction_start
    log.info('feature-extraction-done',
        dataset=dataset,
        total_instances=len(features),
        labeled_instances=labeled,
        duration=feature_extraction_duration,
    )

    #plot_timelines(data, features, groupby=groups,
    #    sensor_columns=data_columns, label_column=label_column)

    # FIXME: keep the windows in evaluation, only ignore for training

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
    model_path = os.path.join(out_dir, f'{dataset}.trees.csv')
    export_model(estimator_path, model_path)

    # Save metadata
    classes = estimator.classes_
    class_mapping = dict(zip(classes, range(len(classes))))
    meta_path = os.path.join(out_dir, f'{dataset}.meta.json')
    metadata = dict(classes=class_mapping, window_length=window_length)
    with open(meta_path, 'w') as f:
        f.write(json.dumps(metadata))

    # Save testdata
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
    parser.add_argument('--config', type=str, default='data/configurations/uci_har.yaml',
                        help='Which dataset/training config to use')
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
        config=args.config,
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
