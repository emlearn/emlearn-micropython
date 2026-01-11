import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import urllib.request
import zipfile
import os.path

from sklearn.multioutput import MultiOutputRegressor


def airquality_download(data_dir='data'):
    """
    UCI Air Quality dataset
    https://archive.ics.uci.edu/dataset/360/air+quality
    """

    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    csv_file = data_path / 'AirQualityUCI.csv'
    
    if not csv_file.exists():
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip"
        zip_path = data_path / 'AirQualityUCI.zip'
        
        urllib.request.urlretrieve(url, zip_path)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_path)
        
        zip_path.unlink()
    
    return csv_file


def airquality_load(csv_file):
    df = pd.read_csv(csv_file, sep=';', decimal=',')
    
    # Remove missing values
    df = df.replace(-200, np.nan)
    df = df.dropna(axis=1, how='all').dropna()
    
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H.%M.%S')
    df = df.drop(['Date', 'Time'], axis=1)

    target_cols = ['CO(GT)', 'NOx(GT)', 'NO2(GT)', 'NMHC(GT)', 'C6H6(GT)']
    exclude_cols = target_cols + ['datetime', 'Unnamed: 15', 'Unnamed: 16']
    feature_cols = [col for col in df.columns if col not in exclude_cols]    

    X = df[feature_cols]
    y = df[target_cols]
    
    return X, y


from emlearn.preprocessing import Quantizer
import emlearn

def convert_multiregressor(multi, out_dir, format=None, prefix='regressor', **kwargs):

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    if format is not None:
        kwargs['format'] = format

    for i, estimator in enumerate(multi.estimators_):
        ext = '.h' if format is None else '.'+format
        p = out_dir / (f'{prefix}{i}' + ext)
        converted = emlearn.convert(estimator)
        converted.save(file=p, **kwargs)



def main():

    print('Load dataset...')
    csv_file = airquality_download()
    X, y = airquality_load(csv_file)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print('Training...')
    rf = RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=-1)
    regressor = MultiOutputRegressor(estimator=rf)

    pipeline = Pipeline([
        ('scaler', Quantizer()), # convert data to int16 range
        ('regressor', regressor),
    ])
    pipeline.fit(X_train, y_train)

    model_dir = 'models/'
    convert_multiregressor(pipeline.named_steps['regressor'], out_dir=model_dir, format='csv')
    print('Models exported to:', model_dir)


    print("Performance Metrics:")
    print("-" * 60)
    y_pred = pd.DataFrame(pipeline.predict(X_test), columns=y_train.columns)
    for i, target in enumerate(y.columns):
        rmse = np.sqrt(mean_squared_error(y_test[target], y_pred[target]))
        r2 = r2_score(y_test[target], y_pred[target])
        mape = mean_absolute_percentage_error(y_test[target], y_pred[target]) * 100
        
        print(f"{target:12} | RMSE: {rmse:8.3f} | MAPE: {mape:6.2f}% | R²: {r2:6.3f}")

if __name__ == '__main__':
    main()


