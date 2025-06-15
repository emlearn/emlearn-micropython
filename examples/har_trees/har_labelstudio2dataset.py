
import re
import os

import pandas


from har_data2labelstudio import load_har_record


def extract_filename(url):

    base = os.path.basename(url)

    # label studio adds a ID- on import
    filename = re.sub(r'(\w+)-', '', base, count=1)

    return filename

def read_labels(path):
    df = pandas.read_csv(path)

    df = df.rename(columns={
        'trend_forecast': 'activity',
        'timeseriesUrl': 'data_url',
    }, errors='ignore')

    # Extract data identifier, used for correlating with data
    df['file'] = df['data_url'].apply(extract_filename)

    columns = ['file', 'activity']
    df = df[columns]

    # Convert to a single label per file
    # even though there may be multiple annotations
    df = df.groupby('file').agg(pandas.Series.mode)

    #print(df.head())

    return df
    


def main():

    out_path = 'har_exercise_1.parquet'
    labels_path = 'project-3-at-2024-12-01-17-29-ba296417.csv'
    data_path = 'data/har_record_excercises/har_record'
    seed = 1

    labels = read_labels(labels_path)

    # Drop files with a mix of class labels
    labels = labels[~labels.activity.isin(['mixed'])]

    # Balance the 'other' category, by downsampling
    without_other = labels[labels.activity != 'other']
    class_occurance = int(without_other.value_counts().median())
    only_other = labels[labels.activity == 'other']
    other_downsampled = only_other.sample(n=class_occurance, random_state=seed)
    labels = pandas.concat([without_other, other_downsampled])

    print('\nFiles after balancing:')
    print(labels.activity.value_counts())

    # Lookup data
    data = load_har_record(data_path)

    # Merge in label data
    dfs = []
    for filename, row in labels.iterrows():
        classname = row.activity
        filename = filename.rstrip('.csv')
        #print(filename, classname)

        try:
            d = data.loc[filename].data
        except KeyError as e:
            print('Load error', e)
            continue

        d = d.reset_index()
        d['subject'] = 'unknown'
        d['file'] = filename
        d['activity'] = classname
        dfs.append(d)

    #p = 'data/processed/pamap2.parquet'
    
    out = pandas.concat(dfs, ignore_index=True)
    print(out)
    out.to_parquet(out_path)

    #return
    # Sanity check
    df = pandas.read_parquet(out_path)
    print(df.columns)
    print(df.activity.value_counts())
    print(df.file.value_counts())
    print(df.head())

if __name__ == '__main__':
    main()
