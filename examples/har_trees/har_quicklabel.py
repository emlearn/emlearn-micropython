
"""
Take the activity markers in har_record.py as labels
"""

import os

import pandas

from har_data2labelstudio import load_har_record

def parse():
    import argparse
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--dataset', type=str, default='uci_har',
                        help='Which dataset to use')
    parser.add_argument('--config', type=str, default='data/configurations/uci_har.yaml',
                        help='Which dataset/training config to use')

    parser.add_argument('--data-dir', metavar='DIRECTORY', type=str, default='./data/raw/uci_har',
                        help='Where the input data is stored')
    parser.add_argument('--out-dir', metavar='DIRECTORY', type=str, default='./data/processed',
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
    out_path = os.path.join(args.out_dir, f'{dataset}.parquet')
    data_path = os.path.join(args.data_dir)

    # Lookup data
    recordings = load_har_record(data_path)

    # Create packed dataframe
    dfs = []
    for filename, row in recordings.iterrows():
        classname = row.classname
        filename = filename
        #print(filename, classname)    

        d = row.data
        d = d.reset_index()
        d['subject'] = 'unknown'
        d['file'] = filename
        d['activity'] = classname
        dfs.append(d)

    #p = 'data/processed/pamap2.parquet'
    
    out = pandas.concat(dfs, ignore_index=True)
    out.to_parquet(out_path)

    #return
    # Sanity check
    df = pandas.read_parquet(out_path)
    print(df.columns)
    print(df.activity.value_counts())
    print(df.file.value_counts())
    print(df.head(5))


if __name__ == '__main__':
    main()
