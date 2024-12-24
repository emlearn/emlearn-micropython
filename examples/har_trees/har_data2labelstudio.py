

import os

import pandas
import numpy

def load_har_record(path,
        samplerate=100,
        sensitivity=2.0,
        maxvalue=32767,
        ):

    suffix = '.npy'

    files = []

    for f in os.listdir(path):
        if f.endswith(suffix):
            p = os.path.join(path, f)
            try:
                data = numpy.load(p, allow_pickle=True)
            except Exception as e:
                print(e)
                continue

            df = pandas.DataFrame(data, columns=['x', 'y', 'z'])

            # Scale values into physical units (g)
            df = df.astype(float) / maxvalue * sensitivity

            # Add a time column, use as index
            t = numpy.arange(0, len(df)) * (1.0/samplerate)
            df['time'] = t
            df = df.set_index('time')

            classname = f.split('_')[1].rstrip(suffix)
            
            # Remove :, special character on Windows
            filename = f.replace(':', '')

            files.append(dict(data=df, filename=filename, classname=classname))

            #print(f, data.shape)

    out = pandas.DataFrame.from_records(files)
    out = out.set_index('filename')
    return out

def main():

    p = './data/har_record_excercises/har_record'
    data = load_har_record(p, samplerate=100)

    print(data.head())
    print(data.shape)
    print(data.classname.value_counts())

    out_dir = 'to_label'

    # Write .CSV files, suitable for importing in Label Studio
    for idx, f in data.iterrows():
        d = f.data.sort_index()
        
        out_path = os.path.join(out_dir, idx+'.csv')
        d.to_csv(out_path)
        print('Wrote', out_path)

if __name__ == '__main__':
    main()


