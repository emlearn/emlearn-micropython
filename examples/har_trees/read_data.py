

import os

import pandas
import numpy

def load_har_record(path, samplerate):
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

            df = pandas.DataFrame(data.T, columns=['x', 'y', 'z'])
            t = numpy.arange(0, len(df)) * (1.0/samplerate)
            df['time'] = t
            df = df.set_index('time')
            classname = f.split('_')[1].rstrip(suffix)

            files.append(dict(data=df, filename=f, classname=classname))

            #print(f, data.shape)

    out = pandas.DataFrame.from_records(files)
    out = out.set_index('filename')
    return out

p = '/home/jon/temp/micropython-test-esptool/har_record'
data = load_har_record(p, samplerate=100)

print(data.head())
print(data.shape)

import plotly.express

for idx, f in data.iterrows():
    d = f.data.reset_index()
    fig = plotly.express.line(d, x='time', y=['x', 'y', 'z'])
    fig.show()
