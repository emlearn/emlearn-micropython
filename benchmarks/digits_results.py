
import pandas
from matplotlib import pyplot as plt
import seaborn


def load_data(path, n_examples = 10):

    df = pandas.read_csv(path).set_index('model')

    # correct for overhead in the test itself
    df['time_us'] = df['time_us'] - df.loc['none', 'time_us']
    df = df.drop('none')

    df['per_second'] = (1e6 / (df['time_us'] / n_examples))
    df.index.name = 'framework'

    return df

def plot(df, path=None):

    fig, ax = plt.subplots(1, figsize=(10, 5))
    seaborn.barplot(data=df.reset_index(), x='framework', y='per_second')
    fig.tight_layout()
    seaborn.despine(fig=fig)
    if not path is None:
        fig.savefig(plot_path)

results_path = 'benchmarks/digits_results.csv'
plot_path = 'benchmarks/digits_bench.png'
results = load_data(results_path)
plot(results, path=plot_path)
print('Wrote to', plot_path)
