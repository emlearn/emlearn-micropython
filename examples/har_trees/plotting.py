
import numpy
import pandas

import plotly.express as px
import plotly.graph_objects as go

from plotly.colors import qualitative

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

def find_runs(labels : pandas.Series,
        label_column='activity',
        name='run'):

    # Detect changes
    change_points = labels != labels.shift()
    run_ids = change_points.cumsum()
    
    # Extract runs
    def foo(g):
        #print(g.iloc[0], g.index[0], g.index[-1])
        out = pandas.Series({
            'label': g.iloc[0][label_column],
            'start_time': g.index[0],
            'end_time': g.index[-1],
        })
        return out

    labels_df = labels.to_frame()
    labels_df[name] = run_ids
    runs = labels_df.groupby(name, as_index=False).apply(foo, include_groups=False)
    return runs

def get_subplot_axes(rows, cols, row, col):
    subplot_num = (row - 1) * cols + col
    if subplot_num == 1:
        return 'x', 'y'
    else:
        return f'x{subplot_num}', f'y{subplot_num}'

def add_events(fig, df, label_colors, subplot={}, cols=0, font_size=16, font_family='sans-serif'):
    
    xaxis, yaxis = get_subplot_axes(fig, cols=cols, **subplot)

    # Plot rectangles
    for _, row in df.iterrows():
        label = row['label']
        if pandas.isna(label):
            #print('Skipping', row)
            continue
        fig.add_shape(
            type='rect',
            x0=row['start_time'],
            x1=row['end_time'],
            y0=0,
            y1=1,
            xref='x',
            yref=f'{yaxis} domain',
            fillcolor=label_colors[row['label']],
            opacity=0.3,
            line_width=0,
            layer='below',
            **subplot,
        )
    
        # Optional label annotation
        fig.add_annotation(
            x=(row['start_time'] + (row['end_time'] - row['start_time']) / 2),
            y=1.02,
            text=row['label'],
            showarrow=False,
            xref='x',
            yref=f'{yaxis} domain',
            font=dict(size=font_size, family=font_family),
            **subplot,
        )

def time_ticks(times, every=30, skip_start=0):
    n_times = len(times)
    start = times.min() + skip_start
    end = times.max()

    def minute_second_format(t : float):
        """MM:SS"""
        return f"{int(t//60):02}:{int(t%60):02}"
    
    tick_vals = numpy.arange(start, end, every)
    tick_text = [ minute_second_format(t) for t in tick_vals]

    return tick_vals, tick_text

def make_label_colors(labels) -> dict[str, object]:

    color_pool = qualitative.Set3 + qualitative.Dark24 + qualitative.Pastel1
    label_colors = dict(zip(labels, color_pool))
    assert len(labels) <= len(color_pool)

    return label_colors

def plot_timeline(fig, df,
                  data: list[str],
                  label_colors=None,
                  data_colors=None,
                  time='time',
                  label='activity',
                  subplot={},
                  cols=1,
                  opacity=0.5,
                 ):

    df = df.reset_index()
    df[time] = convert_times(df[time])
    df = df.sort_values(time)
    
    if label_colors is None and label is not None:
        label_colors = make_label_colors(df[label].unique())

    if data_colors is None:
        data_colors = make_label_colors(list(set(data)))

    if label is not None:
        df = df.set_index(time)
        events = find_runs(df[label], label_column=label)
        events = events[~events.label.isin(['transient'])]
        df = df.reset_index()
        add_events(fig, events, label_colors=label_colors, subplot=subplot, cols=cols)
    
    # Add each axis as a line
    for column in data:
        y = df[column]
        #print(column)
        #print(df[time])
        trace = go.Scatter(x=df[time],
            y=y,
            mode='lines',
            name=column,
            line=dict(color=data_colors[column]),
            opacity=opacity,
        )
        fig.add_trace(trace, **subplot)
       
def convert_times(times):
    out = times
    out = out / pandas.Timedelta(seconds=1)
    out -= out.min()
    return out

def configure_xaxis(fig, times, every=60, col=1, row=1, standoff=10):

    times = convert_times(times)

    tick_vals, tick_text = time_ticks(times, skip_start=30)

    # Customize layout
    fig.update_xaxes(
        tickmode='array',
        tickvals=tick_vals,
        ticktext=tick_text,
        col=col, row=row,
    )

    fig.add_annotation(
        text="Time (MM:SS)",
        x=0,                    # Left edge of plot area
        y=-0.035,                 # Below the plot (negative moves down)
        xref="x domain",        # Relative to subplot domain
        yref="paper",           # Relative to entire figure
        xanchor="left",         # Left-align text
        yanchor="top",          # Anchor to top of text
        showarrow=False,
        font=dict(size=14)
    )


def plot_heatmap(fig, data, columns, y_labels=None, colorscale='RdBu', time='time', zmax=1.0, zmin=None, subplot={}):

    if zmin is None:
        zmin = -zmax
    
    from plotly import graph_objects as go

    if y_labels is None:
        y_labels = columns

    data[time] = convert_times(data[time])
    
    x_labels = data[time]
    z_data = data[columns].T.values  # Transpose for proper orientation

    # Create heatmap
    fig.add_heatmap(
        z=z_data,
        x=x_labels,
        y=y_labels,
        colorscale=colorscale,
        showscale=False,
        zmin=zmin,
        zmax=zmax,
        **subplot,
    )
    
    return fig


def fft_freq_from_bin(bin_idx, length, sample_rate):
    """Convert FFT bin index to frequency"""
    return bin_idx * sample_rate / length

def make_timeline_plot(data, features, predictions,
    colors,
    class_names,
    label_column,
    sensor_columns=[],
    line_feature_columns=[],
    heatmap_feature_columns=[],
    width = 1600,
    aspect = 2.0,
    ):
    
    from plotly.subplots import make_subplots
    from plotting import configure_xaxis, plot_heatmap

    # Create subplots
    rows = 4
    fig = make_subplots(
        rows=rows, cols=1,
        row_titles=('Input', 'Features', 'Predictions'),
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=(0.2, 0.2, 0.6, 0.2),
    )
    subplots = [ dict(col=1, row=i) for i in range(1, rows+1) ] 
    
    # Input data
    if sensor_columns:
        sensor_colors = dict(zip(sensor_columns, ['red', 'green', 'blue']))
        scaled_data = data.copy()
        scale_sensor = 1.0  # XXX: guessing appropriate scaling factor
        scaled_data.loc[:, sensor_columns] = scaled_data.loc[:, sensor_columns] / scale_sensor 
        plot_timeline(fig, scaled_data, data=sensor_columns,
            label=label_column,
            label_colors=colors,
            data_colors=sensor_colors,
            subplot=subplots[0]
        )
        #fig.update_yaxes(range=(-5, 5), **subplots[0], title_text="Acceleration (g)")
        
    # Features
    if line_feature_columns:
        scaler = RobustScaler(quantile_range=(5.0, 95.0), with_centering=False)
        scaler.set_output(transform='pandas')
        features_scaled = scaler.fit_transform(features[line_feature_columns])

        print('features', sorted(features.columns))
        print('pf', features_scaled.head())

        plot_timeline(fig, features,
                      data=line_feature_columns,
                      label=label_column,
                      label_colors=colors,
                      data_colors=colors,
                      opacity=1.0,
                      subplot=subplots[1])


    if heatmap_feature_columns:
        heatmap_colorscale = 'blues'
        scaler = RobustScaler(quantile_range=(5.0, 95.0), with_centering=False)
        scaler.set_output(transform='pandas')
        features_scaled = scaler.fit_transform(features[heatmap_feature_columns])
        plot_heatmap(fig,
                     features_scaled.reset_index(),
                     columns=heatmap_feature_columns,
                     #y_labels=feature_names,
                     subplot=subplots[2],
                     #zmin=0.0, zmax=3.0,
                     colorscale=heatmap_colorscale,
        )


    # Predictions
    if predictions is not None:
        plot_timeline(fig, predictions, data=class_names,
                      label=label_column,
                      label_colors=colors,
                      data_colors=colors,
                      opacity=1.0,
                      subplot=subplots[3])
        fig.update_yaxes(range=(0, 1.0), title_text="Probability", **subplots[3])
        
        
    configure_xaxis(fig, data.index, every=60, row=rows-0)
    
    # Customize layout
    fig.update_layout(
        template='plotly_white',
    )

    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.03,               # Adjust distance from plot
            xanchor="center",
            x=0.5,
            font=dict(size=10),    # Smaller font if needed
            itemwidth=30           # Control item spacing
        ),
        margin=dict(b=30)          # Add bottom margin for legend space
    )

    fig.update_yaxes(tickformat=".1f")
    
    height = int(width / aspect)
    fig.update_layout(
        width=width,
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),  # Fixed margins
        autosize=False
    )

    return fig


