
import numpy
import pandas

import plotly.express as px
import plotly.graph_objects as go

from plotly.colors import qualitative

def find_runs(labels : pandas.Series):

    # Detect changes
    change_points = labels != labels.shift()
    run_ids = change_points.cumsum()
    
    # Extract runs
    def foo(g):
        #print(g.iloc[0], g.index[0], g.index[-1])
        out = pandas.Series({
            'label': g.iloc[0].activity,
            'start_time': g.index[0],
            'end_time': g.index[-1],
        })
        return out
    runs = labels.to_frame().groupby(run_ids, as_index=False).apply(foo)
    return runs

def get_subplot_axes(rows, cols, row, col):
    subplot_num = (row - 1) * cols + col
    if subplot_num == 1:
        return 'x', 'y'
    else:
        return f'x{subplot_num}', f'y{subplot_num}'

def add_events(fig, df, label_colors, subplot={}, cols=0, font_size=16, font_family='sans-serif'):
    
    xaxis, yaxis = get_subplot_axes(fig, cols=cols, **subplot)
    print('add-events', subplot, yaxis)

    # Plot rectangles
    for _, row in df.iterrows():
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
    
    if label_colors is None:
        label_colors = make_label_colors(df[label].unique())

    if data_colors is None:
        data_colors = make_label_colors(list(set(data)))

    if label is not None:
        df = df.set_index(time)
        events = find_runs(df[label])
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

