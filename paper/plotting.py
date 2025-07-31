
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

def add_events(fig, df, subplot={}, cols=0):

    labels = df['label'].unique()
    color_pool = qualitative.Set3 + qualitative.Dark24 + qualitative.Pastel1
    label_colors = dict(zip(labels, color_pool))
    assert len(labels) <= len(color_pool)
    
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
            font=dict(size=10),
            **subplot,
        )

def time_ticks(times, every=30):
    n_times = len(times)
    start = times.min()
    end = times.max()

    def minute_second_format(t : float):
        """MM:SS"""
        return f"{int(t//60):02}:{int(t%60):02}"
    
    tick_vals = numpy.arange(start, end, every)
    tick_text = [ minute_second_format(t) for t in tick_vals]

    return tick_vals, tick_text

def plot_timeline(fig, df,
                  data: list[str],
                  time='time',
                  label='activity',
                  subplot={},
                  cols=1,
                 ):


    df = df.reset_index()
    df[time] = df[time] / pandas.Timedelta(seconds=1) # convert to seconds
    df[time] -= df[time].min()
    df = df.sort_values(time)
    #print(df[time].min(), df[time].max())
    
    if label is not None:
        df = df.set_index(time)
        events = find_runs(df[label])
        events = events[~events.label.isin(['transient'])]
        df = df.reset_index()
        add_events(fig, events, subplot=subplot, cols=cols)
    
    # Add each axis as a line
    for column in data:
        y = df[column]
        #print(column)
        #print(df[time])
        trace = go.Scatter(x=df[time],
            y=y,
            mode='lines',
            name=column
        )
        fig.add_trace(trace, **subplot)
        
    
    tick_vals, tick_text = time_ticks(df[time], every=60)
    
    # Customize layout
    fig.update_layout(
        #title='Tri-Axial Accelerometdf[data[0]]er Data Over Time',
        #xaxis_title='Time',
        #yaxis_title='Acceleration (g)',
        #legend_title='Axis',
        template='plotly_white',
        #hovermode='x unified'
        xaxis=dict(
            title='Elapsed Time (MM:SS)',
            tickmode='array',
            tickvals=tick_vals,
            ticktext=tick_text,
        ),
    )

    return fig


