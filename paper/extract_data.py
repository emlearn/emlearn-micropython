
import pandas

def main():

    data_path = '../examples/har_trees/data/processed/pamap2.parquet'
    data = pandas.read_parquet(data_path)

    # Select only a 15 minute subsection of data
    one = data.loc['subject102']
    sub = one.loc[pandas.Timedelta(minutes=48):pandas.Timedelta(minutes=63)]

    # Only include columns that will be used
    sensor_columns = [
        'hand_acceleration_6g_x',
        'hand_acceleration_6g_y',
        'hand_acceleration_6g_z',
    ]
    sub_columns = sensor_columns + ['activity']
    sub = sub[sub_columns]
    sub['activity'] = sub['activity'].astype('category')

    output_path = 'pamap2_subject102_exerpt.parquet'
    sub.to_parquet(output_path, index=True, compression='gzip')
    print('wrote', output_path)

if __name__  == '__main__':
    main()


