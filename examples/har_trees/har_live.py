
from machine import Pin, I2C
from mpu6886 import MPU6886

import time
import struct
import array

from windower import TriaxialWindower, empty_array
import timebased
import emlearn_trees    

def mean(arr):
    m = sum(arr) / float(len(arr))
    return m

def main():

    classname_index = {"LAYING": 0, "SITTING": 1, "STANDING": 2, "WALKING": 3, "WALKING_DOWNSTAIRS": 4, "WALKING_UPSTAIRS": 5}
    class_index_to_name = { v: k for k, v in classname_index.items() }

    # Load a CSV file with the model
    model = emlearn_trees.new(10, 1000, 10)
    with open('har_uci_trees.csv', 'r') as f:
        emlearn_trees.load_model(model, f)

    mpu = MPU6886(I2C(0, sda=21, scl=22, freq=100000))

    # Enable FIFO at a fixed samplerate
    mpu.fifo_enable(True)
    mpu.set_odr(100)

    window_length = 100
    hop_length = 50
    chunk = bytearray(mpu.bytes_per_sample*hop_length)

    x_values = empty_array('h', hop_length)
    y_values = empty_array('h', hop_length)
    z_values = empty_array('h', hop_length)
    windower = TriaxialWindower(window_length)

    features_typecode = timebased.DATA_TYPECODE
    n_features = timebased.N_FEATURES
    features = array.array(features_typecode, (0 for _ in range(n_features)))

    while True:

        count = mpu.get_fifo_count()
        if count >= hop_length:
            start = time.ticks_ms()
            # read data
            mpu.read_samples_into(chunk)
            mpu.deinterleave_samples(chunk, x_values, y_values, z_values)
            windower.push(x_values, y_values, z_values)
            if windower.full():
                # compute features
                #print('xyz', mean(x_values), mean(y_values), mean(z_values))
                ff = timebased.calculate_features_xyz((x_values, y_values, z_values))
                for i, f in enumerate(ff):
                    features[i] = int(f)

                # Cun classifier
                result = model.predict(features)
                activity = class_index_to_name[result]

                d = time.ticks_diff(time.ticks_ms(), start)
                print('class', activity, d)

        time.sleep_ms(1)


if __name__ == '__main__':
    main()
