
import machine
from machine import Pin, I2C
from mpu6886 import MPU6886
import bluetooth

import time
import struct
import array

from windower import TriaxialWindower, empty_array
import timebased
import emlearn_trees    

def mean(arr):
    m = sum(arr) / float(len(arr))
    return m

def argmax(arr):
    idx_max = 0
    value_max = arr[0]
    for i in range(1, len(arr)):
        if arr[i] > value_max:
            value_max = arr[i]
            idx_max = i

    return idx_max

def copy_array_into(source, target):
    assert len(source) == len(target)
    for i in range(len(target)):
        target[i] = source[i]

def clamp(value, lower, upper) -> float:
    v = value
    v = min(v, upper)
    v = max(v, lower)
    return v

def manufacturer_specific_advertisement(data : bytearray, manufacturer=[0xca, 0xab], limited_disc=False, br_edr=False):
    _ADV_TYPE_FLAGS = const(0x01)
    _ADV_TYPE_CUSTOMDATA = const(0xff)
    _ADV_MAX_PAYLOAD = const(31)

    payload = bytearray()

    # Advertising payloads are repeated packets of the following form:
    #   1 byte data length (N + 1)
    #   1 byte type (see constants below)
    #   N bytes type-specific data
    def _append(adv_type, value):
        nonlocal payload
        payload += struct.pack("BB", len(value) + 1, adv_type) + value

    # Flags
    _append(
        _ADV_TYPE_FLAGS,
        struct.pack("B", (0x01 if limited_disc else 0x02) + (0x18 if br_edr else 0x04)),
    )

    # Specify manufacturer-specific data
    manufacturer_id = bytearray(manufacturer)
    _append(_ADV_TYPE_CUSTOMDATA, (manufacturer_id + data))

    if len(payload) > _ADV_MAX_PAYLOAD:
        raise ValueError("advertising payload too large")

    return payload

def send_bluetooth_le(sequence, probabilities,
        advertisements=4,
        advertise_interval_ms=50,
        format=0xAA,
        version=0x01):
    """
    Send data as BLE advertisements
    Delivery of advertisements are not guaranteed. So we repeat N times to have a decent chance
    """

    # Start BLE
    ble = bluetooth.BLE()   
    ble.active(True)
    mac = ble.config('mac')

    # Encode data as BLE advertisement. Max 29 bytes
    data = bytearray()
    data += struct.pack('B', format)
    data += struct.pack('B', version)
    data += struct.pack('>H', sequence)

    for p in probabilities:
        q = int(clamp(p*255, 0, 255))
        data += struct.pack('B', q)

    payload = manufacturer_specific_advertisement(data)

    print('ble-advertise', mac[1], data)

    # send and wait until N advertisements are sent
    advertise_us = int(1000*advertise_interval_ms)
    ble.gap_advertise(advertise_us, adv_data=payload, connectable=False)
    time.sleep_ms(advertisements*advertise_interval_ms)

    # Turn of BLE
    ble.active(False)

def main():

    dataset = 'har_exercise_1'
    
    if dataset == 'uci_har':
        classname_index = {"LAYING": 0, "SITTING": 1, "STANDING": 2, "WALKING": 3, "WALKING_DOWNSTAIRS": 4, "WALKING_UPSTAIRS": 5}
        window_length = 128
    elif dataset == 'har_exercise_1':
        classname_index = {"jumpingjack": 0, "lunge": 1, "other": 2, "squat": 3}
        window_length = 256
    else:
        raise ValueError('Unknown dataset')

    model_path = f'{dataset}_trees.csv'
    class_index_to_name = { v: k for k, v in classname_index.items() }


    # Load a CSV file with the model
    model = emlearn_trees.new(10, 1000, 10)
    with open(model_path, 'r') as f:
        emlearn_trees.load_model(model, f)

    mpu = MPU6886(I2C(0, sda=21, scl=22, freq=100000))

    # Enable FIFO at a fixed samplerate
    mpu.fifo_enable(True)
    mpu.set_odr(100)

    hop_length = 64
    chunk = bytearray(mpu.bytes_per_sample*hop_length)

    x_values = empty_array('h', hop_length)
    y_values = empty_array('h', hop_length)
    z_values = empty_array('h', hop_length)
    windower = TriaxialWindower(window_length)

    x_window = empty_array('h', window_length)
    y_window = empty_array('h', window_length)
    z_window = empty_array('h', window_length)

    features_typecode = timebased.DATA_TYPECODE
    n_features = timebased.N_FEATURES
    features = array.array(features_typecode, (0 for _ in range(n_features)))
    out = array.array('f', range(model.outputs()))

    prediction_no = 0

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

                copy_array_into(windower.x_values, x_window)
                copy_array_into(windower.y_values, y_window)
                copy_array_into(windower.z_values, z_window)

                ff = timebased.calculate_features_xyz((x_window, y_window, z_window))
                for i, f in enumerate(ff):
                    features[i] = int(f)

                # Cun classifier
                #print(features)
                model.predict(features, out)
                result = argmax(out)
                activity = class_index_to_name[result]

                d = time.ticks_diff(time.ticks_ms(), start)
                print('classify', d, activity, out)

                send_bluetooth_le(prediction_no, out)
                prediction_no += 1

        time.sleep_ms(100)
        #machine.lightsleep(100)


if __name__ == '__main__':
    main()
