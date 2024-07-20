
# Implementing an IMU/accelerometer/gyro driver? Use the FIFO!

Hi everyone,
my name is Jon and I work on machine learning for sensor systems,
including condition monitoring of machines (at [Soundsensing](https://www.soundsensing.no/)),
and an open-source TinyML library called [emlearn](https://github.com/emlearn/emlearn-micropython).

The usecases we care about often require high quality data and low-power.
While trying to use MicroPython in this area, I have noticed one limitation in the ecosystem:
The lack of FIFO support in IMU drivers.

So here is my attempt at raising this concern, and how we could address it :)

## Current state of IMU drivers

Most of the IMU drivers out there today have functions to read the current values
of the accelerometer/gyro/magnetometer data.
Each call of the function returns a single datapoint in time.
Example: `(x, y, z) = get_xyz()`.

When one only does a single readings of the data, this works rather OK.
For example to check the current orientation of a rarely moving object, say for asset tracking.

However many applications requires continious sampling of several samples.
Examples includ tracking a moving object, detect gestures,
implement a regulation loop (PID), measure vibrations etc.

## Problems with continious sampling using repeated polling

Here is example code for continious sampling,
at a slow 25 Hz and processing 1 second time-windows.
This could for example be human activity detection on a smartwatch,
gesture detection for a "magic wand",
animal activity detection on a LoRa tracker, et.c.

```
    SAMPLERATE = 25
    THRESHOLD = 25 
    sensor = SomeIMU(i2c=machine.I2C(...))
    samples = []
    while True:
        t = time.ticks_ms()
        xyz = sensor.get_xyz()
        samples.append(xyz)

        if len(samples) == THRESHOLD:
            process_samples(samples)
            samples = []

        # Try compensate for execution time causing variation in data sampling time
        # But - if more than 1/SAMPLERATE (4ms for 25 Hz) - will miss the entire timestep...
        time_spent = time.ticks_diff(time.ticks_ms(), t)

        # Very little benefit from lightsleep on very short durations
        wait_time = max(1000/SAMPLERATE - time_spent, 0)
        time.sleep_ms(wait_time)
```

Doing continious sampling by fetching single samples is less-than-ideal for several reasons:

1. Any variation/delay in time of reading causes uneven sampling of data. Jitter/noise.
2. Application processor must be constantly running. High power consumption.
3. Hard for the program to do other work concurrently, while ensuring 1)

These problems become particualy painful at high samplerates,
or in low-power / battery-powered scenarios,
or when trying to do multiple things at the same time (concurrency).

These concerns are generally independent of the programming language (more of a system design issue).
However - there are a couple of reasons why it can easily be more problematic in MicroPython:

1. Variable delays in execution time due to **garbage collection**.
When MicroPython code executes, there will at some point be a need to do some garbage collection.
The likelihood increases with how much allocations the code does,
but can generally happen at any point in time.
One garbage collection run may take many milliseconds, and will cause the data to be sampled at the wrong time.
This can mess up the results of many types of analysis - because they rely on regularly sampled data.
Most methods from Digital Signal Processing, control theory, or Machine Learning.

2. Execution time might be too slow for high samplerates.
To read at 1kHz+, code would need to spend under 1 ms per iteration,
which can be hard to achieve in MicroPython.

## The solution: Use the FIFO buffer of the sensor

Any recent digital MEMS IMU will have a buffer that it can collect measurements into.
Typically this is referred to as a FIFO, since it is a First In First Out type of buffer/queue.

For example the LIS2dh/LIS3dh acceleometers have a 32 samples deep FIFO for its 3 axes.
The BMA421 can store up to 1024 bytes, allowing to store 170 samples.

When enabling FIFO mode, the sensor itself is responsible for timing the sampling.
This means that.
And our application code can then .
As long as one reads the FIFO before it overflows, one gets perfectly sampled data - with minimum effort.

## How to implement

To support this kind of usage, the IMU driver should have:

- A function to enable FIFO sampling mode
- A function to read the FIFO fill level
- A function to read data from the FIFO. A)
- Optionally: A function to enable interrupts. B)

A) I recommend that the data reading returns the raw bytes.
This enables the application to decide when/if to convert to a physical unit such as gravity.

B) Most IMUs also support triggering an interrupt
when the FIFO fill level has reached a certain threshold (watermark).
This can be useful to get absolutely the most out of, but is not critical.
One can still get most of the benefits still by polling the FIFO level
and triggering readout based on that.
It might also be that the interrupt pin is not connected, only I2C/SPI pins,
so one should still support usage without interrupts.

## Application pseudo code

Here is some illustrative code for how such a driver could be used in the same scenario as above.


```
    SAMPLERATE = 25
    THRESHOLD = 25 # NOTE: Should be max 75% of IMU FIFO capacity
    imu = SuperIMU2k(i2c=machine.I2C(...), odr=SAMPLERATE)

    while True:
        level = imu.get_fifo_level()
        if level < THRESHOLD:
            samples = imu.read_fifo_data(THRESHOLD)        

            # Can take up to 100 ms without needing to compensate sleep time
            # Or up to 1000ms if one does the compensation
            process_samples(samples) 

        machine.lightsleep(100)
```

This code will spend 90-99% of the time in lightsleep, for 10-100x power savings.
It will also just work if using high samplerates (1000 Hz+).
And the same approach will work to collect data from multiple sensors at the same time.

## Call to Action: Use the FIFO!

So next time you are implementing a driver for an IMU, please consider supporting using the FIFO!
It will make the driver much more powerful and useable for a wider range of scenarios,
enabling people to get higher data quality and lower power consumption.


## Drivers with FIFO support

Here is an initial start of divers that support FIFO.
Please let us know if you have made, or found more, and we can link them here!

- [BMA423](https://github.com/antirez/bma423-pure-mp/pull/5)
- [LIS2DH/LIS3DH](). FIXME: publish a gist


