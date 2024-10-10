
"""
Basic sound level meter example - just prints to the console
"""

import time
import struct
import array

from machine import Pin, I2S

from soundlevel import SoundlevelMeter

# Microphone sensitivity.
# Specifies how to convert digital samples to physical sound level pressure
# Use the dataheet as a starting point. Ideally calibrate with a known sound level
MIC_DBFS=-26 # MSM261S4030H0R

AUDIO_SAMPLERATE = 16000

# NOTE: ensure that the pins match your device
# Example is shown for a LilyGo T-Camera Mic v1.6
audio_in = I2S(0,
    sck=Pin(26),
    ws=Pin(32),
    sd=Pin(33),
    mode=I2S.RX,
    bits=16,
    format=I2S.MONO,
    rate=AUDIO_SAMPLERATE,
    ibuf=40000,
)


# allocate sample arrays
chunk_samples = int(AUDIO_SAMPLERATE * 0.125)
mic_samples = array.array('h', (0 for _ in range(chunk_samples))) # int16
# memoryview used to reduce heap allocation in while loop
mic_samples_mv = memoryview(mic_samples)

soundlevel_db = 0.0

meter = SoundlevelMeter(buffer_size=chunk_samples,
    samplerate=AUDIO_SAMPLERATE,
    mic_sensitivity=MIC_DBFS,
    time_integration=0.125,
    frequency_weighting='A',
)


def audio_ready_callback(arg):
    global soundlevel_db
    start_time = time.ticks_ms()

    db = meter.process(mic_samples)
    soundlevel_db = db

    duration = time.ticks_diff(time.ticks_ms(), start_time)
    if duration >= 125:
        print('warn-audio-processing-too-slow', time.ticks_ms(), duration)

    # re-trigger audio
    num_read = audio_in.readinto(mic_samples_mv)


def main():

    next_display_update = 0.0

    # Setting a callback function makes the readinto() method non-blocking
    audio_in.irq(audio_ready_callback)
    # Start microphonee readout. Callback will re-trigger it
    num_read = audio_in.readinto(mic_samples_mv)
    print('audio-start', num_read)

    while True:
        if time.time() >= next_display_update:
            print(f'Soundlevel: {soundlevel_db:.1f} dB(A)')
            last_display_update = time.time() + 0.100

        time.sleep_ms(10)


if __name__ == '__main__':
    main()


