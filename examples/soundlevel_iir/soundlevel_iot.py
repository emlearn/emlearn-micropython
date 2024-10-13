
"""
IoT-enable sound level meter example - sends data to IoT platform
"""

import machine
import network
import time
import array
from machine import Pin, I2S

from soundlevel import SoundlevelMeter
import secrets

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
chunk_duration = 0.125
chunk_samples = int(AUDIO_SAMPLERATE * chunk_duration)
mic_samples = array.array('h', (0 for _ in range(chunk_samples))) # int16
# memoryview used to reduce heap allocation in while loop
mic_samples_mv = memoryview(mic_samples)

meter = SoundlevelMeter(buffer_size=chunk_samples,
    samplerate=AUDIO_SAMPLERATE,
    mic_sensitivity=MIC_DBFS,
    time_integration=0.125,
    frequency_weighting='A',
    summary_interval=10.0,
)

wlan = network.WLAN(network.STA_IF)

if secrets.BLYNK_AUTH_TOKEN:
    from iot_blynk import BlynkClient
    BLYNK_PIN_MAPPING = dict(Leq='v2', Lmin='v3', Lmax='v4')
    api = BlynkClient(token=secrets.BLYNK_AUTH_TOKEN)
else:
    raise ValueError('No IoT API configured')

def audio_ready_callback(arg):
    global soundlevel_db
    start_time = time.ticks_ms()

    meter.process(mic_samples)

    duration = time.ticks_diff(time.ticks_ms(), start_time)
    if duration >= 125:
        print('warn-audio-processing-too-slow', time.ticks_ms(), duration)

    # re-trigger audio
    num_read = audio_in.readinto(mic_samples_mv)

def wifi_connect():

    wlan.active(True)
    wlan.connect(secrets.WIFI_SSID, secrets.WIFI_PASSWORD)


def main():
    STATUS_UPDATE_INTERVAL = 5.0
    next_status_update = time.time() + STATUS_UPDATE_INTERVAL

    # Setting a callback function makes the readinto() method non-blocking
    audio_in.irq(audio_ready_callback)
    # Start microphonee readout. Callback will re-trigger it
    num_read = audio_in.readinto(mic_samples_mv)
    print('audio-start', num_read)
    
    # Start connecting to WiFi
    wifi_connect()


    while True:

        # check our current status
        if time.time() >= next_status_update:
            print('main-alive-tick', wlan.status())
            # Tiny blink to show we are alive
            #led_pin.value(1)
            #time.sleep_ms(1)
            #led_pin.value(0)

            if not wlan.isconnected():
                print('wifi-reconnect')
                wlan.active(False)
                wifi_connect()

            next_status_update = time.time() + STATUS_UPDATE_INTERVAL

        # check for soundlevel data ready to send
        if len(meter._summary_queue) and wlan.isconnected():
            print('send-metrics', len(meter._summary_queue))
            m = meter._summary_queue.pop()
            vv = { pin: m[key] for key, pin in BLYNK_PIN_MAPPING.items() }
            values = [ vv ]
            try:
                api.post_telemetry(values)
            except Exception as e:
                print('post-error', e)


        time.sleep_ms(10)


if __name__ == '__main__':
    main()


