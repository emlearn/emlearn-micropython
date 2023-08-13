
import time
import array
from machine import Pin


class Button():
    
    SHORT_PRESS = 1
    LONG_PRESS = 2

    def __init__(self, pin, long_press=3000):
        self.pin = pin
        self.start_press = None
        self.long_press = long_press
        
    def read(self):
       
        t = time.ticks_ms()
        pressed = self.pin.value() == 0 # active low
        
        if self.start_press is None and pressed:
            # press down
            self.start_press = t
        elif self.start_press and not pressed:
            # release
            self.start_press = None
            return Button.SHORT_PRESS
        elif self.start_press and pressed and (t - self.start_press) > self.long_press:
            # long press
            self.start_press = None
            return Button.LONG_PRESS
    
        return None







from sequence_lock import SequenceLock, \
     TRIGGER_EVENT, MODE_SWITCH_EVENT, \
     TRAINING_STATE, UNLOCKED_STATE

def main():
    # App logic setup
    lock = SequenceLock(sequence_length=6, unlock_time=10000)

    # I/O setup
    button = Button(pin=Pin(24, Pin.IN, Pin.PULL_UP))
    mode_pin = Pin(25, Pin.OUT)
    out_pin = Pin(22, Pin.OUT)

    # main loop
    while True:
        # get current time
        t = time.ticks_ms()
        
        # read button, map to our events
        button_event = button.read()
        event = None
        if button_event == Button.LONG_PRESS:
            event = MODE_SWITCH_EVENT
        if button_event == Button.SHORT_PRESS:
            event = TRIGGER_EVENT
        
        # run application logic
        lock.run(t, event)
    
        # set outputs
        mode_pin.value(lock.state == TRAINING_STATE)
        out_pin.value(lock.state == UNLOCKED_STATE)
     
        # wait for next iteration
        time.sleep_ms(10)

main()