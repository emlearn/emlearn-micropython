
import time
from machine import Pin


class Button():
    
    SHORT_PRESS = 1
    LONG_PRESS = 2

    def __init__(self, pin, long_press=3000):
        self.pin = pin
        self.start_press = None
        self.long_press = long_press
        
    def read(self):
       
        t = time.time()
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

button = Button(pin=Pin(24, Pin.IN, Pin.PULL_UP))

PREDICT_MODE = 11
TRAIN_MODE = 12

mode = PREDICT_MODE
last_press = None
times = []
SEQUENCE_LENGTH = 6
FEATURES_LENGTH = SEQUENCE_LENGTH-1 # n-1 distances between events in sequence
TRAINING_SEQUENCES = 3

def make_model():
    m = emlneighbors.new(FEATURES_LENGTH, TRAINING_SEQUENCES)
    return m

import emlneighbors
model = make_model()


while True:

    button_event = button.read()
    t = time.ticks_ms()
    
    if button_event == Button.LONG_PRESS:
        # switch mode
        if mode == PREDICT_MODE:
            mode = TRAIN_MODE
            model = make_model() # forget old
        else:
            mode = PREDICT_MODE
        print('switch-mode-to', t, mode) 
        
    elif button_event == Button.SHORT_PRESS:
        
        if last_press is None:
            last_press = t
        else:
            duration = (t - last_press)
            print('distance', t, button_event)
            
 
                
            if mode == PREDICT_MODE:
                
                # shift current input sequence one over
                if len(times) == FEATURES_LENGTH:
                    times = times[1:]
                times.append(duration)

                if len(times) == FEATURES_LENGTH:
                    # check it
                    f = array.array('h', times)
                    out = model.predict(f)
                    # FIXME: compute distances instead
                    print('ttt', times)
                    
            else: # TRAIN_MODE
                
                times.append(duration)
                if len(times) == FEATURES_LENGTH:
                    # add this example
                    f = array.array('h', times)
                    model.add_item()
                    times = []
                else:
                    # wait for sequence to complete
                    pass
            
            last_press = t

    time.sleep_ms(10)

            