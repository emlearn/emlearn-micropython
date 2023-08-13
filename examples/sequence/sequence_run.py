
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


# Configuration
SEQUENCE_LENGTH = 6
TRAINING_SEQUENCES = 2


PREDICT_MODE = 11
TRAIN_MODE = 12
mode = None
mode_switch_time = time.ticks_ms()

last_press = None
times = []
button = Button(pin=Pin(24, Pin.IN, Pin.PULL_UP))
model = None

FEATURES_LENGTH = SEQUENCE_LENGTH-1 # n-1 distances between events in sequence

import emlneighbors
def reset_model():
    global model
    global training_items
    model = emlneighbors.new(TRAINING_SEQUENCES, FEATURES_LENGTH, TRAINING_SEQUENCES)
    training_items = 0

def switch_mode(target):
    global times
    global mode_switch_time
    global last_press
    global mode
    t = time.ticks_ms()
    print('switch-mode-to', t, target)
    mode_switch_time = t
    times = []
    #last_press = None
    mode = target

reset_model()
switch_mode(PREDICT_MODE)

def get_distances(features):
    
    # Run inference
    _ = model.predict(features)
    
    # Get distances for all
    distances = []
    for i in range(TRAINING_SEQUENCES):
        data_item, distance, label = model.getresult(i)
        distances.append(distance)
        
    return distances



def median(lst):
    n = len(lst)
    s = sorted(lst)
    return (s[n//2-1]/2.0+s[n//2]/2.0, s[n//2])[n % 2] if n else None


while True:

    button_event = button.read()
    t = time.ticks_ms()
 
    if button_event == Button.LONG_PRESS:
        # switch mode
        if mode == PREDICT_MODE:
            switch_mode(TRAIN_MODE)
            reset_model() # forget old
            
        else:
            switch_mode(PREDICT_MODE)       
        
        if last_press is None:
            last_press = t
        
    elif button_event == Button.SHORT_PRESS:
        
        if t - mode_switch_time < 400:
            # ignore. Likely duplicate from transition
            continue
        
        if last_press is None:
            last_press = t
        else:
            duration = (t - last_press)
            print('distance', t, mode, button_event, len(times))
                        
            if mode == PREDICT_MODE:
                
                # shift current input sequence one over
                if len(times) == FEATURES_LENGTH:
                    times = times[1:]
                times.append(duration)

                if len(times) == FEATURES_LENGTH:
                    # check it
                    f = array.array('h', times)
                    distances = get_distances(f)
                    print('predict', t, times, distances)
                    
            else: # TRAIN_MODE
                
                times.append(duration)
                if len(times) == SEQUENCE_LENGTH:
                    # add this example
                    times = times[1:]
                    print('train-add-item', t, training_items, times)                
                    f = array.array('h', times)
                    model.additem(f, 0)
                    training_items += 1
                    times = []
                    
                    # TODO: check that variance is not too high
                    if training_items == TRAINING_SEQUENCES:
                        switch_mode(PREDICT_MODE)
                        train_distances = get_distances(f)
                        print('training-done', t, train_distances)
                else:
                    # wait for input sequence to complete
                    pass
            
            last_press = t

    time.sleep_ms(10)

            