
import time
from pynput import keyboard

# TODO: also implement sliding window

class TimeBetweenEvents:

    def __init__(self, max_time=1000e3, min_time=10e3):
        self.max_time = max_time
        self.min_time = min_time
        self.last_time = None
        self.current_state = False

    def push(self, time : int, state : bool):
        """
        Returns None if no event, or time since previous event on event
        """
        #print(time, state, self.last_time, self.current_state)

        # Update state
        # NOTE: since we modify this in the beginning, must use previous_state for logic
        previous_state = self.current_state
        self.current_state = state

        # Initial condition
        if self.last_time is None:
            self.last_time = time
            return None

        # Handle timeouts
        since_last = time - self.last_time
        if since_last >= self.max_time:
            # timeout, do transition
            self.last_time = time
            # NOTE: can be more than self.max_time if function not called regularly
            return since_last

        # trigger on falling edge
        above_minimum = since_last >= self.min_time
        trigger = previous_state and not state and above_minimum
        if trigger:
            self.last_time = time
            return since_last
        else:
            return None





def test_timeouts():
    processor = TimeBetweenEvents(max_time=100, min_time=10)

    emitted = []
    for t in range(0, 1000, 20):
        out = processor.push(t, False)
        if out is not None:
            emitted.append(out)
    assert len(emitted) == 9, (len(emitted), emitted)
    assert emitted == ([processor.max_time] * 9)


def test_time_between_events():
    
    processor = TimeBetweenEvents(max_time=100, min_time=20)

    # input state is False: no emission
    assert processor.push(10, False) is None
    assert processor.push(20, False) is None

    # over max_time, emit even if False
    assert processor.push(140, False) == 140-10

    # first event, emit
    assert processor.push(150, True) is None
    assert processor.push(200, False) == 200-140

    # second event, emit
    assert processor.push(250, True) is None
    assert processor.push(290, False) == 290-200

    # too short time, no emission
    assert processor.push(295, True) is None
    assert processor.push(300, False) is None

    # third event, emit
    assert processor.push(320, True) is None
    assert processor.push(330, False) == 330-290


import pynput
import pynput.keyboard
from pynput.keyboard import Key
import queue
import time

class KeyboardInput():

    def __init__(self, key=Key.space):
        self.key = key

        # pynput listener callbacks are on separate thread
        # use a queue to process the events from there
        self.queue = queue.Queue()

        def on_press(key):
            if key != self.key:
                return
            self.queue.put((time.time(), True))

        def on_release(key):
            if key != self.key:
                return
            self.queue.put((time.time(), False))

        self.listener = pynput.keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.start()

        self.state = False

    def check(self):

        # check if anything has changed
        any_presses = False
        while not self.queue.empty():
            event = self.queue.get_nowait()
            t, state = event
            if state:
                any_presses = True
            self.state = state
            #print('queue-message', t, state, any_presses)

        return any_presses

    def close():
        self.listener.close()

def main():
    max_time = 2e6

    processor = TimeBetweenEvents(max_time=max_time)
    keyboard = KeyboardInput()
    
    start_time = time.time()

    import sys
    path = sys.argv[1]

    with open(path, 'w') as f:

        f.write('event,time,delta\n')

        while True:
            t = (time.time() - start_time) * 1e6
            s = keyboard.check()
            e = processor.push(t, s)

            if e and e < max_time:
                bar = ''.join(['*'] * int(50 * (e / max_time)))
                print('event', round(t/1000.0, 0), round(e/1000.0, 0), bar)

                f.write(f'event,{round(t/1000.0, 3)},{round(e/1000.0, 3)}\n')


            time.sleep(0.001)
    

if __name__ == '__main__':
    main()


