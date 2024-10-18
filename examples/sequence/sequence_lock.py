
import os
import array
import emlearn_neighbors

TRAINING_STATE = 'training'
UNLOCKED_STATE = 'unlocked'
LOCKED_STATE = 'locked'
MODE_SWITCH_EVENT = 'mode-switch'
TRIGGER_EVENT = 'trigger'

def file_exists(filename):
    try:
        return (os.stat(filename)[0] & 0x4000) == 0
    except OSError:
        return False

class SequenceLock():
    
    def __init__(self,
        sequence_length=6,
        model_path='sequence_model.csv',
        unlock_time=3000,
        ticks=-1):

        # config
        self.sequence_length = sequence_length
        self.training_examples = 2
        self.model_path = model_path
        self.unlock_time = unlock_time

        # state
        self.state = None
        self.state_switch_time = None
               
        self.model = None
        self.training_items = 0
        
        self.last_press = None
        self.times = []
        self.threshold = 500 # FIXME: learn from data

        if self.model_path and file_exists(self.model_path):
            self.load_model()
            self._set_state(LOCKED_STATE, ticks)
        else:
            self._set_state(TRAINING_STATE, ticks)

    def __repr__(self):
        keys = [ 'state', 'state_switch_time', 'last_press', 'training_items', 'times' ]
        out = []
        for k in keys:
            v = getattr(self, k)
            out.append(f'{k}={v}')
        return 'SequenceLock ' + ' '.join(out)

    def load_model(self):
        assert self.model_path
        self._reset_model()

        with open(self.model_path, 'r') as f:
            for line in f:
                line = line.rstrip('\r')
                line = line.rstrip('\n')
                tok = line.split(',')
                values = [ int(v) for v in tok ]
                #print('load-model-line', values)
                a = array.array('h', values)
                self.model.additem(a, 0)

    def save_model(self):
        assert self.model_path
        features = self.sequence_length-1

        with open(self.model_path, 'w') as f:
            for n in range(self.training_examples):
                item = array.array('h', [0] * features)
                self.model.getitem(n, item)
                values = ','.join([ str(v) for v in item ])
                #print('save-model-line', values)
                f.write(values)
                f.write('\n')


    def run(self, t, event):
        # Delegate based on which state we are in
        if self.state == LOCKED_STATE:
            self._run_locked_state(t, event)
        elif self.state == TRAINING_STATE:
            self._run_training_state(t, event)
        elif self.state == UNLOCKED_STATE:
            self._run_unlocked_state(t, event)
        else:
            raise Exception("invalid state")

    def _run_training_state(self, t, event):
        if event == MODE_SWITCH_EVENT:
            self._set_state(LOCKED_STATE, t)
            return

        if event == TRIGGER_EVENT:
            if self.last_press is None:
                self.last_press = t
                return # ?

            if (t - self.state_switch_time) < 400:
                # Likely duplicate event from state transition. Ignore
                return

            duration = (t - self.last_press)
            self.times.append(duration)
            print(t, 'training-trigger', len(self.times), duration)
            
            if len(self.times) == self.sequence_length:
                # add this example
                self.times = self.times[1:]
                print(t, 'train-add-item', self.training_items, self.times)                
                self.model.additem(array.array('h', self.times), 0)
                self.training_items += 1
                current_times = self.times
                self.times = []
                
                assert self.training_examples == 2, 'Logic assumes 2 training examples'
                if self.training_items == self.training_examples:

                    train_distances = self._get_distances(current_times)
                    print(t, 'training-done', train_distances)

                    # TODO: check that variance is not too high
                    # FIXME: set a threshold based on variance

                    self.save_model()
                    self.load_model()           
                    self._set_state(LOCKED_STATE, t)

            else:
                # wait for input sequence to complete
                pass

            self.last_press = t


    def _run_locked_state(self, t, event):
        if event == MODE_SWITCH_EVENT:
            self._set_state(TRAINING_STATE, t)
            return

        if event == TRIGGER_EVENT:
            if self.last_press is None:
                self.last_press = t
                return

            duration = (t - self.last_press)
            print(t, 'locked-trigger', len(self.times), duration)
            
            features_length = self.sequence_length - 1
            # shift current input sequence one over
            if len(self.times) == features_length:
                self.times = self.times[1:]
            self.times.append(duration)

            if len(self.times) == features_length:
                # check it
                
                distances = self._get_distances(self.times)
                distance = max(distances)
                allow_unlock = distance < self.threshold
                print(t, 'try-unlock', self.times, distances, distance, self.threshold)
                if allow_unlock:
                    self._set_state(UNLOCKED_STATE, t)
            
            self.last_press = t
            
    def _run_unlocked_state(self, t, event):
        if (t - self.state_switch_time) > self.unlock_time:
            self._set_state(LOCKED_STATE, t)
        
        # NOTE: I/O of actually triggering unlocking is handled outside

    def _set_state(self, target, ticks):
        print(ticks, 'set-state-to', target)

        if target == TRAINING_STATE:
            self._reset_model() # forget old model

        self.state_switch_time = ticks
        self.times = []
        #self.last_press = None
        self.state = target


    def _reset_model(self):
        
        items = self.training_examples
        features = self.sequence_length-1
        self.model = emlearn_neighbors.new(items, features, items)

        # XXX: could be dropped if emlearn_neighbors allowed accessing n_items
        self.training_items = 0 

    def _get_distances(self, features):
        
        # convert to fixed-size int16 (halfword) array 
        f = array.array('h', features)
        
        # Run inference
        _ = self.model.predict(f)
        
        # Get distances for all items in model
        distances = []
        for i in range(self.training_examples):
            data_item, distance, label = self.model.getresult(i)
            distances.append(distance)
            
        return distances
