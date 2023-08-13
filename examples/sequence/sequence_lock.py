
import array
import emlneighbors

TRAINING_STATE = 'training'
UNLOCKED_STATE = 'unlocked'
LOCKED_STATE = 'locked'
MODE_SWITCH_EVENT = 'mode-switch'
TRIGGER_EVENT = 'trigger'

class SequenceLock():
    
    def __init__(self, sequence_length=6):

        # config
        self.sequence_length = sequence_length
        self.training_examples = 2
        
        # state
        self.state = None
        self.state_switch_time = None
               
        self.model = None
        self.training_items = 0
        
        self.last_press = None
        self.times = []

        self._set_state(TRAINING_STATE, -1)
    
    def __repr__(self):
        keys = [ 'state', 'state_switch_time', 'last_press', 'training_items', 'times' ]
        out = []
        for k in keys:
            v = getattr(self, k)
            out.append(f'{k}={v}')
        return 'SequenceLock ' + ' '.join(out)

    def load_model(self, f):
        # FIXME: actually load the data into model

        self._set_state(LOCKED_STATE)

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
            self._set_state(UNLOCKED_STATE)

        if event == TRIGGER_EVENT:
            if self.last_press is None:
                self.last_press = t
                return # ?

            duration = (t - self.last_press)
            self.times.append(duration)
            
            if len(self.times) == self.sequence_length:
                # add this example
                self.times = self.times[1:]
                print(t, 'train-add-item', self.training_items, self.times)                
                self.model.additem(array.array('h', self.times), 0)
                self.training_items += 1
                current_times = self.times
                self.times = []
                
                # TODO: check that variance is not too high
                # FIXME: set a threshold based on variance
                assert self.training_examples == 2, 'Logic assumes 2 training examples'
                if self.training_items == self.training_examples:
                    self._set_state(LOCKED_STATE, t)
                    train_distances = self._get_distances(current_times)
                    print(t, 'training-done', train_distances)
            else:
                # wait for input sequence to complete
                pass

            self.last_press = t


    def _run_locked_state(self, t, event):
        if event == MODE_SWITCH_EVENT:
            self._set_state(TRAINING_STATE)
            return

        if event == TRIGGER_EVENT:
            if self.last_press is None:
                self.last_press = t
                return

            if (t - self.state_switch_time) < 400:
                # Likely duplicate event from state transition. Ignore
                return

            duration = (t - self.last_press)
            print(t, 'distance', mode, event, len(self.times))
            
            features_length = self.sequence_length - 1
            # shift current input sequence one over
            if len(self.times) == features_length:
                self.times = self.times[1:]
            self.times.append(duration)

            if len(self.times) == features_length:
                # check it
                
                distances = self._get_distances(f)
                print(t, 'try-unlock', self.times, distances)
                # FIXME: check wrt threshold. Transition to
                # self._set_state(UNLOCKED_STATE)
            
            self.last_press = t
            
    def _run_unlocked_state(self, t, event):
        if (t - self.state_switch_time) > 1000:
            self._set_state(LOCKED_STATE)
        
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
        self.model = emlneighbors.new(items, features, items)

        # XXX: could be dropped if emlneighbors allowed accessing n_items
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
