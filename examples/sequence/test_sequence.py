
import os

from sequence_lock import SequenceLock, \
    TRAINING_STATE, UNLOCKED_STATE, LOCKED_STATE, \
    TRIGGER_EVENT, MODE_SWITCH_EVENT, \
    file_exists

def test_startup_no_model():
    lock = SequenceLock()
    lock.run(0, None)
    assert lock.state == TRAINING_STATE, lock

def test_startup_with_model():
    model_path = 'foo_model.csv'
    lock = SequenceLock(model_path=model_path)

    # initial state
    lock.run(0, None)
    assert lock.state == LOCKED_STATE, lock

def test_mode_switch():
    lock = SequenceLock()

    # training -> locked
    lock.run(10, MODE_SWITCH_EVENT)
    assert lock.state == LOCKED_STATE, lock
    # locked -> training
    lock.run(20, MODE_SWITCH_EVENT)
    assert lock.state == TRAINING_STATE, lock

def test_training():     
    # start without model

    model_path = 'train_save_model.csv'
    if file_exists(model_path):
        os.unlink(model_path)

    lock = SequenceLock(model_path=model_path)
    assert lock.state == TRAINING_STATE, lock

    

    # FIXME: this sequence should be one shorter?
    triggers = [ 10, 100, 100, 100, 100, 300, 300 ]

    # first example. Training not complete
    start = 1000
    for t in triggers:
        start += t
        lock.run(start, TRIGGER_EVENT)
    assert lock.state == TRAINING_STATE, lock
    assert lock.training_items == 1, lock

    # second example. Training done
    start = 3000
    for t in triggers:
        start += t
        lock.run(start, TRIGGER_EVENT)
    assert lock.state == LOCKED_STATE, lock

    model_exists = file_exists(model_path)
    assert model_exists

def test_unlock():
    model_path = 'foo_model.csv'
    lock = SequenceLock(model_path=model_path)
    assert lock.state == LOCKED_STATE, lock

    # invalid sequence. Stays locked
    invalid = [ 10, 10000, 100, 100, 300, 300 ]
    start = 10000
    for t in invalid:
        lock.run(start+t, TRIGGER_EVENT)
    assert lock.state == LOCKED_STATE, lock

    # Valid sequence. Unlocks
    valid = [ 10, 100, 100, 100, 300, 300 ]
    start = 20000
    for t in valid:
        lock.run(start+t, TRIGGER_EVENT)
    assert lock.state == UNLOCKED_STATE, lock

    # Stays unlocked for a while
    start = lock.state_switch_time
    lock.run(start+1000, None)
    assert lock.state == UNLOCKED_STATE, lock
    # Then locks again
    lock.run(start+4000, None)
    assert lock.state == LOCKED_STATE, lock

test_startup_no_model()
test_mode_switch()
test_training()
test_startup_with_model()
test_unlock()
