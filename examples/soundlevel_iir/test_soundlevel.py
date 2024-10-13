
import struct
import array

from soundlevel import SoundlevelMeter
from soundlevel_file import process_file

def test_soundlevel_time_integration():
    """ """

    with open('out.csv', 'w') as out:
        with open('test_burst.wav', 'rb') as f:

    # FIXME: check that the level drops off at the expected rate

# TODO: add test for A weighting filter


if __name__ == '__main__':
    test_soundlevel_time_integration()
    
