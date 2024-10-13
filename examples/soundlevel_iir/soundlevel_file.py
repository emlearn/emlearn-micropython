
import sys
import struct
import array

from soundlevel import SoundlevelMeter

# TODO: create/use a standard wavefile module, installed via mip

def read_wave_header(wav_file):
    # based on https://github.com/miketeachman/micropython-i2s-examples/blob/master/examples/wavplayer.py
    # Copyright (c) 2022 Mike Teachman

    chunk_ID = wav_file.read(4)
    if chunk_ID != b"RIFF":
        raise ValueError("WAV chunk ID invalid")
    chunk_size = wav_file.read(4)
    format = wav_file.read(4)
    if format != b"WAVE":
        raise ValueError("WAV format invalid")
    sub_chunk1_ID = wav_file.read(4)
    if sub_chunk1_ID != b"fmt ":
        raise ValueError("WAV sub chunk 1 ID invalid")
    sub_chunk1_size = wav_file.read(4)
    audio_format = struct.unpack("<H", wav_file.read(2))[0]
    num_channels = struct.unpack("<H", wav_file.read(2))[0]

    sample_rate = struct.unpack("<I", wav_file.read(4))[0]
    byte_rate = struct.unpack("<I", wav_file.read(4))[0]
    block_align = struct.unpack("<H", wav_file.read(2))[0]
    bits_per_sample = struct.unpack("<H", wav_file.read(2))[0]

    # usually the sub chunk2 ID ("data") comes next, but
    # some online MP3->WAV converters add
    # binary data before "data".  So, read a fairly large
    # block of bytes and search for "data".

    binary_block = wav_file.read(200)
    offset = binary_block.find(b"data")
    if offset == -1:
        raise ValueError("WAV sub chunk 2 ID not found")

    first_sample_offset = 44 + offset

    return num_channels, sample_rate, bits_per_sample, first_sample_offset

    
def int16_from_bytes(buf, endianness='<'):
    
    fmt = endianness + 'h'
    gen = (struct.unpack(fmt, buf[i:i+3])[0] for i in range(0, len(buf), 2))
    arr = array.array('h', gen)
    assert len(arr) == len(buf)//2
    return arr


def read_wav(file, samplerate, frames):
    """
    """

    file_channels, file_sr, file_bitdepth, data_offset = read_wave_header(file)
    assert samplerate == file_sr, (samplerate, file_sr)
    # only int16 mono is supported
    assert file_channels == 1, file_channels
    assert file_bitdepth == 16, file_bitdepth

    #samples = array.array('h', (0 for _ in range(frames)))
    file.seek(data_offset)

    while True:
        data = file.read(2*frames)
        read_frames = len(data)//2
        samples = int16_from_bytes(data)
        yield samples


def process_file(audiofile, mic_sensitivity,
        chunk_duration=0.125,
        frequency_weighting='A',
        time_integration=0.125,
        summary_interval=10.0):
    """Compute soundlevels for a .wav file"""

    # NOTE: only 16 kHz sample rate supported
    SR = 16000
    chunk_size = int(SR * chunk_duration)
    meter = SoundlevelMeter(chunk_size, SR, mic_sensitivity,
        frequency_weighting=frequency_weighting,
        time_integration=time_integration,
        summary_interval=summary_interval,
    )

    t = 0.0 # seconds
    for chunk in read_wav(audiofile, SR, frames=chunk_size):

        short_leq = meter.process(chunk)
        dt = len(chunk)/float(SR)
        t += dt

        out = dict(time=t, short_leq=short_leq)

        if len(meter._summary_queue):
            summaries = meter._summary_queue.pop()
            out.update(summaries)

        yield out


def main():

    import json
    mic_sensitivity = 0.0

    if len(sys.argv) != 2:
        raise ValueError('Usage: micropython soundlevel_file.py AUDIO.wav')

    audio_path = sys.argv[1]
    with open(audio_path, 'rb') as audiofile:
        measurements = process_file(audiofile, mic_sensitivity)
        for m in measurements:
            cleaned = { k: round(v, 4) for k, v in m.items() }
            serialized = json.dumps(cleaned)
            print(serialized)

if __name__ == '__main__':
    main()
    
