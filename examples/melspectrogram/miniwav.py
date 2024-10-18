
import sys
import struct
import array

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

    return num_channels, sample_rate, bits_per_sample, first_sample_offset, byte_rate


def get_wav_samples(path):

    with open(path, 'rb') as wav_file:
        header = wav_file.read(45)

        channels = struct.unpack_from('<H', header, 22)[0]
        samplerate = struct.unpack_from('<I', header, 24)[0]
        bitdepth = struct.unpack_from('<H', header, 34)[0]
        data_size = struct.unpack_from('<I', header, 40)[0]

    bytes_per_sample = channels * (bitdepth//8)
    samples = data_size // bytes_per_sample

    return samples


def int16_from_bytes(buf, endianness='<'):
    
    fmt = endianness + 'h'
    gen = (struct.unpack(fmt, buf[i:i+3])[0] for i in range(0, len(buf), 2))
    arr = array.array('h', gen)
    assert len(arr) == len(buf)//2
    return arr


def read_wav(file, samplerate, frames):
    """
    """

    file_channels, file_sr, file_bitdepth, data_offset, _ = read_wave_header(file)
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

