
import struct 
import array

format_mapping = {
    # npy format => (array.array typecode, itemsize in bytes)
    b'f8': ('d', 8),
    b'f4': ('f', 4),
}

def find_section(data, prefix, suffix):
    start = data.index(prefix) + len(prefix)
    end = start + data[start:].index(suffix)
    
    print('tt', start, data[start:])
    section = data[start:end]
    return section

def array_from_bytes(typecode, buffer):
    # Workaround due 
    return array.array(typecode, buffer)

def read_npy(path):
    """
    References:
    https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
    https://numpy.org/doc/1.13/neps/npy-format.html#a-simple-file-format-for-numpy-arrays
    """
    with open(path, 'rb') as f:

        # Read header data
        expected_header_maxlength = 16*10
        data = f.read(expected_header_maxlength)

        # Check magic
        npy_magic = b'\x93NUMPY'
        magic = data[0:len(npy_magic)] 
        assert magic == npy_magic, magic

        # Check version
        major, minor = struct.unpack_from('BB', data, len(npy_magic))
        if major == 0x01:
            header_length = struct.unpack_from('<H', data, len(npy_magic)+2)[0]
            header_start = len(npy_magic)+2+2
        elif major == 0x02:
            header_length = struct.unpack_from('<I', data, len(npy_magic)+2)[0]
            header_start = len(npy_magic)+2+2
        else:
            raise ValueError("Unsupported npy format version")

        #print('hs', header_start, data[header_start:header_start+header_length])

        # Parse header info
        type_info = find_section(data, b"'descr': '", b"',")
        type_endianess = type_info[0]
        assert type_endianess == b'<'[0], (type_endianess)
        type_format = type_info[1:]
        #print('tt', type_info)

        try:
            typecode, itemsize = format_mapping[type_format]
        except KeyError:
            raise ValueError(f"Unsupported data format: {type_format}")
        
        fortran_order = find_section(data, b"'fortran_order': ", b",")
        assert fortran_order == b'False', fortran_order # only C order supported

        shape_info = find_section(data, b"'shape': (", b"),")
        shape = tuple((int(d) for d in shape_info.split(b',') if d != b''))
        #print('ss', shape_info, shape)
        
        data_start = header_start + header_length
        assert (data_start % 16) == 0, data_start # should always be 16 bytes aligned

        # determine amount of data expected
        total_items = 1
        for d in shape:
            total_items *= d
        total_data_bytes = itemsize * total_items
        
        # read the data
        f.seek(data_start)

        chunksize = 100
        chunksize_bytes = itemsize * chunksize
        read_bytes = 0
        while read_bytes < total_data_bytes:
            sub = f.read(chunksize_bytes)
            arr = array_from_bytes(typecode, sub)
            yield arr
            read_bytes += len(sub)



for s in read_npy('benchmarks/iir/noise.npy'):
    print(s)

    # testcases
    # supported
    # super short file. Single value, or 2x2 array
    # common sizes and common formats. 1d/2d/3 uint8/int16/float

    # unsupported
    # object type
