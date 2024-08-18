
import os
import sys

import npyfile

ST_SIZE = 6

def iir_process_file(inp, out, filters, chunksize):

    coefficients_shape, coefficients = npyfile.load(filters)
    print('c', coefficients_shape, coefficients)
    

    # NOTE: filter shape and type checked by the C modules

    # Processing is done by streaming, to keep RAM usage low
    with npyfile.Reader(inp) as reader:

        # Check inputs and setup filters
        print('r', inp, os.stat(inp)[ST_SIZE], reader.shape, reader.typecode, reader.itemsize)

        #if reader.typecode == 'i':
        #    reader.typecode = 'h'
        #    coefficients = array.array('h', coefficients)

        if len(reader.shape) != 1:
            raise ValueError("Input must be 1d")
        if reader.typecode == 'f':
            import emliir
            filter = emliir.new(coefficients)
        elif reader.typecode == 'h':
            import eml_iir_q15
            filter = eml_iir_q15.new(coefficients)
        else:
            raise ValueError("Input must either be float32/f or int16/h")

        with npyfile.Writer(out, reader.shape, reader.typecode) as writer:

            # Apply filters
            for chunk in reader.read_data_chunks(chunksize):
                filter.run(chunk) # operates in-place
                writer.write_values(chunk)

def main():

    args = sys.argv
    if len(args) < 3:
        print('Usage: micropython iir_run.py INPUT.npy OUTPUT.npy FILTERS.npy')

    _, inp, out, filters = args

    chunksize = 100

    iir_process_file(inp, out, filters, chunksize=chunksize)

if __name__ == '__main__':
    main()
    
