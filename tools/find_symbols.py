
"""
Tool to help find which object files (.o) defines certain symbols

In particular made to be used to find soft-floating-point functions from libgcc.a
for use when building MicroPython native modules with mpy_ld.py

License: MIT
Copyright: Jon Nordby
"""

import subprocess
import glob
import re
import os.path

import pandas

def find_objects_dir():

    # Find in a directory of .o files extracted from libgcc.a
    dir = '.'
    for filename in glob.glob('*.o', root_dir=dir):
    
        out = subprocess.check_output([nm_bin, filename]).decode('utf-8')
        if f'{symbol}' in out:
            print(filename)
            print(out)


def nm_find_symbol(s : str):
    """
    For example on one of these forms

    00000000 T _Unwind_GetTextRelBase
             U _Unwind_VRS_Pop
    0000000000000050 r .LC12
                     U _GLOBAL_OFFSET_TABLE_
    """
    
    # drop address
    tok = s.strip().split(' ')
    if len(tok) == 3:
        c = ' '.join(tok[1:])
    else:
        c = ' '.join(tok[0:])

    regex = r"(\w) (.+)"
    match = re.match(regex, c)
    if match is None:
        return None

    kind, symbol = match.groups()
    o = {
        #'address': address,
        'kind': kind,
        'symbol': symbol,
    }
    return o

def nm_parse_output(s : str):

    lines = s.split('\n')

    out = []
    object_file : str = None

    for line in lines:
        stripped = line.strip()
        symbol = nm_find_symbol(line)
        if stripped.endswith('.o:'):
            # section enter, mark which file it is
            object_file = line.rstrip(':')
        elif symbol:
            # a symbol line, collect
            assert object_file, 'must be inside a object section'
            symbol.update({'file': object_file})
            out.append(symbol)
        elif stripped == '':
            # section end
            object_file = None
        else:
            #print('unknown', line)
            assert False, ('unknown line', line)

    return out

    

def find_symbols(archive : str, symbols : list[str], nm_bin='arm-none-eabi-nm') -> list[str]:

    # Run "nm" to get the symbol information from an archive file
    args = [
        nm_bin,
        archive,
    ]
    cmd = ' '.join(args)
    #print('RUN', cmd)
    stdout = subprocess.check_output(args).decode('utf-8')
    out = nm_parse_output(stdout)

    # create a mapping of symbol name -> object file name
    defined_symbols_file = {}
    for s in out:
        kind = s['kind']
        symbol = s['symbol']
        filename = s['file']
        if kind not in ('T', 't'):
            continue

        #assert symbol not in defined_symbols_file, (symbol, filename, defined_symbols_file[symbol])
        defined_symbols_file[symbol] = filename

    # lookup the symbols of interest
    found = [ defined_symbols_file.get(s) for s in symbols ]

    # try fuzzy match
    for want, f in zip(symbols, found):
        if f:
            continue
        for c, f in defined_symbols_file.items():
            if want in c:
                print('c', want, c, f)


    assert len(found) == len(symbols)
    return found    


def parse(args=None):
    import argparse
    parser = argparse.ArgumentParser(description='Find missing symbols')
    a = parser.add_argument

    a('--archive', type=str, metavar='FILE', default=None,
      help='Archive to look in')
    a('--symbols', type=str, default=None,
      help='Symbols to look for')
    a('--nm-bin', type=str, default='nm',
      help='Which "nm" to use')

    parsed = parser.parse_args(args)
    parsed.symbols = parsed.symbols.split(',')
    return parsed

def main():
    args = parse()

    archive_path = args.archive
    assert os.path.exists(archive_path), archive_path

    found = find_symbols(archive_path, args.symbols, nm_bin=args.nm_bin)
    print(found)


if __name__ == '__main__':
    main()
