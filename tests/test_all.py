
import sys

# Find the module path (architecture+version specific)
sys_mpy = sys.implementation._mpy
mpy_arch = [None, 'x86', 'x64',
    'armv6', 'armv6m', 'armv7m', 'armv7em', 'armv7emsp', 'armv7emdp',
    'xtensa', 'xtensawin', 'rv32imc'][sys_mpy >> 10]
mpy_major = sys_mpy & 0xff
mpy_minor = sys_mpy >> 8 & 3

module_dir = f'{mpy_arch}_{mpy_major}.{mpy_minor}' 

# make sure we can import .mpy modules
sys.path.insert(0, './dist/'+module_dir)

# make sure we can import test files
sys.path.insert(0, './tests')


TEST_MODULES=[
    'test_arrayutils',
    'test_cnn',
    'test_fft',
    'test_iir',
    'test_iir_q15',
    'test_kmeans',
    'test_linreg',
    'test_plsr',
    #'test_linreg_california',
    'test_neighbors',
    'test_trees',
]

def main():

    # Find which tests are enabled
    # Default: all
    
    modules = TEST_MODULES
    if len(sys.argv) >= 2:
        modules = sys.argv[1].split(',') 

    passed = 0
    failed = 0

    for module_name in modules:
        mod = None
        print(f'{module_name}:')
        try:
            mod = __import__(module_name)
        except Exception as e:
            print(f'Error while importing {module_name}:')
            sys.print_exception(e)
            print() # spacing for readability 
            failed += 1
            continue

        module_attributes = dir(mod)
        tests = [ o for o in module_attributes if o.startswith('test_') ]
        for test_name in tests:
            test_function = getattr(mod, test_name)
            print(f'{module_name}.py/{test_name}:')
            try:
                test_function()
            except Exception as e:
                print(f'\tFAIL')
                sys.print_exception(e)
                print() # spacing for readability 
                failed += 1
                continue

            print(f'\t PASS')
            passed += 1

    print(f'Passed: {passed}')
    print(f'Failed: {failed}')

    # Let status code reflect number of failures
    return failed

if __name__ == '__main__':
    sys.exit(main())
