
import sys
import gc

# Find the module path (architecture+version specific)
sys_mpy = sys.implementation._mpy
mpy_arch = [None, 'x86', 'x64',
    'armv6', 'armv6m', 'armv7m', 'armv7em', 'armv7emsp', 'armv7emdp',
    'xtensa', 'xtensawin', 'rv32imc'][sys_mpy >> 10]
mpy_major = sys_mpy & 0xff
mpy_minor = sys_mpy >> 8 & 3

module_dir = f'{mpy_arch}_{mpy_major}.{mpy_minor}'

# make sure we can import .mpy modules (skip with 'nomodules' arg)
if 'nomodules' not in sys.argv:
    sys.path.insert(0, './dist/'+module_dir)

# make sure we can import test files
sys.path.insert(0, './tests')


TEST_MODULES=[
    'test_arrayutils',
    'test_cnn',
    'test_fft',
    'test_iir',
    #'test_iir_q15', # skip, not functional
    'test_kmeans',
    'test_linreg',
    'test_linreg_california',
    'test_logreg',
    'test_logreg_cancer',
    'test_plsr',
    'test_neighbors',
    'test_trees',
    'test_extratrees',
    'test_extratrees_xor',
    'test_extratrees_cancer',
    'test_extratrees_wine',
    'test_plsr_airquality',
    'test_plsr_spectrofood',
]

def main():

    # Find which tests are enabled
    # Default: all
    
    modules = TEST_MODULES
    # Filter out non-test flags from argv
    test_args = [a for a in sys.argv[1:] if a != 'nomodules']
    if len(test_args) >= 1:
        config = test_args[0].split(',')
        skip = [ m[1:] for m in config if m[0] == '-' ]
        add = [ m for m in config if m[0] != '-' ]
        if len(skip):
            modules = [m for m in TEST_MODULES if not m in skip ]
            print('SKIPPING', skip)
        if len(add):
            modules = add
            print('RUN ONLY', add)

    passed = 0
    failed = 0

    # Test markers for external test runners (mpremote, etc.)
    # These allow the runner to know when tests are complete without relying on timeouts
    print('\n=== TEST START ===')

    free = gc.mem_free()
    alloc = gc.mem_alloc()
    print(f"RAM free={free} used={alloc} total={free+alloc}")

    print('sys.path', sys.path)

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

        # Try to free space
        gc.collect()

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

            # Try to free space
            gc.collect()

    print(f'Passed: {passed}')
    print(f'Failed: {failed}')
    print('\n=== TEST END ===')

    # Let status code reflect number of failures
    return failed

if __name__ == '__main__':
    sys.exit(main())
