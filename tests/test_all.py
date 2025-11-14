

import sys

TEST_MODULES=[
    'test_arrayutils',
    'test_cnn',
    'test_fft',
    'test_iir',
    'test_iir_q15',
    'test_kmeans',
    'test_linreg',
    'test_linreg_california',
    'test_neighbors',
    'test_trees',
]

def main():

    # Find which tests are enabled
    # Default: all
    
    modules = TEST_MODULES
    #if sys.argv 

    passed = 0
    failed = 0

    for module_name in modules:
        mod = None
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
            try:
                test_function()
            except Exception as e:
                print(f'{module_name}.py/{test_name}: FAIL')
                sys.print_exception(e)
                print() # spacing for readability 
                failed += 1
                continue

            print(f'{module_name}.py/{test_name}: PASS')
            passed += 1

    print(f'Passed: {passed}')
    print(f'Failed: {failed}')

    # Let status code reflect number of failures
    return failed

if __name__ == '__main__':
    sys.exit(main())
