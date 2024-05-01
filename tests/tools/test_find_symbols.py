
from tools.find_symbols import nm_find_symbol

import pytest

nm_find_symbols_valid = [
    ('00000000 T _Unwind_GetTextRelBase', ('T', '_Unwind_GetTextRelBase')),
    ('0000000000000050 r .LC12', ('r', '.LC12')),
    ('                 U _GLOBAL_OFFSET_TABLE_', ('U', '_GLOBAL_OFFSET_TABLE_')),
]

@pytest.mark.parametrize('data', nm_find_symbols_valid)
def test_nm_find_symbols(data):
    
    s, (expect_kind, expect_name) = data
    found = nm_find_symbol(s)
    expect = {
        'kind': expect_kind,
        'symbol': expect_name,
    }
    assert found == expect

