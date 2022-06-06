from instacartlib.utils import timer_contextmanager
from instacartlib.utils import dummy_contextmanager
from instacartlib.utils import format_size
from instacartlib.utils import get_df_info
from instacartlib.utils import split_counter_suffix
from instacartlib.utils import increment_counter_suffix

import pandas as pd
import pytest

def test_timer_contextmanager(capsys):
    with timer_contextmanager('msg 1'):
        pass
    out_1, err_1 = capsys.readouterr()
    assert out_1.startswith('msg 1 (')
    assert out_1.endswith(')\n')
    assert err_1 == ''


def test_dummy_contextmanager(capsys):
    with dummy_contextmanager():
        pass
    outerr = capsys.readouterr()
    assert outerr == ('', '')


@pytest.mark.parametrize("test_input,expected", [
    (0, '0 KB'),
    (1, '1 KB'),
    (1024 + 511, '1 KB'),
    (1024 + 512, '2 KB'),
    (1024 * 999, '999 KB'),
    (1024 * 1000, '1 MB'),
    (1024 * 1024 * 999, '999 MB'),
    (1024 * 1024 * 1000, '1 GB'),
])
def test_format_size(test_input, expected):
    assert format_size(test_input) == expected


def test_get_df_info():
    test_input = pd.DataFrame([0])
    output = get_df_info(test_input)
    assert output.startswith('<DataFrame ')
    assert output.endswith('>')
    assert 'shape=' in output
    assert 'memory=' in output


def test_get_df_info_None():
    with pytest.raises(TypeError):
        get_df_info(None)


@pytest.mark.parametrize("test_input,expected", [
    ('abc_123'  , ['abc'    , '123']),
    ('abc__123' , ['abc_'   , '123']),
    ('abc___123', ['abc__'  , '123']),
    ('_123'     , [''       , '123']),
    ('__123'    , ['_'      , '123']),
    ('abc_1_2'  , ['abc_1'  ,   '2']),
    ('abc_000'  , ['abc'    , '000']),
    ('abc_-0'   , ['abc_-0' ,  None]),
    ('abc_0.0'  , ['abc_0.0',  None]),
    ('abc_0.'   , ['abc_0.' ,  None]),
    ('abc_.0'   , ['abc_.0' ,  None]),
    ('123'      , ['123'    ,  None]),
    ('123_'     , ['123_'   ,  None]),
    ('abc'      , ['abc'    ,  None]),
    ('abc_'     , ['abc_'   ,  None]),
    ('_'        , ['_'      ,  None]),
    ('__'       , ['__'     ,  None]),
    (''         , [''       ,  None]),
])
def test_split_counter_suffix(test_input, expected):
    assert split_counter_suffix(test_input) == expected


@pytest.mark.parametrize("test_input,expected", [
    ('abc_0'    , 'abc_2'    ),
    ('abc__00'  , 'abc__02'  ),
    ('_000'     , '_002'     ),
    ('__0000'   , '__0002'   ),
    ('abc_1_9'  , 'abc_1_11' ),
    ('abc_999'  , 'abc_1001' ),
    ('abc_-0'   , 'abc_-0_2' ),
    ('abc_0.0'  , 'abc_0.0_2'),
    ('abc_0.'   , 'abc_0._2' ),
    ('abc_.0'   , 'abc_.0_2' ),
    ('123'      , '123_2'    ),
    ('123_'     , '123__2'   ),
    ('abc'      , 'abc_2'    ),
    ('abc_'     , 'abc__2'   ),
    ('_'        , '__2'      ),
    ('__'       , '___2'     ),
    (''         , '_2'       ),
])
def test_increment_counter_suffix_2_times(test_input, expected):
    output = increment_counter_suffix(test_input)
    output = increment_counter_suffix(output)
    assert output == expected

