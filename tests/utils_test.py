from instacartlib.utils import timer_contextmanager
from instacartlib.utils import dummy_contextmanager
from instacartlib.utils import format_size
from instacartlib.utils import get_df_info

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

