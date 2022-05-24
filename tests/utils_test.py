from instacartlib.utils import timer_contextmanager
from instacartlib.utils import dummy_contextmanager

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
