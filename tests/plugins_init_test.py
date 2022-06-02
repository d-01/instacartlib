
import shutil
import filecmp
import os

import pytest


# The scope prevents multiple calls to fixture if there are multiple tests that 
# depend on it.
@pytest.fixture(scope='session')
def plugins():
    src = 'instacartlib/feature_extractors/__init__.py'
    dst = 'tests/plugins/__init__.py'

    if not (os.path.exists(dst) and filecmp.cmp(src, dst)):
        shutil.copyfile(src, dst)

    from tests import plugins
    return plugins
    

def test_plugins_module(plugins):
    assert hasattr(plugins, 'exports')
    assert type(plugins.exports) == dict
    assert set(plugins.exports.keys()) == {
        '000_test_plugin.function_A',
        '000_test_plugin.function_B',
        '000_test_plugin.ClassA',
        '001_test_plugin.function_C',
    }
