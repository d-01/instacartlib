
import shutil
import filecmp
import os

import pytest


# The scope prevents multiple calls to fixture if there are multiple tests that
# depend on it.
@pytest.fixture(scope='session')
def copy_init():
    src = 'instacartlib/feature_extractors/__init__.py'
    dst = 'tests/plugins/__init__.py'

    if not (os.path.exists(dst) and filecmp.cmp(src, dst)):
        shutil.copyfile(src, dst)


@pytest.mark.usefixtures('copy_init')
def test_plugins_module(caplog):
    from tests import plugins

    assert hasattr(plugins, 'exports')
    assert type(plugins.exports) == dict
    assert set(plugins.exports.keys()) == {
        '000_single_export.function_C',
        '001_multiple_exports.function_A',
        '001_multiple_exports.function_B',
        '001_multiple_exports.ClassA',
    }

    assert "BadModuleError('i am bad')" in caplog.text
    assert ("TypeError('\"exports\" attribute expected to be dictionary"
        in caplog.text)
