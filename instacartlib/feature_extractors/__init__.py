import logging as _logging
from pathlib import Path as _Path
from importlib import import_module as _import_module


_logger = _logging.getLogger(__name__)
_pwd = __path__[0]

exports = {}

for _path in sorted(_Path(_pwd).iterdir()):
    if (_path.is_dir() or
        _path.suffix != '.py' or
        _path.name in ['__init__.py']
    ):
        continue

    try:
        _module = _import_module(f'.{_path.stem}', package=__name__)
        if type(_module.exports) != dict: #pragma: no cover
            # tested in `tests/plugins/__init__.py` copy
            raise TypeError(f'"exports" attribute expected to be dictionary, '
                f'got: {type(_module.exports)}')
    except Exception as e: #pragma: no cover
        # tested in `tests/plugins/__init__.py` copy
        _logger.warning(f'Failed to import ".{_path.stem}" from package '
            f'"{__name__}". Caused by exception: {e!r}')
        continue

    for _name, _obj in _module.exports.items():
        _export_name = f'{_path.stem}.{_name}'
        exports[_export_name] = _obj


