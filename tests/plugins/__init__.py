from pathlib import Path as _Path
from importlib import import_module as _import_module

_pwd = __path__[0]

exports = {}

for _path in _Path(_pwd).iterdir():
    if (_path.is_dir() or
        _path.suffix != '.py' or
        _path.name in ['__init__.py']
    ):
        continue

    _module = _import_module(f'.{_path.stem}', package=__name__)
    if not hasattr(_module, 'exports'):
        raise AttributeError(
            f'Module in "{_path}" has no attribute `exports`.')
    try:
        _module_exports = list(_module.exports)
    except TypeError:
        raise TypeError(
            f'Module in "{_path}" expected to has attribute `exports` '
            f'of iterable type: got "{type(_module.exports)}"')
    
    for _obj in _module_exports:
        _export_name = f'{_path.stem}.{_obj.__name__}'
        exports[_export_name] = _obj
    
