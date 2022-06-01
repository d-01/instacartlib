

## Plugin system

*Plugin* is a python module located in `instacartlib/plugins` folder. Plugin should have global variable `exports` with a list of exported functions:

```python
# 000_ui_freq.py

def freq(df_trns):
    return df_trns.groupby(['uid', 'iid'], sort=False).size().to_frame('freq')

exports = [freq]
```

All exported functions will be available in `plugins` variable. For example, given this files structure:
```
instacartlib/
  plugins/
    000_ui_freq.py
    001_ui_avg_cart_pos.py
```

Import will be:

```python
>>> from instacartlib.plugins import plugins
>>> plugins
{'000_ui_freq.freq': <function instacartlib.plugins.000_ui_freq.freq(df_trns)>,
 '001_ui_avg_cart_pos.avg_cart_pos': <function instacartlib.plugins.001_ui_avg_cart_pos.avg_cart_pos(df_trns)>}
```
Were key is `<plugin_name>.<function_name>`.
