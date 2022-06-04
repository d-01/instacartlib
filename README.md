

## Feature extractors

*Feature extractor* is a function (or callable object) that takes transactions and products data and returns a dataframe with one or more features. Example:

```python
df_trns = pd.read_fwf(io.StringIO('''
    oid     uid  iord     iid  reord  dow  hour  days_prev  cart_pos
order_A  user_A     1  item_A      0    0     2         -1         1
order_A  user_A     0  item_B      0    0     2         -1         2
order_B  user_A     0  item_A      1    0     2         -1         1
order_C  user_B     1  item_B      0    0     2         -1         1
order_D  user_B     0  item_B      1    0     2         -1         1
'''))

df_prod = pd.DataFrame()  # not used, but argument is required

index = pd.MultiIndex.from_tuples([
    ('user_A', 'item_A'),
    ('user_A', 'item_B'),
    ('user_B', 'item_A'),
    ('user_B', 'item_B'),
], names=['uid', 'iid'])


def freq(ui_index: pd.MultiIndex,
         df_trns: pd.DataFrame,
         df_prod: pd.DataFrame) -> pd.DataFrame:
    return (df_trns
        .groupby(['uid', 'iid'], sort=False)
        .size()
        .to_frame('freq')
        .reindex(ui_index, fill_value=0)
    )

freq(index, df_trns, df_prod)
#                freq
# uid    iid         
# user_A item_A     2
# user_A item_B     1
# user_B item_A     0
# user_B item_B     2
```

Requirements:

1. Feature extractor must take 3 parameters:
   1. `ui_index` is a list of user-item pairs. Feature must be present for each user-item pair in output dataframe. Simply put, `output.index` must be equal to `ui_index`.
   1. `df_trns` is a dataframe with transactions (order id, user id, product id, order number, etc.).
   1. `df_prod` is a dataframe with products information (aisle, department, product name, etc.).
1. Feature extractor must return a pandas.DataFrame with 1 or more columns:
   1. `columns`: each column is a feature.
   1. `index`: index is a list of (user, item) pairs equals to `ui_index`.

Feature extractor can be added two ways:

1. Automatically using plugin system.
1. Manually using `InstacartDataset.register_feature_extractors({"name": func})` method.

### Plugin system

*Plugin* is a python module located in `instacartlib/feature_extractors` folder. Plugin must have a global variable `exports` with a dictionary of exported functions:

```python
# instacartlib/feature_extractors/000_ui_freq.py

def freq(ui_index, df_trns, df_prod):
    return (df_trns
        .groupby(['uid', 'iid'], sort=False)
        .size()
        .to_frame('freq')
        .reindex(ui_index, fill_value=0)
    )

exports = {'freq': freq}
```

All exported functions will be available using `exports` variable of the `feature_extractors` module. For example, given this files structure:
```
instacartlib/
  feature_extractors/
    __init__.py
    000_ui_freq.py
    001_ui_avg_cart_pos.py
```

Import will be:

```python
>>> from instacartlib.feature_extractors import exports as feature_extractors
>>> feature_extractors
{'000_ui_freq.freq': <function instacartlib.feature_extractors.000_ui_freq.freq(ui_index, df_trns, df_prod)>,
 '001_ui_avg_cart_pos.avg_cart_pos': <function instacartlib.feature_extractors.001_ui_avg_cart_pos.avg_cart_pos(ui_index, df_trns, df_prod)>}
```
Were key is `<feature_extractor_name>.<export_key>`.

Plugins that can't be imported or don't meet the requirements (have an `exports` attribute of a dict type) will be skipped with warning.

