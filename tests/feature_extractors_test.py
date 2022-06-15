
from instacartlib.feature_extractors import exports as feature_extractors

import io
import pandas as pd

import pytest


DF_ORD = '''\
order_id     uid  order_n  order_dow  order_hour_of_day  days_since_prior_order
   ord_A  user_A        1          1                 16                       0
   ord_E  user_A        0          4                  8                      30
   ord_C  user_B        1          4                 11                      30
   ord_D  user_B        0          3                 10                      13
'''

DF_TRNS = '''\
order_id     uid     iid  cart_pos  is_reordered  days_until_same_item
   ord_A  user_A  item_B         1             0                  60.0
   ord_A  user_A  item_A         2             0                  30.0
   ord_E  user_A  item_A         1             1                  30.0
   ord_E  user_A  item_C         2             0                  30.0
   ord_C  user_B  item_C         1             0                  13.0
   ord_C  user_B  item_D         2             0                  36.5
   ord_D  user_B  item_C         1             0                  23.5
   ord_D  user_B  item_A         2             1                  23.5
'''

DF_PROD = '''\
   iid  department_id  aisle_id  department    aisle  product_name
item_A              1         1      dept A  aisle A  product name
item_B              1         1      dept A  aisle A  product name
item_C              1         2      dept A  aisle B  product name
item_D              2         3      dept B  aisle C  product name
item_E              2         4      dept B  aisle D  product name
'''


@pytest.fixture
def df_ord():
    return pd.read_csv(io.StringIO(DF_ORD), sep=r'\s+')


@pytest.fixture
def df_trns():
    return pd.read_csv(io.StringIO(DF_TRNS), sep=r'\s+')


@pytest.fixture
def df_trns_target(df_trns):
    return df_trns[df_trns.order_id.isin(['ord_E', 'ord_D'])]


@pytest.fixture
def df_prod():
    return pd.read_csv(io.StringIO(DF_PROD), sep=r'\s+')


@pytest.fixture
def ui_index():
    return pd.read_fwf(io.StringIO('''
           uid     iid
        user_A  item_A
        user_A  item_B
        user_A  item_C
        user_A  item_D
        user_A  item_E
        user_B  item_A
        user_B  item_B
        user_B  item_C
        user_B  item_D
        user_B  item_E
    ''')).set_index(['uid', 'iid']).index


@pytest.fixture
def dataframes(df_ord, df_trns, df_prod):
    return dict(
        df_ord=df_ord,
        df_trns=df_trns,
        df_prod=df_prod,
    )


@pytest.fixture
def dataframes_target(df_ord, df_trns, df_prod, df_trns_target):
    return dict(
        df_ord=df_ord,
        df_trns=df_trns,
        df_trns_target=df_trns_target,
        df_prod=df_prod,
    )


@pytest.mark.parametrize("extractor_name", feature_extractors.keys())
def test_feature_extractors_output_valid(extractor_name, ui_index,
        dataframes_target):
    function = feature_extractors[extractor_name]
    test_output = function(ui_index, **dataframes_target)
    pd.testing.assert_index_equal(test_output.index, ui_index)
    assert test_output.isna().values.sum() == 0

    with pytest.raises(TypeError,
            match=r'missing \d+ required positional argument'):
        test_output = function()

    extra_dataframes = {'unused_1': 1, 'unused_2': 2, **dataframes_target}
    test_output = function(ui_index, **extra_dataframes)


@pytest.mark.skipif(
    '001_ui_freq.freq' not in feature_extractors,
    reason="feature extractor was not registered",
)
def test_ui_freq(ui_index, dataframes):
    freq = feature_extractors['001_ui_freq.freq']

    test_output = freq(ui_index, **dataframes)
    expected = pd.read_fwf(io.StringIO('''
           uid     iid  freq
        user_A  item_A     2
        user_A  item_B     1
        user_A  item_C     1
        user_A  item_D     0
        user_A  item_E     0
        user_B  item_A     1
        user_B  item_B     0
        user_B  item_C     2
        user_B  item_D     1
        user_B  item_E     0
    ''')).set_index(['uid', 'iid'])
    pd.testing.assert_frame_equal(test_output, expected)


@pytest.mark.skipif(
    '002_ui_avg_cart_pos.avg_cart_pos' not in feature_extractors,
    reason="feature extractor was not registered",
)
def test_avg_cart_pos(ui_index, dataframes):
    avg_cart_pos = feature_extractors['002_ui_avg_cart_pos.avg_cart_pos']

    test_output = avg_cart_pos(ui_index, **dataframes)
    expected = pd.read_fwf(io.StringIO('''
           uid     iid  avg_cart_pos
        user_A  item_A           1.5
        user_A  item_B           1.0
        user_A  item_C           2.0
        user_A  item_D         999.0
        user_A  item_E         999.0
        user_B  item_A           2.0
        user_B  item_B         999.0
        user_B  item_C           1.0
        user_B  item_D           2.0
        user_B  item_E         999.0
    ''')).set_index(['uid', 'iid'])
    pd.testing.assert_frame_equal(test_output, expected)


@pytest.mark.skipif(
    '000_ui_in_target.in_target' not in feature_extractors,
    reason="feature extractor was not registered",
)
def test_in_target(ui_index, dataframes_target):
    in_target = feature_extractors['000_ui_in_target.in_target']

    test_output = in_target(ui_index, **dataframes_target)
    expected = pd.read_fwf(io.StringIO('''
           uid     iid  in_target
        user_A  item_A          1
        user_A  item_B          0
        user_A  item_C          1
        user_A  item_D          0
        user_A  item_E          0
        user_B  item_A          1
        user_B  item_B          0
        user_B  item_C          1
        user_B  item_D          0
        user_B  item_E          0
    ''')).set_index(['uid', 'iid'])
    pd.testing.assert_frame_equal(test_output, expected)
