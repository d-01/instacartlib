
from instacartlib.feature_extractors import exports as feature_extractors

import io
import pandas as pd

import pytest


DF_ORD = '''\
order_id     uid  order_n  order_dow  order_hour_of_day  days_since_prior_order
   ord_A  user_A        1          1                 16                       0
   ord_E  user_A        2          4                  8                      30
   ord_C  user_B        1          4                 11                      30
   ord_D  user_B        2          3                 10                      13
'''

DF_TRNS = '''\
order_id     uid     iid  cart_pos  is_reordered  order_r  days_until_same_item
   ord_A  user_A  item_B         1             0        2                  45.0
   ord_A  user_A  item_A         2             0        2                  30.0
   ord_E  user_A  item_A         1             1        1                  15.0
   ord_E  user_A  item_C         2             0        1                  15.0
   ord_C  user_B  item_C         1             0        2                  13.0
   ord_C  user_B  item_D         2             0        2                  36.5
   ord_D  user_B  item_C         1             0        1                  23.5
   ord_D  user_B  item_A         2             1        1                  23.5
'''

# days_since_prior_order
# ---------------------
# order_n         1   2
# uid    iid
# user_A item_A   0  30
#        item_B   0   .
#        item_C   .  30
# user_B item_A   .  13
#        item_C  30  13
#        item_D  30   .


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
        user_B  item_A
        user_B  item_B
        user_B  item_C
        user_B  item_D
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
    '001_ui_buy_counts.buy_counts' not in feature_extractors,
    reason="feature extractor was not registered",
)
def test_ui_buy_counts():
    df_trns = pd.read_csv(io.StringIO('''
           uid     iid  order_r
        user_A  item_A        2
        user_A  item_A        1
        user_A  item_B        1
        user_B  item_A        1
    '''), sep=r'\s+')
    # user_A, order_1: [item_A]
    # user_A, order_2: [item_A, item_B]
    # user_B, order_1: [item_A]

    dataframes = {'df_trns': df_trns}

    ui_index = pd.read_csv(io.StringIO('''
           uid     iid
        user_A  item_A
        user_A  item_B
        user_B  item_A
        user_B  item_B
    '''), sep=r'\s+').set_index(['uid', 'iid']).index

    expected = pd.DataFrame({
        'u_n_orders': [2, 2, 1, 1],
        'ui_n_chances': [2, 1, 1, 0],
        'ui_total_buy': [2, 1, 1, 0],
        'ui_total_buy_ratio': [1.0, 0.5, 1.0, 0.0],
        'ui_chance_buy_ratio': [1.0, 1.0, 1.0, 0.0],
        'u_n_transactions': [3, 3, 1, 1],
        'u_unique_items': [2, 2, 1, 1],
        'u_order_size_mid': [1.5, 1.5, 1.0, 1.0],
        'i_n_popularity': [2, 1, 2, 1],
        'i_n_orders_mid': [1.5, 1.0, 1.5, 1.0]
    }, index=ui_index)
    # Example: `'ui_total_buy': [2, 1, 1, 0]` means
    #   user_A bought item_A 2 times
    #   user_A bought item_B 1 time
    #   user_B bought item_A 1 time
    #   user_B bought item_B 0 times

    extractor_fn = feature_extractors['001_ui_buy_counts.buy_counts']
    test_output = extractor_fn(ui_index, **dataframes)
    pd.testing.assert_frame_equal(test_output, expected, check_dtype=False)


@pytest.mark.skipif(
    '002_ui_avg_cart_pos.avg_cart_pos' not in feature_extractors,
    reason="feature extractor was not registered",
)
def test_avg_cart_pos(ui_index, dataframes):
    avg_cart_pos = feature_extractors['002_ui_avg_cart_pos.avg_cart_pos']

    test_output = avg_cart_pos(ui_index, **dataframes)
    expected = pd.read_fwf(io.StringIO('''
           uid     iid  ui_avg_cart_pos
        user_A  item_A              1.5
        user_A  item_B              1.0
        user_A  item_C              2.0
        user_A  item_D            999.0
        user_B  item_A              2.0
        user_B  item_B            999.0
        user_B  item_C              1.0
        user_B  item_D              2.0
    ''')).set_index(['uid', 'iid']).astype('float32')
    pd.testing.assert_frame_equal(test_output, expected)


@pytest.mark.skipif(
    '000_ui_in_target.in_target' not in feature_extractors,
    reason="feature extractor was not registered",
)
def test_in_target(ui_index, dataframes_target):
    in_target = feature_extractors['000_ui_in_target.in_target']

    test_output = in_target(ui_index, **dataframes_target)
    expected = pd.read_fwf(io.StringIO('''
           uid     iid  ui_in_target
        user_A  item_A             1
        user_A  item_B             0
        user_A  item_C             1
        user_A  item_D             0
        user_B  item_A             1
        user_B  item_B             0
        user_B  item_C             1
        user_B  item_D             0
    ''')).set_index(['uid', 'iid']).astype('uint8')
    pd.testing.assert_frame_equal(test_output, expected)


@pytest.mark.skipif(
    '003_ui_buy_delays.buy_delays' not in feature_extractors,
    reason="feature extractor was not registered",
)
def test_buy_delays():
    df_trns = pd.read_csv(io.StringIO('''
           uid     iid  days_until_same_item
        user_A  item_A                  17.0
        user_A  item_A                  11.0
        user_A  item_B                  11.0
        user_B  item_A                   3.0
    '''), sep=r'\s+')
    # user_A:
    #             order_3  order_2  order_1
    #       item
    #     item_A       17        .       11
    #     item_B        .        .       11
    #
    # user_B:
    #             order_3  order_2  order_1
    #       item
    #     item_A        .        .        3

    dataframes = {'df_trns': df_trns}

    ui_index = pd.read_csv(io.StringIO('''
           uid     iid
        user_A  item_A
        user_A  item_B
        user_B  item_A
        user_B  item_B
    '''), sep=r'\s+').set_index(['uid', 'iid']).index

    expected = pd.DataFrame({
        'ui_days_delay_max': [17.0, 11.0, 3.0, 999.0],
        'ui_days_delay_mid': [14.0, 11.0, 3.0, 999.0],
        'i_days_delay_global_mid': [11.0, 11.0, 11.0, 11.0],
        'ui_days_passed': [11.0, 11.0, 3.0, 999.0],
        'ui_readyness_max': [-6.0, 0.0, 0.0, 0.0],
        'ui_readyness_max_abs': [6.0, 0.0, 0.0, 0.0],
        'ui_readyness_mid': [-3.0, 0.0, 0.0, 0.0],
        'ui_readyness_mid_abs': [3.0, 0.0, 0.0, 0.0],
        'ui_readyness_global_mid': [0.0, 0.0, -8.0, 988.0],
        'ui_readyness_global_mid_abs': [0.0, 0.0, 8.0, 988.0]
    }, index=ui_index)
    # Example: `'ui_days_passed': [11.0, 11.0, 3.0, 999.0],` means
    #   11 days passed since user_A last bought item_A
    #   11 days passed since user_A last bought item_B
    #   3 days passed since user_B last bought item_A
    #   999 days passed since user_B last bought item_B

    extractor_fn = feature_extractors['003_ui_buy_delays.buy_delays']
    test_output = extractor_fn(ui_index, **dataframes)
    pd.testing.assert_frame_equal(test_output, expected, check_dtype=False)

