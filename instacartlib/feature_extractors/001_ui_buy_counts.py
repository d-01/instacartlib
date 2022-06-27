
import pandas as pd

def buy_counts(index, df_trns, **kwargs):
    """
    u_n_orders: total number of orders made by user.
    ui_n_chances: number of orders in which user A had a chance to buy item B.
        That is a number of orders following the order in which user A bought
        item B, including the order itself.
    ui_total_buy: times user A bought item B.
    ui_total_buy_ratio: = ui_total_buy / u_n_orders
    ui_chance_buy_ratio: = ui_total_buy / u_n_chances

    u_n_transactions: number of user's A transactions.
    u_unique_items: number of unique items in user's A history.
    u_order_size_mid: user's A average order size.

    i_n_popularity: number of users who purchaised this item at least once.
    i_n_orders_mid: number of purchaises on average across all
        users who purchaised this item at least once.

    index: pd.MultiIndex
        uid: level=0
        iid: level=1
    """
    # `order_r` for oldest transaction = number of orders
    u_n_orders = (
        df_trns
        .drop_duplicates('uid', keep='first')
        .set_index('uid')
        .order_r
        .reindex(index, level='uid', fill_value=0)
    )
    ui_n_chances = (
        df_trns
        .drop_duplicates(['uid', 'iid'], keep='first')
        .set_index(['uid', 'iid'])
        .order_r
        .reindex(index, fill_value=0)
    )
    ui_total_buy = (
        df_trns
        .value_counts(['uid', 'iid'], sort=False)
        .astype('uint8')
        .reindex(index, fill_value=0)
    )
    ui_total_buy_ratio = (
        (ui_total_buy / u_n_orders)
        .astype('float32')
        .fillna(0)
    )
    ui_chance_buy_ratio = (
        (ui_total_buy / ui_n_chances)
        .astype('float32')
        .fillna(0)
    )
    u_n_transactions = (
        df_trns
        .value_counts('uid', sort=False)
        .astype('uint32')
        .reindex(index, level='uid', fill_value=0)
    )
    u_unique_items = (
        df_trns
        .groupby('uid')
        .iid.nunique()
        .astype('uint32')
        .reindex(index, level='uid', fill_value=0)
    )
    u_order_size_mid = (
        df_trns
        .value_counts(['uid', 'order_r'])
        .groupby('uid')
        .median()
        .astype('float32')
        .reindex(index, level='uid', fill_value=0)
    )
    i_n_popularity = (
        df_trns
        .drop_duplicates(['uid', 'iid'])
        .value_counts('iid')
        .astype('uint32')
        .reindex(index, level='iid', fill_value=0)
    )
    i_n_orders_mid = (
        df_trns
        .value_counts(['uid', 'iid'], sort=False)
        .groupby('iid')
        .median()
        .astype('float32')
        .reindex(index, level='iid', fill_value=0)
    )

    return pd.DataFrame({
        'u_n_orders': u_n_orders,
        'ui_n_chances': ui_n_chances,
        'ui_total_buy': ui_total_buy,
        'ui_total_buy_ratio': ui_total_buy_ratio,
        'ui_chance_buy_ratio': ui_chance_buy_ratio,
        'u_n_transactions': u_n_transactions,
        'u_unique_items': u_unique_items,
        'u_order_size_mid': u_order_size_mid,
        'i_n_popularity': i_n_popularity,
        'i_n_orders_mid': i_n_orders_mid,
    }, index=index)


exports = {'buy_counts': buy_counts}