
import pandas as pd


def buy_counts(index, df_trns, **kwargs):
    """
    n_orders: total number of orders made by user.
    n_chances: number of orders in which user A had a chance to buy item B.
        That is a number of orders following the order in which user A bought
        item B, including the order itself.
    total_buy: times user A bought item B.
    total_buy_ratio: = total_buy / n_orders
    chance_buy_ratio: = total_buy / n_chances
    """
    # oldest transaction reverse num = number of orders
    n_orders = (
        df_trns
        .drop_duplicates('uid', keep='first')
        .set_index('uid')
        .order_r
        .reindex(index, level=0, fill_value=0)
    )
    n_chances = (
        df_trns
        .drop_duplicates(['uid', 'iid'], keep='first')
        .set_index(['uid', 'iid'])
        .order_r
        .reindex(index, fill_value=0)
    )
    total_buy = (
        df_trns
        .value_counts(['uid', 'iid'], sort=False)
        .astype('uint8')
        .reindex(index, fill_value=0)
    )
    total_buy_ratio = (total_buy / n_orders).astype('float32').fillna(0)
    chance_buy_ratio = (total_buy / n_chances).astype('float32').fillna(0)

    return pd.DataFrame({
        'n_orders': n_orders,
        'n_chances': n_chances,
        'total_buy': total_buy,
        'total_buy_ratio': total_buy_ratio,
        'chance_buy_ratio': chance_buy_ratio,
    }, index=index)


exports = {'buy_counts': buy_counts}