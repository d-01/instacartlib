import pandas as pd


def buy_delays(index, df_trns, **kwargs):
    days_delay_max = (
        df_trns
        .groupby(['uid', 'iid'], sort=False)
        .days_until_same_item.max()
        .astype('float32')
        .reindex(index, fill_value=999.)
    )
    days_delay_mid = (
        df_trns
        .groupby(['uid', 'iid'], sort=False)
        .days_until_same_item.median()
        .astype('float32')
        .reindex(index, fill_value=999.)
    )
    _days_delay_mid_global = (
        df_trns
        .groupby('iid', sort=False)
        .days_until_same_item.median()
        .astype('float32')
        .rename('days_delay_mid_global')
    )
    days_delay_mid_global = (
        pd.DataFrame(index=index)
        .join(_days_delay_mid_global)
        .fillna(999.)
        .loc[:, 'days_delay_mid_global']
    )
    days_passed = (
        df_trns
        .drop_duplicates(['uid', 'iid'], keep='last')
        .set_index(['uid', 'iid'])
        .days_until_same_item
        .astype('float32')
        .reindex(index, fill_value=999.)
    )

    readyness_max = (days_passed - days_delay_max)
    readyness_max_abs = readyness_max.abs()
    readyness_mid = (days_passed - days_delay_mid)
    readyness_mid_abs = readyness_mid.abs()
    readyness_mid_global = (days_passed - days_delay_mid_global)
    readyness_mid_global_abs = readyness_mid_global.abs()

    return pd.DataFrame({
        'days_delay_max': days_delay_max,
        'days_delay_mid': days_delay_mid,
        'days_delay_mid_global': days_delay_mid_global,
        'days_passed': days_passed,
        'readyness_max': readyness_max,
        'readyness_max_abs': readyness_max_abs,
        'readyness_mid': readyness_mid,
        'readyness_mid_abs': readyness_mid_abs,
        'readyness_mid_global': readyness_mid_global,
        'readyness_mid_global_abs': readyness_mid_global_abs,
    })

exports = {'buy_delays': buy_delays}