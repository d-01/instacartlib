import pandas as pd


def buy_delays(index, df_trns, **kwargs):
    """
    ui_days_delay_max: the longest (in days) user A gone without buying item B.
    ui_days_delay_mid: median number of days user A gone without buying item B.
    i_days_delay_global_mid: median number of days any user gone without buying
        item B.

    ui_days_passed: days passed since last order (prediction based on medium
        delay between user's orders).
    ui_readyness_max: days passed exceeds max delay for particular item.
        Example:
        readyness=10: means user hasn't bought the item more then 10 days
            above his longest delay.
        readyness=-10: 10 days left until delay exceeds maximum delay, i.e.
            user probably can go longer without buying this item.
    ui_readyness_max_abs: absolute value of readyness. The bigger the value the
        more unusual is delay.
    ui_readyness_mid: days passed exceeds median delay for particular item.
    ui_readyness_mid_abs: absolute value of `ui_readyness_mid`.

    ui_readyness_global_mid: user readyness relative to global delay for
        particular item.
    ui_readyness_global_mid_abs: absolute value of `ui_readyness_global_mid`.
    """
    ui_days_delay_max = (
        df_trns
        .groupby(['uid', 'iid'], sort=False)
        .days_until_same_item.max()
        .astype('float32')
        .reindex(index, fill_value=999.)
    )
    ui_days_delay_mid = (
        df_trns
        .groupby(['uid', 'iid'], sort=False)
        .days_until_same_item.median()
        .astype('float32')
        .reindex(index, fill_value=999.)
    )
    i_days_delay_global_mid = (
        df_trns
        .groupby('iid', sort=False)
        .days_until_same_item.median()
        .astype('float32')
        .reindex(index, level='iid', fill_value=999.)
    )
    ui_days_passed = (
        df_trns
        .drop_duplicates(['uid', 'iid'], keep='last')
        .set_index(['uid', 'iid'])
        .days_until_same_item
        .astype('float32')
        .reindex(index, fill_value=999.)
    )

    ui_readyness_max = (ui_days_passed - ui_days_delay_max)
    ui_readyness_max_abs = ui_readyness_max.abs()
    ui_readyness_mid = (ui_days_passed - ui_days_delay_mid)
    ui_readyness_mid_abs = ui_readyness_mid.abs()
    ui_readyness_global_mid = (ui_days_passed - i_days_delay_global_mid)
    ui_readyness_global_mid_abs = ui_readyness_global_mid.abs()

    return pd.DataFrame({
        'ui_days_delay_max': ui_days_delay_max,
        'ui_days_delay_mid': ui_days_delay_mid,
        'i_days_delay_global_mid': i_days_delay_global_mid,
        'ui_days_passed': ui_days_passed,
        'ui_readyness_max': ui_readyness_max,
        'ui_readyness_max_abs': ui_readyness_max_abs,
        'ui_readyness_mid': ui_readyness_mid,
        'ui_readyness_mid_abs': ui_readyness_mid_abs,
        'ui_readyness_global_mid': ui_readyness_global_mid,
        'ui_readyness_global_mid_abs': ui_readyness_global_mid_abs,
    })

exports = {'buy_delays': buy_delays}