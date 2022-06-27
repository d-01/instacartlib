
def avg_cart_pos(index, df_trns, **kwargs):
    """ Position of item B in user's A cart on average.
             avg_cart_pos
    uid iid
    0   0        2.333333
        2        5.333333
        5        4.666667
    1   18       1.000000
        24       9.000000
        35      11.000000
    """
    return (df_trns
        .groupby(['uid', 'iid'], sort=False)
        .cart_pos.mean()
        .astype('float32')
        .to_frame('ui_avg_cart_pos')
        .reindex(index, fill_value=999)
    )


exports = {'avg_cart_pos': avg_cart_pos}
