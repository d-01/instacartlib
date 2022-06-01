
def avg_cart_pos(df_trns):
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
    return (df_trns.groupby(['uid', 'iid'], sort=False).in_cart_ord.mean()
        .to_frame('avg_cart_pos'))


exports = [avg_cart_pos]
