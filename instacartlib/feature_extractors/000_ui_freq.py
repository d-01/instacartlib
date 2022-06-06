
def freq(index, df_trns, **kwargs):
    """ Number of times user A purchaised item B.
             freq
    uid iid
    0   0       3
        7       1
        14      2
    1   18      1
        24      2
        35      1
    """
    return (df_trns
        .groupby(['uid', 'iid'], sort=False)
        .size()
        .to_frame('freq')
        .reindex(index, fill_value=0)
    )


exports = {'freq': freq}