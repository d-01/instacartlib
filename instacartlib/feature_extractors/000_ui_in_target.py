def in_target(index, df_trns_target, **kwargs):
    """ 1 if item is in the user's target order, 0 otherwise.
             freq
    uid iid
    0   0       1
        7       1
        14      0
    1   18      0
        24      0
        35      1
    """
    return (df_trns_target
        .groupby(['uid', 'iid'], sort=False)
        .size()
        .astype('uint8')
        .to_frame('ui_in_target')
        .reindex(index, fill_value=0)
    )


exports = {'in_target': in_target}