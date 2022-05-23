"""
Extract features from transactions and products dataframes, and cache them to
files.

Types of features:
1. UI - user-item features (interactions). Indexed by (uid, iid) pairs.
   * Example: `User A` has purchased `Item B` N times.
2. U - user features, not related to particular item. Indexed by (uid).
   * Example: `User A` has N orders total.
3. I - item features, not related to particular user. Indexed by (iid).
   * Example: `Item B` purchased by % users at least once.
"""

def get_ui_freq(df_trns):
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
    return df_trns.groupby(['uid', 'iid']).size().rename('freq')

def get_ui_avg_cart_pos(df_trns):
    """ Position of item B in user's A cart on average.
             in_cart_ord
    uid iid             
    0   0       2.333333
        2       5.333333
        5       4.666667
    1   18      1.000000
        24      9.000000
        35     11.000000
    """
    return (df_trns.groupby(['uid', 'iid']).in_cart_ord.mean()
        .rename('avg_cart_pos'))

