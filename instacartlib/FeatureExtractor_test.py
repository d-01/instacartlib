
from .FeatureExtractor import get_ui_freq
from .FeatureExtractor import get_ui_avg_cart_pos

import pandas as pd


def test_get_ui_freq(df_trns):
    out = get_ui_freq(df_trns)
    assert type(out) == pd.Series
    assert out.name == 'freq'
    assert out.loc[0, [0, 1, 2]].to_list() == [10, 1, 10]


def test_get_ui_avg_cart_pos(df_trns):
    out = get_ui_avg_cart_pos(df_trns)
    assert type(out) == pd.Series
    assert out.name == 'avg_cart_pos'
    assert out.loc[0, [0, 1, 2]].to_list() == [1.4, 2.0, 3.3]
