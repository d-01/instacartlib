"""
Instacart recommender system - high-level API.

Capabilities:
1. Managing settings (config) for underlying modules.
2. Managing downloaded files and feature cache files.
3. Data reading and preprocessing.
4. Train / test dataset generation process.
5. Model train.
6. Model save load.
7. Predict products for given user ids.
8. Write predictions to csv file.
"""

"""
API draft:
```python

from instacartlib import NextBasketPrediction
from instacartlib import InstacartDataset


InstacartDataset(verbose=1).download('instacart_data')

nbp = NextBasketPrediction(verbose=1)
nbp.add_data('instacart_data')


# Train

nbp.train_model()
nbp.save_model('instacartlib_model.dump')


# Predict

nbp.load_model('instacartlib_model.dump')
predicted_item_ids = nbp.predict(user_ids=[...])
```
"""

from instacartlib import InstacartDataset
from instacartlib import FeaturesDataset
from .utils import drop_duplicates

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

from tqdm import tqdm


def _update_datasets(instacart_dataset, features_dataset, path_dir):
    instacart_dataset.read_dir(path_dir)
    features_dataset.extract_features(**instacart_dataset.dataframes)


class NextBasketPrediction:
    def __init__(self, verbose=0):
        self.verbose = verbose

        self.icds_train = InstacartDataset(train=True, n_orders_limit=5,
            verbose=self.verbose)
        self.icds_predict = InstacartDataset(train=False, n_orders_limit=5,
            verbose=self.verbose)
        self.features_train = FeaturesDataset(features_cache_dir=None,
            verbose=self.verbose)
        self.features_predict = FeaturesDataset(features_cache_dir=None,
            verbose=self.verbose)
        self.predictions = pd.DataFrame()

        self.model = GradientBoostingClassifier(verbose=self.verbose)
        self._model_trained = False

        self.path_dir = None
        self._train_update_needed = True
        self._predict_update_needed = True


    def _print(self, message, indent=0):
        if self.verbose > 0:
            message = str(message)
            if indent > 0:
                pad = ' ' * indent
                message = pad + message.replace('\n', '\n' + pad)
            print(message)


    def add_data(self, path_dir):
        self.path_dir = path_dir
        self._train_update_needed = True
        self._predict_update_needed = True
        return self


    def train_model(self):
        if self.path_dir is None:
            raise ValueError('Model needs data to be trainded on. '
                'Use `.add_data(path_dir)` to set path to directory with data.')

        if self._train_update_needed:
            _update_datasets(self.icds_train, self.features_train,
                self.path_dir)
            self._train_update_needed = False

        x = self.features_train.df_ui.drop(columns='ui_in_target').values
        y = self.features_train.df_ui['ui_in_target'].values

        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=.01,
            stratify=y)

        self.model.fit(x_train, y_train)
        self._model_trained = True

        y_pred = self.model.predict(x_val)
        acc = np.mean(y_pred == y_val)
        self._print(f'Model\'s accuracy: {acc}')
        return self


    def save_model(self, path):
        joblib.dump(self.model, path)
        return self


    def load_model(self, path):
        self.model = joblib.load(path)
        self._model_trained = True
        return self


    def get_predictions(self, user_ids):
        if self.path_dir is None:
            raise ValueError('Model needs data to make predictions. '
                'Use `.add_data(path_dir)` to set path to directory with data.')

        if self._model_trained == False:
            raise ValueError('Model has to be trained to make predictions. '
                'Use `.train_model()` or `.load_model(path)`.')

        if self._predict_update_needed:
            _update_datasets(self.icds_predict, self.features_predict,
                self.path_dir)

            x_test = self.features_predict.df_ui.values
            y_prob = self.model.predict_proba(x_test)[:, 1]
            self.predictions = (
                pd.Series(
                    y_prob,
                    index=self.features_predict.df_ui.index,
                    name='in_target_prob'
                )
                .reset_index()
                .sort_values(['uid', 'in_target_prob'], ascending=[True, False])
            )

            self._predict_update_needed = False

        if np.isscalar(user_ids):
            user_ids = [user_ids]
        user_ids = list(user_ids)
        predictions = self.predictions[self.predictions.uid.isin(user_ids)]

        product_names = (
            self.icds_predict.df_prod
            .set_index('iid')
            .product_name
        )
        predictions = predictions.join(product_names, on='iid')
        return predictions


    def predictions_to_csv(self, path, add_popular=True):
        """
        1. Take top 10 predictions with max probability.

        If add_popular=True:

        2. If there is less then 10 predicted products take first one and
           add 3 most popular products from the same aisle.
        3. If there is still less then 10 predicted products take first one and
           add 3 most popular products from the same department.
        4. If there is still less then 10 predicted products add 10 overall most
           popular products.
        5. Drop duplicates and keep first 10 products from this list.
        """
        n_limit=10

        if add_popular:
            df_prod = (
                self.icds_predict._products.df
                .set_index('product_id')
                .loc[:, ['aisle_id', 'department_id']]
            )
            #             aisle_id  department_id
            # product_id
            # 49688             73             11

            df_prod_n = (
                self.icds_predict._transactions.df
                .value_counts('product_id', sort=False)
                .to_frame('n')
                .join(df_prod)
            )
            #              n  aisle_id  department_id
            # product_id
            # 49688       55        73             11

            aisle_id_top3_prod = (
                df_prod_n
                .groupby('aisle_id', sort=False)
                .n.nlargest(3)
                .reset_index()
                .groupby('aisle_id')
                .product_id.apply(list)
            )
            # aisle_id
            # 134    [37923, 10607, 36885]

            dept_id_top3_prod = (
                df_prod_n
                .groupby('department_id', sort=False)
                .n.nlargest(3)
                .reset_index()
                .groupby('department_id')
                .product_id.apply(list)
            )
            # department_id
            # 21     [41149, 7035, 14010]

            top10_prod = df_prod_n.nlargest(10, 'n').index.to_list()

        rows = ['user_id,product_id']

        for uid, iids in tqdm(self.predictions.groupby('uid').iid,
                disable=(self.verbose < 1)):
            iids = iids.to_list()

            if add_popular and len(iids) < 10:
                aisle_id = df_prod.aisle_id[iids[0]]
                append_iids = aisle_id_top3_prod[aisle_id]
                iids = drop_duplicates(iids + append_iids)

                if len(iids) < 10:
                    dept_id = df_prod.department_id[iids[0]]
                    append_iids = dept_id_top3_prod[dept_id]
                    iids = drop_duplicates(iids + append_iids)

                    if len(iids) < 10:
                        iids = drop_duplicates(iids + top10_prod)

            iids = iids[:n_limit]
            iid_list_str = ' '.join(map(str, iids))
            rows.append(f'{uid},{iid_list_str}')

        with open(path, 'wt', newline='') as f:
            f.write('\n'.join(rows))







