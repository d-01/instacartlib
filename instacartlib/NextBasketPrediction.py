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
        self._update_train_needed = True
        self._update_predict_needed = True


    def _print(self, message, indent=0):
        if self.verbose > 0:
            message = str(message)
            if indent > 0:
                pad = ' ' * indent
                message = pad + message.replace('\n', '\n' + pad)
            print(message)


    def add_data(self, path_dir):
        self.path_dir = path_dir
        self._update_train_needed = True
        self._update_predict_needed = True
        return self


    def train_model(self):
        if self.path_dir is None:
            raise ValueError('Model needs data to be trainded on. '
                'Use `.add_data(path_dir)` to set path to directory with data.')

        if self._update_train_needed:
            _update_datasets(self.icds_train, self.features_train,
                self.path_dir)
            self._update_train_needed = False

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
        if self.path_dir is None:
            raise ValueError('Model needs data to make predictions. '
                'Use `.add_data(path_dir)` to set path to directory with data.')

        self.model = joblib.load(path)
        self._model_trained = True

        if self._update_predict_needed:
            _update_datasets(self.icds_predict, self.features_predict,
                self.path_dir)
            self._update_predict_needed = False

        # Update predictions
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
        self._add_popular_products()
        self.predictions = (
            self.predictions
            .reset_index()
            .sort_values(['uid', 'index'])
            .drop(columns='index')
            .reset_index(drop=True)
        )

        return self


    def get_predictions(self, user_ids, n_limit=10):

        if self._model_trained == False:
            raise ValueError('Model has to be trained to make predictions. '
                'Use `.train_model()` or `.load_model(path)`.')

        if np.isscalar(user_ids):
            user_ids = [user_ids]
        user_ids = list(user_ids)
        predictions = self.predictions[self.predictions.uid.isin(user_ids)]
        if n_limit is not None:
            predictions = predictions.groupby('uid', sort=False).head(n_limit)

        product_names = (
            self.icds_predict.df_prod
            .set_index('iid')
            .product_name
        )
        predictions = predictions.join(product_names, on='iid')
        return predictions


    def _add_popular_products(self):
        """
        Take all users with less then 10 predicted products and add most
        popular products. This function ensures that each user will have no
        less then 10 predicted products.

        For each user with less then 10 predicted products:
        1. Add 3 most popular products from the same aisle as the first
           predicted product.
        2. Add 3 most popular products from the same department as the first
           predicted product.
        3. Add 10 overall most popular products.
        4. Drop duplicated products.
        """
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

        uid_n_predictions = self.predictions.value_counts('uid')
        uid_add_predictions = uid_n_predictions[uid_n_predictions < 10].index

        uid_predicted_products = (
            self.predictions
            .drop_duplicates('uid', keep='first')
            .set_index('uid')
            .loc[uid_add_predictions, ['iid']]  # frame
            # add `aisle_id` and `department_id` columns
            .join(df_prod, on='iid')
        )

        aisle_top3 = (
            uid_predicted_products
            .aisle_id.map(aisle_id_top3_prod)
            .explode()
            .rename('iid')
            .reset_index()
        )
        department_top3 = (
            uid_predicted_products
            .department_id.map(dept_id_top3_prod)
            .explode()
            .rename('iid')
            .reset_index()
        )
        top10 = (
            pd.Series(
                [top10_prod] * len(uid_predicted_products),
                index=uid_predicted_products.index)
            .explode()
            .rename('iid')
            .reset_index()
        )

        self.predictions = (
            pd.concat([self.predictions, aisle_top3, department_top3, top10],
                axis='rows', ignore_index=True)
            .drop_duplicates(['uid', 'iid'])
        )


    def predictions_to_csv(self, path):
        """
        user_id,product_id
        1,196 12427 10258 25133 46149 38928 39657 49235 13032 35951
        2,47209 1559 19156 18523 33754 16589 24852 21709 22124 32792
        ...
        """
        n_limit=10

        rows = ['user_id,product_id']

        for uid, iids in tqdm(self.predictions.groupby('uid').iid,
                disable=(self.verbose < 1)):
            iids = iids.values[:n_limit]
            iid_list_str = ' '.join(map(str, iids))
            rows.append(f'{uid},{iid_list_str}')

        with open(path, 'wt', newline='') as f:
            f.write('\n'.join(rows))







