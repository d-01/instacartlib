# Instacartlib

*Predict the products that will be in the user's next order.*

## Quick start

### Installation

```
pip install https://github.com/d-01/instacartlib/archive/main.zip
```

Optional dependency for `catboost` model:

```
pip install catboost
```

### Download data

```python
from instacartlib import InstacartDataset, NextBasketPrediction

DATA_DIRECTORY = 'instacart_data'

InstacartDataset(verbose=1).download(to_dir=DATA_DIRECTORY)

nbp = NextBasketPrediction(verbose=1)
nbp.add_data(DATA_DIRECTORY)
```

### Use model to get predictions

Load pretrained model:

```python
nbp.load_model('catboost')
# id='gbc' (default): Gradient Boosting Classifier model is used by default.
# id='catboost': CatBoost model gives more accurate predictions but requires
#                `catboost` library to be installed.
```

Get predictions for specified user ids:

```python
nbp.get_predictions(user_ids=[20001, 85768])
```

Export predictions for all users to csv file:

```python
nbp.predictions_to_csv('submission.csv')
# > head -n 4 submission.csv
# user_id,product_id
# 1,196 12427 25133 10258 46149 39657 38928 35951 13032 49235
# 2,47209 19156 1559 18523 33754 16589 21709 24852 39928 22825
# 3,39190 47766 43961 21903 17668 18599 16797 48523 32402 22035
```

### Train model from scratch

Train any scikit-learn-compatible model (~15min in Google Colab):

```python
from catboost import CatBoostClassifier

model = CatBoostClassifier(verbose=100)

nbp = NextBasketPrediction(model=model, verbose=1)
nbp.add_data(DATA_DIRECTORY)
nbp.train_model()
# Model's accuracy: 0.8407207833594067
```

Save trained model to file:

```python
nbp.save_model('instacart_nbp_model.dump')
```

Open demo in [Google Colab](https://colab.research.google.com/drive/1aFc-e_u5W-BrA7cdp6E2qZsgZtiJCGC8?usp=sharing)

## NBP task description

*Next Basket Prediction*

Given 5.3M transactions (transposed) for 100k users:

```
                                0          1          2  
order_id                  2539329    2539329     473747  
user_id                         1          1          1  
order_number                    1          1          2  
order_dow                       2          2          6  
order_hour_of_day               8          8         15  ...
days_since_prior_order        NaN        NaN         15  
product_id                    196      14084        196  
add_to_cart_order               1          2          1  
reordered                       0          0          1  
```

And additional information about 50k products:

```
   iid  department_id  aisle_id department              aisle      product_name
0    1             19        61     snacks      cookies cakes  Chocolate San...
1    2             13       104     pantry  spices seasonings  All-Seasons Salt
2    3              7        94  beverages                tea  Robust Golden...
...
```

Predict products that most likely will be in the user's next order:

```
user_id,product_id
1,196 12427 10258 25133 46149 38928 39657 49235 13032 35951
2,47209 1559 19156 18523 33754 16589 24852 21709 22124 32792
3,39190 47766 43961 21903 17668 18599 16797 48523 32402 22035
...
```

