# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: 'Python 3.10.7 (''venv_maishacks2022'': venv)'
#     language: python
#     name: python3
# ---

# %%
import pandas as pd

# ML imports
from sklearn.pipeline import Pipeline
from sklearn.ensemble import *
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

# for saving the model
from joblib import dump, load


# %%
seed = 6


# %% [markdown]
# # Load data
#

# %%
wine = pd.read_csv("../data/wine_processed.csv", header=0)

wine.dropna(inplace=True, axis=0, how="any")

# get the features
X = wine.drop(["points", "price", "title", "location"], axis=1)
# get the targets (points, price)
y = wine.drop(wine.columns.difference(["points", "price"]), axis=1)


# %%
X.head()

# %%

y.head()

# %% [markdown]
# ## Train/test split
#

# %%
# split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed
)


# %% [markdown]
# # Build model
#

# %%
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.multioutput import MultiOutputRegressor

from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import *
import tensorflow as tf


# %%
# build the model
def init_model(n_inputs, n_outputs):
    model = Sequential()
    model.add(
        Dense(
            20, input_dim=n_inputs, kernel_initializer="normal", activation="relu"
        )
    )
    model.add(Dense(n_outputs, kernel_initializer="normal"))
    model.compile(loss="mae", optimizer="adam")
    return model


# %%
callbacks = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=3,
    verbose=2,
    mode='auto',
)

n_inputs, n_outputs = X.shape[1], y.shape[1]
model = init_model(n_inputs, n_outputs)
# pipe = Pipeline(
#     [
#         ("norm", Normalizer()),  # normalize data
#         ("scale", RobustScaler()),  # normalize data
#         # ("selector", VarianceThreshold()),  # rm features with low variance
#         ("nn", init_model(n_inputs, n_outputs)),  # model
#     ]
# )

model.fit(X_train, y_train, callbacks=callbacks, epochs=10, verbose=2)


# %%
# print("Training set score: " + str(pipe.score(X_train, y_train)))
# print("Test set score: " + str(pipe.score(X_test, y_test)))

# %%
# save the model
dump(model, '../models/wine-model.joblib') 
