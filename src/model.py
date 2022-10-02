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
from sklearn import preprocessing as pp
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

# for saving the model
from joblib import dump, load

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.neural_network import MLPRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.multioutput import MultiOutputRegressor

from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import *
import tensorflow as tf

import keras


import os.path as osp


# %%
seed = 745


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


# %%
X_norm = pp.normalize(X)

pp_pipe = Pipeline(
    [
        ("norm", Normalizer()),  
        ("scale", StandardScaler()), 
        ("selector", VarianceThreshold(threshold=(.8 * (1 - .8)))),  # rm features with low variance
    ]
)

X_norm = pp_pipe.fit_transform(X)


# %% [markdown]
# ## Train/test split
#

# %%
# split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y, test_size=0.3, random_state=seed
)


# %% [markdown]
# # Build model
#

# %%
# build the model
def init_model(n_inputs, n_outputs, name="wine_model"):
    model = Sequential(name=name) # create a sequential model

    # input layer
    model.add(Dense(64, input_dim=n_inputs, activation="relu"))

    # hidden layers
    model.add(Dense(128, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(128, activation="relu"))

    # output layer
    model.add(Dense(n_outputs, activation="linear"))

    # compile the model 
    model.compile(loss="mae", optimizer=tf.keras.optimizers.Adam(lr=0.01, amsgrad=True),  metrics=['mean_absolute_error', 'mean_squared_error', 'accuracy'])

    return model



# %%
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "../models", verbose=1, save_best_only=True, mode="min"
)

# keep track of the model training progression
history = tf.keras.callbacks.History()

callbacks = [
    checkpoint,
    tf.keras.callbacks.EarlyStopping(patience=3, monitor="loss", mode="auto"),
    tf.keras.callbacks.TensorBoard(log_dir=osp.join("..", "logs")),
    history,
]


n_inputs, n_outputs = X.shape[1], y.shape[1]
model = init_model(n_inputs, n_outputs)

model.summary()

# %%

model.fit(X_train, y_train, callbacks=callbacks, epochs=100, verbose=2, batch_size=128)


# %%
import joblib

joblib.dump(model, "../models/wine_model.pkl")


