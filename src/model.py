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
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection.VarianceThreshold import VarianceThreshold


# %%
seed = 6


# %% [markdown]
# # Load data
#

# %%
wine = pd.read_csv("../data/wine.csv", header=0)
# get the features
X = wine.drop(["points", "price", "title"], axis=1)
# get the targets (points, price)
y = wine.drop(wine.columns.difference(["points", "price"]), axis=1)


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
# build a pipeline for the data processing & model architecture
pipe = Pipeline(
    [
        ("scaler", StandardScaler()),  # normalize data
        ("selector", VarianceThreshold()),  # rm features with low variance
        ("randomforestregressor", RandomForestRegressor(random_state=seed, max_depth=5, n_estimators=50)),
    ]
)  # model

pipe.fit(X_train, y_train)  # fit the model to the data


# %%
print("Training set score: " + str(pipe.score(X_train, y_train)))
print("Test set score: " + str(pipe.score(X_test, y_test)))

