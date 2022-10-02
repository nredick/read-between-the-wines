---
jupyter:
  jupytext:
    cell_metadata_filter: -all
    custom_cell_magics: kql
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: 'Python 3.10.7 (''venv_maishacks2022'': venv)'
    language: python
    name: python3
---

# Train a Model to Predict the Price and Quality of Wine

## Imports

```python
import os.path as osp

import joblib
import pandas as pd
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from sklearn import preprocessing as pp
from sklearn.ensemble import *
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import *
```


```python
seed = 745
```


# Load data


```python
wine = pd.read_csv("../data/wine_processed.csv", header=0)

wine.dropna(inplace=True, axis=0, how="any")

# get the features
X = wine.drop(["points", "price", "title", "location", "lat", "lon", "year"], axis=1)
# get the targets (points, price)
y = wine.drop(wine.columns.difference(["points", "price"]), axis=1)
```


```python
X.head()
```


```python
y.head()
```


```python
X_norm = pp.normalize(X)

pp_pipe = Pipeline(
    [
        ("norm", Normalizer()),
        # ("features", PolynomialFeatures()),
        ("scale", RobustScaler()), 
        # ("selector", VarianceThreshold(threshold=(.5 * (1 - .8)))),  # rm features with low variance
    ]
)

X_norm = pp_pipe.fit_transform(X)
```


## Train/test split


```python
# split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y, test_size=0.2, random_state=seed
)
```


# Build model


```python
BATCH_SIZE = 128
```

```python
from tensorflow.keras import layers

# build the model
def init_model(n_inputs, n_outputs, name="wine_model"):
    model = Sequential(name=name)  # create a sequential model

    # input layer
    model.add(Dense(64, input_dim=n_inputs, activation="relu"))

    # hidden layers
    model.add(Dense(128, activation="relu"))
    model.add(layers.Dropout(.5, input_shape=(2,)))
    model.add(Dense(256, activation="relu"))
    model.add(layers.Dropout(.5, input_shape=(2,)))
    model.add(Dense(128, activation="relu"))

    # output layer
    model.add(Dense(n_outputs, activation="linear"))

    # compile the model
    model.compile(
        loss=tf.keras.losses.MeanSquaredLogarithmicError(),
        optimizer=tf.keras.optimizers.Nadam(),
        metrics=['accuracy'],
    )

    return model


```

```python
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "../models", verbose=1, save_best_only=True, mode="min"
)

# keep track of the model training progression
history = tf.keras.callbacks.History()

callbacks = [
    checkpoint,
    tf.keras.callbacks.EarlyStopping(patience=3, monitor="val_loss", mode="auto"),
    tf.keras.callbacks.TensorBoard(log_dir=osp.join("..", "logs")),
    history,
]


n_inputs, n_outputs = X_train.shape[1], y_train.shape[1]
model = init_model(n_inputs, n_outputs)

model.summary()
```

```python
model.fit(
    X_train,
    y_train,
    callbacks=callbacks,
    epochs=100,
    verbose=1,
    batch_size=BATCH_SIZE,
    steps_per_epoch=250,
    validation_split=0.2,
)
```


```python
metrics = pd.DataFrame(model.evaluate(X_test, y_test, batch_size=BATCH_SIZE), index=model.metrics_names).T

metrics
```

```python
joblib.dump(model, "../models/wine_model.pkl")
```


