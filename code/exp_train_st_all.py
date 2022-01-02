
#!/usr/bin/python

# %%
# Import relevant libraries
from helper_functions import *
import warnings
from icecream import ic
ic("Importing packages...")
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    import os
    from scipy.stats import mstats
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import sys
    import getopt

    import pandas as pd
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow import keras
    from zipfile import ZipFile
    import os
    import random
    import bcolors

    import mne
    import mne_nirs

    from mne_nirs.experimental_design import make_first_level_design_matrix
    from mne_nirs.statistics import run_glm
    from mne_nirs.channels import (get_long_channels,
                                   get_short_channels,
                                   picks_pair_to_idx)
    from mne.preprocessing.nirs import optical_density, beer_lambert_law

    from nilearn.plotting import plot_design_matrix

    from itertools import compress
    from icecream import ic

    # Performance logging
    import wandb
    from wandb.keras import WandbCallback
    import time

wb = train = True
v = p = p_loss = False

config = {
    "learning_rate": 0.00002,
    "epochs": 200,
    "batch_size": 24,
    "loss_function": "mae",
    "optimizer": "nadam",
    "dropout": 0.5,
    "units": 100,
    "past": 39,
    "future": 16,
    "preprocess": "medium",
    "bidirectional": True,
    "activation_function": "tanh",
    "normalize": True,
    "l_pass": 0.7,
    "h_pass": 0.01,
    "train_split": 0.7,
    "raw_path": "data/snirf/pretrain_3.snirf",
    "architecture": "LSTM-3",  # Dense, LSTM, LSTM-3
    "pretrain_dense_units": 128,
    "test_channel": 0,
    "patience": 25
}

raw_path = config.get("raw_path")

paths = ["data/snirf/pretrain_1.snirf",
         "data/snirf/pretrain_2.snirf",
         "data/snirf/pretrain_3.snirf",
         "data/snirf/pretrain_4.snirf",
         "data/snirf/pretrain_5.snirf", ]

if config.get("preprocess") == "none":
    filter_haemo = [preprocess(p,
                               config.get("l_pass"),
                               config.get("h_pass"),
                               bandpass=False,
                               short_ch_reg=False,
                               tddr=False,
                               negative_correlation=False,
                               verbose=v) for p in paths]
if config.get("preprocess") == "simple":
    if v:
        ic("Simple preprocessing:", paths)
    filter_haemo = [preprocess(p,
                               config.get("l_pass"),
                               config.get("h_pass"),
                               bandpass=True,
                               short_ch_reg=False, tddr=False, negative_correlation=False, verbose=v) for p in paths]

elif config.get("preprocess") == "medium":
    if v:
        ic("Medium preprocessing:", paths)
    filter_haemo = [preprocess(p,
                               config.get("l_pass"),
                               config.get("h_pass"),
                               bandpass=True,
                               short_ch_reg=False, tddr=True, negative_correlation=False, verbose=v) for p in paths]

elif config.get("preprocess") == "advanced":
    if v:
        ic("Advanced preprocessing:", paths)
    filter_haemo = [preprocess(p,
                               config.get("l_pass"),
                               config.get("h_pass"),
                               bandpass=True,
                               short_ch_reg=True, tddr=True, negative_correlation=True, verbose=v) for p in paths]

# %%
# Make the hemoglobin values into a dataframe
full_df = [haemo.to_data_frame() for haemo in filter_haemo]

df_nrows = [df.shape[0] for df in full_df]
df_ncols = [df.shape[1] for df in full_df]

nrows_sum = np.sum(df_nrows)
nrows_test = int(nrows_sum * (1 - config.get("train_split")))

idx = np.abs(np.subtract(df_nrows, nrows_test)).argmin()

ic("Calculated test data rows")
ic(nrows_test)
ic(idx)
ic("Rows for the test data:")
ic(df_nrows[idx])
ic("All rows:")
ic(df_nrows)

if v:
    ic("Normalise the data with a basis in the train split so there's no data leakage")

val_data = normalize_and_remove_time(
    full_df[idx])

train_data = [normalize_and_remove_time(df) for df in full_df]
train_data = pd.concat(train_data)

if p:
    if v:
        ic("Plot this stuff.")

    test_features.plot(use_index=True, alpha=0.2, title="Test data").get_figure().savefig(
        "output/test_data.png")
    train_data.plot(use_index=True, alpha=0.2, title="Train data").get_figure(
    ).savefig("output/train_data.png")

# Show heatmap of all column correlations (channel correlations)
if p:
    show_heatmap(train_data)

# Setting the step size (downsampling basically)
step = 1

# How much data can it use
past = config.get("past")

# How long into the future should it predict
future = config.get("future")

# What is the batch size
batch_size = config.get("batch_size")

# How many epochs
epochs = config.get("epochs")

# Settings patience
patience = config.get("patience")

# What is the time column?
date_time_key = "time"

if v:
    ic("Select the start and end indices for the training data")
start = past + future
end = start + len(train_data)

if v:
    ic("Select all train data from start to end as one big array of arrays")
# y = one value per array in x


x_train = train_data.iloc[:-future].values
y_train = train_data.iloc[start:].values[:, config.get("test_channel")]

if v:
    ic("Define sequence length from past values divided by the step (in this case 1)")
sequence_length = int(past / step)

print(x_train.shape)
print(y_train.shape)

if v:
    ic("Make a training dataset from arrays with definitions of sequence length")
if config.get("architecture") == "Dense":
    dataset_train = keras.preprocessing.timeseries_dataset_from_array(
        x_train[:, config.get("test_channel")].flatten(),
        y_train,
        sequence_length=sequence_length,
        sampling_rate=step,
        batch_size=batch_size,
    )
elif config.get("architecture") in ["LSTM", "LSTM-3"]:
    dataset_train = keras.preprocessing.timeseries_dataset_from_array(
        x_train,
        y_train,
        sequence_length=sequence_length,
        sampling_rate=step,
        batch_size=batch_size,
    )

if v:
    ic("Make a validation dataset from like the train dataset")
x_val = val_data.iloc[:-future].values
y_val = val_data.iloc[start:].values[:, config.get("test_channel")]
# y_val = np.append(y_val, [1000000000 for i in range(past)])

if v:
    ic("Calculate the chance levels for the validation data")
y_list = y_val.ravel()
x_list = x_val[past:, config.get("test_channel")]

ic(y_list.shape)
ic(x_list.shape)

chance_df = pd.DataFrame(
    data={
        "y": y_list,
        "x": x_list
    })

chance_df["guess"] = [np.mean(y_list) for i in range(len(chance_df))]
chance_df["gauss"] = [np.random.normal(0) for i in range(len(chance_df))]
chance_df["zero"] = [0 for i in range(len(chance_df))]

chance_df["diff_mean"] = [
    np.abs(
        chance_df["guess"].iloc[i] -
        chance_df["y"].iloc[i])
    for i in range(len(chance_df))]

chance_df["diff_gauss"] = [
    np.abs(
        chance_df["gauss"].iloc[i] -
        chance_df["y"].iloc[i])
    for i in range(len(chance_df))]

chance_df["diff_last"] = [
    np.abs(
        chance_df["x"].iloc[i] -
        chance_df["y"].iloc[i])
    for i in range(len(chance_df))]

chance_df["diff_zero"] = [
    np.abs(
        chance_df["zero"].iloc[i] -
        chance_df["y"].iloc[i])
    for i in range(len(chance_df))]

chance_gauss = chance_df["diff_gauss"].mean()
chance_mean = chance_df["diff_mean"].mean()
chance_last = chance_df["diff_last"].mean()
chance_zero = chance_df["diff_zero"].mean()

print(f"{bcolors.HEADER}Mean value performance:",
      chance_mean, f"{bcolors.ENDC}")
print(f"{bcolors.HEADER}Last value chance performance:",
      chance_last, f"{bcolors.ENDC}")
print(f"{bcolors.HEADER}Gaussian chance performance:",
      chance_gauss, f"{bcolors.ENDC}")
print(f"{bcolors.HEADER}Zero chance performance:",
      chance_zero, f"{bcolors.ENDC}")

if v:
    ic("Make a validation dataset with the same definitions as the training dataset")
if config.get("architecture") == "Dense":
    dataset_val = keras.preprocessing.timeseries_dataset_from_array(
        x_val[:, config.get("test_channel")].flatten(),
        y_val,
        sequence_length=sequence_length,
        sampling_rate=step,
        batch_size=batch_size,
    )
elif config.get("architecture") in ["LSTM", "LSTM-3"]:
    dataset_val = keras.preprocessing.timeseries_dataset_from_array(
        x_val,
        y_val,
        sequence_length=sequence_length,
        sampling_rate=step,
        batch_size=batch_size,
    )
if v:
    print(
        f"Take batches out of the training dataset (currently {batch_size} samples)")
for batch in dataset_train.take(1):
    inputs, targets = batch

print("Input shape:", inputs.numpy().shape)
print("Target shape:", targets.numpy().shape)

path_checkpoint = "model_weights.h5"

if wb:
    wandb.init(project="thought_classification", entity="esbenkran",
               name=f"{config.get('architecture')}_{int(random.random() * 1000)}", config=config, tags=["ALL", "used_model"])
    config = wandb.config

if train:
    if v:
        ic("Define the model architecture")
    if config.get("architecture") == "Dense":
        inputs = keras.layers.Input(shape=(inputs.shape[1]))
    elif config.get("architecture") == "LSTM-3":
        inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))

        lstm_1 = keras.layers.Bidirectional(
            keras.layers.LSTM(
                config.get("units"),
                activation=config.get("activation_function"),
                dropout=config.get("dropout"),
                return_sequences=True,
            ))(inputs)
        lstm_2 = keras.layers.Bidirectional(
            keras.layers.LSTM(
                config.get("units"),
                activation=config.get("activation_function"),
                dropout=config.get("dropout"),
                return_sequences=True,
            ))(lstm_1)
        lstm_out = keras.layers.Bidirectional(
            keras.layers.LSTM(
                config.get("units"),
                activation=config.get("activation_function"),
                dropout=config.get("dropout"),
            ))(lstm_2)
    elif config.get("architecture") in ["LSTM"]:
        inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
        lstm_out = keras.layers.LSTM(
            config.get("units"),
            activation=config.get("activation_function"),
            dropout=config.get("dropout"))(inputs)

    dense_out = keras.layers.Dense(config.get("pretrain_dense_units"), activation="relu")(
        lstm_out if config.get("architecture") in ["LSTM", "LSTM-3"] else inputs)
    outputs = keras.layers.Dense(1)(dense_out)

    # if v:
    #     ic("Generate a learning rate schedule with exponential decay")
    # lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=config.get("learning_rate"),
    #     decay_steps=10000,
    #     decay_rate=0.96,
    #     staircase=True)

    if v:
        ic("Define the optimizer")
    optimizer = keras.optimizers.Adam()

    if v:
        ic("Compile the model")
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizer,
        loss=config.get("loss_function"),
        metrics=['mean_absolute_error', 'mean_squared_error'])

    if v:
        ic("Plot model summary")
    model.summary()

    if v:
        ic("Save checkpoints (W&B also does this)")
    path_checkpoint = "model_checkpoint.h5"

    if v:
        print(f"Set early stopping with {patience} patience")
    es_callback = keras.callbacks.EarlyStopping(
        monitor="val_mean_absolute_error", min_delta=0, patience=patience, verbose=1, mode="min")

    if v:
        ic("Set a callback to save checkpoints")
    modelckpt_callback = keras.callbacks.ModelCheckpoint(
        monitor="mean_absolute_error",
        filepath=path_checkpoint,
        verbose=1,
        save_weights_only=True,
        save_best_only=True,
    )

    if v:
        ic("Fit the model and save results to history. W&B has a callback to save everything")
    if wb:
        history = model.fit(
            dataset_train,
            epochs=epochs,
            validation_data=dataset_val,
            callbacks=[es_callback, modelckpt_callback,
                       WandbCallback(data_type="time series")],
        )
    else:
        history = model.fit(
            dataset_train,
            epochs=epochs,
            validation_data=dataset_val,
            callbacks=[es_callback, modelckpt_callback],
        )
else:
    if v:
        ic("Load the model")
    model = keras.models.load_model(path_checkpoint)

if v:
    ic("Calculate the model accuracy")

predictions = []
diff = []

for batch in dataset_val.take(1):
    inputs, targets = batch

if config.get("architecture") == "Dense":
    predictions = model.predict(inputs).flatten()
else:
    predictions = [model.predict(i[None, ...]) for i in inputs]

diff = [np.abs(predictions[i][0] - y_val[i])
        for i in range(len(predictions))]

ic("Model predictions: ", np.mean(diff))

if train:
    # Visualize the loss
    if p_loss:
        visualize_loss(history, "Training and Validation Loss")

    if p_loss:
        ic("Visualize 5 predictions")
        for x, y in dataset_val.take(5):
            show_plot(
                [x[0][:, 1].numpy(), y[0].numpy(), model.predict(x)[0]],
                future,
                "Single Step Prediction",
            )
