from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from helper_functions import *
from helper_functions import preprocess, visualize_loss, show_plot
import warnings
from icecream import ic
ic("Importing packages...")
with warnings.catch_warnings():
    import mne
    import mne_nirs
    import numpy as np
    import pandas as pd
    import os
    import sys
    import matplotlib.pyplot as plt
    import getopt
    from tensorflow import keras
    from tensorflow.keras import metrics
    import tensorflow as tf
    import time

    import wandb
    from wandb.keras import WandbCallback

    import re
    import bcolors


def main():
    v = True

    config = {
        'dropout': 0.5,
        'dropout_2': 0.5,
        'train_split': 0.6,
        "learning_rate": 0.00005,
        'preprocess': "medium",
        'batch_size': 16,
        'epochs': 500,
        'trainable': False,
        'pretrained': False,
        'dense_units': 256,
        'layers_transferred': 5,  # 1-5
        'bci_task': "data/snirf/bci_task_2_arithmetic_audiobook.snirf"
    }

    wandb.init(
        project="thought_classification", entity="esbenkran",
        tags=["test"], config=config)

    config = wandb.config

    raw_path = config.get("bci_task")

    past = 39
    split_fraction = config.get("train_split")
    date_time_key = "time"

    if config.get("preprocess") == "medium":
        filter_haemo = \
            preprocess(raw_path,
                       0.7,
                       0.01,
                       bandpass=True,
                       short_ch_reg=False,
                       tddr=True,
                       negative_correlation=False,
                       verbose=v)
    elif config.get("preprocess") == "simple":
        filter_haemo = \
            preprocess(raw_path,
                       0.7,
                       0.01,
                       bandpass=True,
                       short_ch_reg=False,
                       tddr=False,
                       negative_correlation=False,
                       verbose=v)

    ic("Extract epochs from raw data...")

    task_1 = re.findall(r'(?<=\d_).*(?=_)', raw_path)[0]
    task_2 = re.findall(r'(?<=_)[a-z]{3,10}(?=.snirf)', raw_path)[0]

    filter_haemo.annotations.rename(
        {
            '0': 'Nothing',
            '1': task_1,
            '2': task_2
        })

    events, event_dict = mne.events_from_annotations(
        filter_haemo, verbose=False)

    epochs = mne.Epochs(filter_haemo, events=events, event_id=event_dict,
                        tmin=0.0, tmax=10.0, baseline=(0, 0.5),
                        preload=True,
                        verbose=False)

    df = epochs[[task_2, task_1]].to_data_frame()

    # kfold = KFold(n_splits=10, shuffle=True)

    # samples, 39, 200
    print(f"{bcolors.ITALIC}Augmenting data from shape {df.shape}")
    df = augment_data(df, past)

    # Creating the training and test set
    task_1_epochs = df.groupby("condition")["epoch"].unique()[task_1]
    task_2_epochs = df.groupby("condition")["epoch"].unique()[task_2]

    task_1_train_split = int(len(task_1_epochs) * split_fraction)
    task_2_train_split = int(split_fraction * len(task_2_epochs))

    task_1_train_epochs = task_1_epochs[:task_1_train_split]
    task_2_train_epochs = task_2_epochs[:task_2_train_split]

    task_1_test_epochs = task_1_epochs[task_1_train_split:]
    task_2_test_epochs = task_2_epochs[task_2_train_split:]

    task_1_train_data = df.loc[df["epoch"].isin(task_1_train_epochs)]
    task_2_train_data = df.loc[df["epoch"].isin(task_2_train_epochs)]

    task_1_test_data = df.loc[df["epoch"].isin(task_1_test_epochs)]
    task_2_test_data = df.loc[df["epoch"].isin(task_2_test_epochs)]

    train_df = pd.concat([task_1_train_data, task_2_train_data])
    test_df = pd.concat([task_1_test_data, task_2_test_data])

    x_train = train_df.drop(["condition", "epoch", "time"], axis=1).values
    y_train = train_df.groupby("epoch").first()["condition"]
    y_train = [1 if y == task_1 else 0 for y in y_train]

    x_test = test_df.drop(["condition", "epoch", "time"], axis=1).values
    y_test = test_df.groupby("epoch").first()["condition"]
    y_test = [1 if y == task_1 else 0 for y in y_test]

    x_train = normalize(x_train)
    x_test = normalize(x_test)

    batch_size = config.get("batch_size")
    dense_units = config.get("dense_units")

    dataset_train = keras.preprocessing.timeseries_dataset_from_array(
        x_train,
        y_train,
        shuffle=True,
        batch_size=batch_size,
        sequence_length=past,
        sequence_stride=past)

    dataset_val = keras.preprocessing.timeseries_dataset_from_array(
        x_test,
        y_test,
        shuffle=False,
        batch_size=batch_size,
        sequence_length=past,
        sequence_stride=past)

    if v:
        print(
            f"Take batches out of the training dataset (currently {batch_size} samples)")
    for batch in dataset_train.take(1):
        inputs, targets = batch

    print("Y in test set", targets.numpy().flatten())

    print("Input shape:", inputs.numpy().shape)
    print("Target shape:", targets.numpy().shape)

    model_path = "models/model-3-stack-16-batch-16.h5"
    path_checkpoint = "model_checkpoint.h5"

    source_model = keras.models.load_model(model_path)

    units = source_model.layers[1].layer.units
    dropout = source_model.layers[1].layer.dropout

    ic(units, dropout)

    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2])))

    # model.add(keras.layers.Dense(
    #     512, activation="relu", name="de_out"))

    model.add(keras.layers.Bidirectional(
        keras.layers.LSTM(units=units,
                          activation='tanh',
                          dropout=dropout,
                          return_sequences=True
                          ), name="bi"))

    model.add(keras.layers.Dense(
        dense_units, activation="relu", name="de_end"))

    model.add(keras.layers.Dense(1, activation="sigmoid", name="de_output"))

    model.summary()

    opt = keras.optimizers.Nadam(learning_rate=config.get("learning_rate"))
    model.compile(loss="binary_crossentropy",
                  optimizer=opt,
                  metrics=[
                      'binary_crossentropy',
                      custom_binary_accuracy,
                      f1
                  ])

    es_callback = keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=0, patience=50, verbose=1, mode="max")

    modelckpt_callback = keras.callbacks.ModelCheckpoint(
        monitor="binary_crossentropy",
        filepath=path_checkpoint,
        verbose=1,
        save_weights_only=True,
        save_best_only=True,)

    history = model.fit(
        dataset_train,
        epochs=config.get("epochs"),
        validation_data=dataset_val,
        callbacks=[es_callback, modelckpt_callback,
                   WandbCallback(data_type="time series")],
    )


if __name__ == '__main__':
    main()
