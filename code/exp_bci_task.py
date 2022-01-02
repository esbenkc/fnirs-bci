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
        'batch_size': 24,
        'epochs': 250,
        'trainable': False,
        'dense_units': 256,
        'layers_transferred': 0,  # [0, 1, 2, 3, 4]
        'bci_task': "data/snirf/bci_task_3_arithmetic_rotation.snirf",
        'n_augmentations': 10,  # 0, 10, 50
        'model': "models/model-lstm.h5",
        'test_channel': 0,
    }

    wandb.init(
        project="thought_classification", entity="esbenkran",
        tags=["transfer_learning", "final", "extension"], config=config)

    config = wandb.config

    raw_path = config.get("bci_task")
    task_1 = re.findall(r'(?<=\d_).*(?=_)', raw_path)[0]
    task_2 = re.findall(r'(?<=_)[a-z]{3,10}(?=.snirf)', raw_path)[0]

    try:
        pre_path = f"data/datasets/{task_1}_{task_2}_{config.get('n_augmentations')}"
        x_train = np.load(f"{pre_path}_x_train.npy")
        y_train = np.load(f"{pre_path}_y_train.npy")
        x_test = np.load(f"{pre_path}_x_test.npy")
        y_test = np.load(f"{pre_path}_y_test.npy")
    except:
        raise Exception(
            f"{bcolors.FAIL}\nNo preprocessed data found for {task_1}_{task_2} with {config.get('n_augmentations')} augmentations.\n\nPlease make a data/datasets directory and run code/generate_datasets.py first.\n\n{bcolors.ENDC}")

    past = 39
    split_fraction = config.get("train_split")
    date_time_key = "time"

    batch_size = config.get("batch_size")
    dense_units = config.get("dense_units")

    # Make each index repeat 39 times in Y
    y_train = np.repeat(y_train, past, axis=0)
    y_test = np.repeat(y_test, past, axis=0)

    if "dense" in config.get("model"):
        dataset_train = keras.preprocessing.timeseries_dataset_from_array(
            x_train[:, config.get("test_channel")].flatten(),
            y_train,
            shuffle=True,
            batch_size=batch_size,
            sequence_length=past,
            sequence_stride=past)

        dataset_val = keras.preprocessing.timeseries_dataset_from_array(
            x_test[:, config.get("test_channel")].flatten(),
            y_test,
            shuffle=False,
            batch_size=batch_size,
            sequence_length=past,
            sequence_stride=past)
    elif "lstm" in config.get("model"):
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
    for batch in dataset_val.take(1):
        inputs, targets = batch

    print("Input shape:", inputs.numpy().shape)
    print("Target shape:", targets.numpy().shape, f"{bcolors.ENDC}")

    # print(f"{bcolors.HEADER}Test set Y {y_test}{bcolors.ENDC}")
    # print(f"{bcolors.HEADER}Y in test set", targets.numpy().flatten())
    # print(f"{bcolors.HEADER}X in test set", inputs.numpy())

    path_checkpoint = "model_checkpoint.h5"

    print(f"{bcolors.ITALIC}Loading model...{config.get('model')}.{bcolors.ENDC}")

    source_model = keras.models.load_model(config.get("model"))
    model = keras.models.Sequential(source_model.layers[:-1])

    print(
        f"{bcolors.ITALIC}The input layer is: {model.layers[0].input_shape}\nand input actual is: {source_model.layers[0].output_shape}{bcolors.ENDC}")

    units = 100
    dense_units = config.get("dense_units")
    dropout = 0.5

    print(f"{bcolors.ITALIC}Source model layers with {units} units (LSTM) or {dense_units} units (Dense) and dropout {dropout}.{bcolors.ENDC}")

    # Reset all layers above layers_transferred
    for layer in range(len(model.layers)):
        # Layers transferred will be none, lstm1 (dense), lstm2 (dense), lstm3, lstm3+dense up to 4 [0, 1, 2, 3, 4]
        if layer not in list(range(config.get("layers_transferred"))):
            if "lstm" in config.get("model"):
                if "de" in model.layers[layer].name:
                    if "lstm-3" in config.get("model"):
                        print(
                            f"{bcolors.HEADER}Resetting dense layer in LSTM-3 {model.layers[layer].name}{bcolors.ENDC}")
                        reset_weights(
                            model.layers[layer], model, "data/weights-dense-128-200-layer.npy")
                    elif "lstm" in config.get("model"):
                        print(
                            f"{bcolors.HEADER}Resetting dense layer in LSTM {model.layers[layer].name}{bcolors.ENDC}")
                        reset_weights(
                            model.layers[layer], model, "data/weights-dense-128-100-layer.npy")
                elif "bi" in model.layers[layer].name:
                    print(
                        f"{bcolors.HEADER}Resetting Bidirectional LSTM layer {model.layers[layer].name}{bcolors.ENDC}")
                    reset_weights(
                        model.layers[layer], model, "data/weights-lstm-bi-layer.npy")
                else:
                    print(
                        f"{bcolors.HEADER}Resetting LSTM uni layer {model.layers[layer].name}{bcolors.ENDC}")
                    reset_weights(
                        model.layers[layer], model, "data/weights-lstm-uni-layer.npy")
            elif "dense" in config.get("model"):
                print(
                    f"{bcolors.HEADER}Resetting dense layer {model.layers[layer].name}{bcolors.ENDC}")
                reset_weights(model.layers[layer], model,
                              "data/weights-dense-128-39-layer.npy")
        if not config.get("trainable"):
            if layer in list(range(config.get("layers_transferred"))):
                model.layers[layer].trainable = False

    model.add(keras.layers.Dense(
        dense_units, activation="relu", name="de_transfer"))
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
        monitor="val_loss", min_delta=0, patience=500, verbose=1, mode="max")

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
