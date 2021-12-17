#!/usr/bin/python

# %%
# Import relevant libraries
import warnings
from icecream import ic
ic("Importing packages...")
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    import os
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


# %%


def preprocess(path, l_pass=0.7, h_pass=0.01, bandpass=True, short_ch_reg=True, tddr=True, negative_correlation=True, verbose=True, return_all=False):
    """
    Load raw data and preprocess
    :param str path: path to the raw data
    :param float l_pass: low pass frequency
    :param float h_pass: high pass frequency
    :param bool bandpass: apply bandpass filter
    :param bool short_ch_reg: apply short channel regression
    :param bool tddr: apply tddr
    :param bool negative_correlation: apply negative correlation
    :param bool verbose: print progress
    :return: preprocessed data
    """
    if verbose:
        ic("Loading ", path)
    raw_intensity = mne.io.read_raw_snirf(path, preload=True)
    step_od = mne.preprocessing.nirs.optical_density(raw_intensity)

    # sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od, l_freq=0.7, h_freq=1.5)
    # raw_od.info['bads'] = list(compress(raw_od.ch_names, sci < 0.5))

    if verbose:
        ic("Apply short channel regression.")
    if short_ch_reg:
        step_od = mne_nirs.signal_enhancement.short_channel_regression(step_od)

    if verbose:
        ic("Do temporal derivative distribution repair on:", step_od)
    if tddr:
        step_od = mne.preprocessing.nirs.tddr(step_od)

    if verbose:
        ic("Convert to haemoglobin with the modified beer-lambert law.")
    step_haemo = beer_lambert_law(step_od, ppf=0.1)

    if verbose:
        ic("Apply further data cleaning techniques and extract epochs.")
    if negative_correlation:
        step_haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(
            step_haemo)

    if not return_all:
        if verbose:
            ic("Separate the long channels and short channels.")
        short_chs = get_short_channels(step_haemo)
        step_haemo = get_long_channels(step_haemo)

    if verbose:
        ic("Bandpass filter on:", step_haemo)
    if bandpass:
        step_haemo = step_haemo.filter(
            h_pass, l_pass, h_trans_bandwidth=0.3, l_trans_bandwidth=h_pass*0.25)
    return step_haemo


def show_heatmap(data):
    """
    Show a heatmap of all column correlations
    """
    plt.matshow(data.corr())
    plt.xticks(range(data.shape[1]),
               data.columns, fontsize=14, rotation=90)
    plt.gca().xaxis.tick_bottom()
    plt.yticks(range(data.shape[1]), data.columns, fontsize=14)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title("Feature Correlation Heatmap", fontsize=14)
    plt.show()


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def normalize(data, train_split):
    """
    Normalize the data to z distribution
    """
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std


def visualize_loss(history, title):
    """
    Visualize the loss
    """
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def show_plot(plot_data, delta, title):
    """
    Show plot to evaluate the model
    """
    labels = ["History", "True Future", "Model Prediction"]
    marker = [".-", "rx", "go"]
    time_steps = list(range(-(plot_data[0].shape[0]), 0))
    # ic(time_steps)
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, val in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i],
                     markersize=10, label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(
            ), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future * 2)])
    plt.xlabel("Time-Step")
    plt.show()
    return


wb = True
v = False
p = False
train = True
p_loss = False

wb = train = True
v = p = p_loss = False

if __name__ == "__main__":
    args = getopt.getopt(sys.argv[1:], "hi:o:", ["ifile=", "ofile="])
    wb = False if "wb" in args[1] else True
    v = True if "v" in args[1] else False
    p = True if "p" in args[1] else False
    train = False if "train" in args[1] else True
    p_loss = True if "loss" in args[1] else False

    print(f"{bcolors.HEADER}Running with the following arguments:",
          args[1], f"{bcolors.ENDC}")

    config = {
        "learning_rate": 0.00002,
        "epochs": 100,
        "batch_size": 48,
        "loss_function": "mse",
        "optimizer": "adam",
        "dropout": 0.5,
        "units": 4,
        "past": 39,
        "future": 4,
        "preprocess": "simple",
        "bidirectional": True,
        "activation_function": "tanh",
        "dropout": 0.0,
        "normalize": True,
        "l_pass": 0.7,
        "h_pass": 0.01,
        "train_split": 0.7,
        "raw_path": "data/2021-09-22_004.snirf"
    }

    if wb:
        wandb.init(project="fnirs_ml", entity="esbenkran",
                   config=config)
        config = wandb.config

    raw_path = config.get("raw_path")

    if config.get("preprocess") == "none":
        filter_haemo = preprocess(raw_path,
                                  config.get("l_pass"),
                                  config.get("h_pass"),
                                  bandpass=False,
                                  short_ch_reg=False,
                                  tddr=False,
                                  negative_correlation=False,
                                  verbose=v)
    if config.get("preprocess") == "simple":
        if v:
            ic("Simple preprocessing:", raw_path)
        filter_haemo = preprocess(raw_path,
                                  config.get("l_pass"),
                                  config.get("h_pass"),
                                  bandpass=True,
                                  short_ch_reg=False, tddr=False, negative_correlation=False, verbose=v)

    elif config.get("preprocess") == "medium":
        if v:
            ic("Medium preprocessing:", raw_path)
        filter_haemo = preprocess(raw_path,
                                  config.get("l_pass"),
                                  config.get("h_pass"),
                                  bandpass=True,
                                  short_ch_reg=True, tddr=False, negative_correlation=False, verbose=v)

    elif config.get("preprocess") == "advanced":
        if v:
            ic("Advanced preprocessing:", raw_path)
        filter_haemo = preprocess(raw_path,
                                  config.get("l_pass"),
                                  config.get("h_pass"),
                                  bandpass=True,
                                  short_ch_reg=True, tddr=True, negative_correlation=True, verbose=v)

    # Check the data
    if p:
        pl = filter_haemo.plot(n_channels=20, duration=100000,
                               start=100, show_scrollbars=False)

    # %%
    # Make the hemoglobin values into a dataframe
    df = filter_haemo.to_data_frame()

    # Show heatmap of all column correlations (channel correlations)
    if p:
        show_heatmap(df)

    # How big proportion should be training data
    split_fraction = config.get("train_split")

    # Splitting the data into training and test set
    train_split = int(split_fraction * int(df.shape[0]))

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
    patience = 15

    # What is the time column?
    date_time_key = "time"

    if v:
        ic("Define the features and set index to the time column of all data")
    features = df
    features.index = df[date_time_key]
    features = features.drop('time', axis=1)

    if v:
        ic("Normalise the data with a basis in the train split so there's no data leakage")

    features = normalize(features.values, train_split)
    features = pd.DataFrame(features)

    if v:
        ic("Select the training data from 0 to the train split")
    train_data = features.loc[0: train_split - 1]

    if v:
        ic("Select the test data from train_split to end")
    val_data = features.loc[train_split:]

    if v:
        ic("Select the start and end indices for the training data")
    start = past + future
    end = start + train_split

    if v:
        ic("Select all train data from start to end as one big array of arrays")
    # y = one value per array in x

    x_train = train_data[[i for i in range(len(train_data.columns))]].values
    y_train = features.iloc[start:end][[1]]

    if v:
        ic("Define sequence length from past values divided by the step (in this case 1)")
    sequence_length = int(past / step)

    print(y_train.shape)
    print(x_train.shape)

    if v:
        ic("Make a training dataset from arrays with definitions of sequence length")
    dataset_train = keras.preprocessing.timeseries_dataset_from_array(
        x_train,
        y_train,
        sequence_length=sequence_length,
        sampling_rate=step,
        batch_size=batch_size,
    )

    if v:
        ic("Set the max end index for the validation data")
    x_end = len(val_data) - past - future

    if v:
        ic("Set the label start")
    label_start = train_split + past + future

    if v:
        ic("Make a validation dataset from like the train dataset")
    x_val = val_data.iloc[:x_end][[
        i for i in range(len(train_data.columns))]].values
    y_val = features.iloc[label_start:][[1]]

    if v:
        ic("Calculate the chance level with gaussian distribution")
    y_df = pd.DataFrame(y_val.append(y_train))
    y_df.columns = ["y"]
    y_df["guess"] = [np.random.normal(0) for i in range(len(y_df))]
    y_df["diff"] = [np.abs(y_df["guess"].iloc[i] - y_df["y"].iloc[i])
                    for i in range(len(y_df))]
    chance = y_df["diff"].mean()
    print(f"{bcolors.HEADER}Chance performance:", chance, f"{bcolors.ENDC}")

    if v:
        ic("Calculate the chance level with simple same-value prediction")
    x_df = pd.DataFrame(np.append(x_val[:, -1], x_train[:, -1]))
    x_df.columns = ["x"]
    y_df["guess_last"] = x_df["x"]
    y_df["diff_last"] = [np.abs(y_df["guess_last"].iloc[i] - y_df["y"].iloc[i])
                         for i in range(len(y_df))]
    chance = y_df["diff_last"].mean()
    print(f"{bcolors.HEADER}Last value chance performance:",
          chance, f"{bcolors.ENDC}")

    # ic(y_df)
    # ic(x_val[:, -1])

    if v:
        ic("Make a validation dataset with the same definitions as the training dataset")
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

    if train:
        if v:
            ic("Define the model architecture")
        inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
        if config.get("bidirectional"):
            lstm_out = keras.layers.Bidirectional(
                keras.layers.LSTM(
                    config.get("units"),
                    activation=config.get("activation_function"),
                    dropout=config.get("dropout"),
                ))(inputs)
        else:
            lstm_out = keras.layers.LSTM(
                config.get("units"),
                activation=config.get("activation_function"),
                dropout=config.get("dropout"))(inputs)

        outputs = keras.layers.Dense(1)(lstm_out)

        if v:
            ic("Generate a learning rate schedule with exponential decay")
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=config.get("learning_rate"),
            decay_steps=10000,
            decay_rate=0.96,
            staircase=True)

        if v:
            ic("Define the optimizer")
        if config.get("optimizer") == "adam":
            optimizer = keras.optimizers.Adam(lr_schedule)
        elif config.get("optimizer") == "sdg":
            optimizer = keras.optimizers.SGD(lr_schedule)

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

    index = 0
    # for x, y in dataset_val:
    #     index += 1
    #     predictions.append(model.predict(x)[0])
    #     diff.append(model.predict(x)[0] - y[0])
    #     printProgressBar(index, len(dataset_val), length=50,
    #                      prefix=f"{bcolors.HEADER}Progress:", suffix=f"{index}/{len(dataset_val)}{bcolors.ENDC}")

    #  std = sqrt(mean(x)) , where x = abs(a - a. mean())**2
    print(
        f"{bcolors.HEADER}Mean delta (distance from real): {bcolors.WARN}{np.mean(np.abs(diff))}{bcolors.ENDC}\n{bcolors.HEADER}Chance delta (distance from real): {bcolors.WARN}{chance} {bcolors.ENDC}\n{bcolors.HEADER}Standard deviation: {bcolors.WARN}1.0{bcolors.ENDC}\n{bcolors.HEADER}Mean: {bcolors.WARN}0.0{bcolors.ENDC}")

    for batch in dataset_val.take(1):
        inputs, targets = batch
    # ic(inputs.shape)

    predictions = [model.predict(i[None, ...]) for i in inputs if i.ndim == 3]
    # print(predictions)
    diff = [np.abs(predictions[i][0] - y_val.iloc[i])
            for i in range(len(predictions))]
    # print(diff)

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

    # Save the model
    model.save("model_weights.h5")

    # %%
    """
    Fine-tune the model
    """

    # source_model = keras.models.load_model(path_checkpoint)

    # model = keras.Sequential()
    # for layer in source_model.layers[:-1]:
    #     # Go through until last layer to remove the Dense 1-neuron output layer
    #     model.add(layer)

    # model.summary()
    # model.add(keras.layers.Dropout(0.2))
    # model.add(keras.layers.Dense(100, activation="relu"))
    # model.add(keras.layers.Dense(2, activation="softmax"))
    # model.compile(loss='categorical_crossentropy',
    #                 optimizer='adam', metrics=['mean_absolute_error', 'mean_squared_error'])

    # model.summary()

    # %%

    """
    Pre-training: LSTM på OG pretraining data
    Transfer learning: Lås alting ud over yderste lag, når klassificeringen tegnes
    Fine-tuning: Alting er ulåst, og så bliver den trænet

    Pretraining model bliver ført ind i transfer learning-model, der låser træningen og ændrer lag, som så bliver ført ind i fine-tuning-model, hvor alt er ulåst.

    - Udregn accuracy af modellen med en eller anden metode. Tutorial har ikke noget om dette.
    - Tag ROIs ud af fNIRS-data for at teste, om mere præcis data kan fungere bedre
    - Basically lige finde ud af, om LSTM fungerer
        - LSTM bør fungere godt pga. hel masse data-kanaler med nogle korrelative features
    - Potentielt enhance features:
        - Sigmoid function på input-data, så man får lidt en gate-funktion
        - Negative correlation enhancement
    - Basic machine learning
        - Skal have en virkelig god idé om, hvor god modellerne er for at sammenligne dem
        - Klare variabler såsom accuracy og distance-to-value etc.
    - Først skal vi finde ud af, om den rent faktisk (som GPT) kan forudsige næste værdi
        - Measure of accuracy
        - Leg med hyperparametre: Hvor mange sekunder før gæt, hvor langt ud i fremtiden, mængden af parametre etc.
        - Med forbehold for interpretation af de forskellige parametre

    1. Find ud af, hvordan modellen skal valideres
    - Tjek tutorial på https://keras.io/guides/training/
    - Se om andre papers bruger samme metoder og stjæl dem :))
    - Se _præcist_ hvordan modellen valideres
    2. Kan modellen nok til at få så meget data ind
    3. På min data
        - Kan vi få den til at fungere
        - Hvornår tør vi at sige, at den fungerer?
        - Specifikke parametre, som kan ændres
    4. Andre papers med lignende modeller, LSTM, på forecasting for at se, hvilke accuracies er nice

    ^ Alt the above skal være fungerende, før vi rent faktisk kan transfer learn over til BCI-data

    GPT = næste ord
    GPT-transfer = ...

    MAR

    """
