
#!/usr/bin/python

# %%
# Import relevant libraries
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
    step_haemo = beer_lambert_law(step_od, ppf=6)

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
    :param int iteration: current iteration
    :param int total: total iterations
    :param str prefix: prefix string
    :param str suffix: suffix string
    :param int decimals: positive number of decimals in percent complete
    :param int length: character length of bar
    :param str fill: bar fill character
    :param str printEnd: end character
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def normalize_and_remove_time(df, df_ref=None):
    """
    Normalize all numerical values in dataframe
    :param df: dataframe
    :param df_ref: reference dataframe
    """
    df.index = df['time']
    if df_ref is None:
        df_ref = df
    df_ref.index = df_ref['time']
    df_norm = (df - df_ref.mean()) / df_ref.std()
    df_norm.drop('time', axis=1, inplace=True)
    return df_norm


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


def using_mstats_df(df):
    return df.apply(using_mstats, axis=0)


def using_mstats(s):
    return mstats.winsorize(s, limits=[0.05, 0.05])


v = True
p = False
p_loss = False

wb = train = True
v = p = p_loss = False

# if __name__ == "__main__":
#     args = getopt.getopt(sys.argv[1:], "hi:o:", ["ifile=", "ofile="])
#     wb = True if "wb" in args[1] else False
#     v = True if "v" in args[1] else False
#     p = True if "p" in args[1] else False
#     train = True if "train" in args[1] else False
#     p_loss = True if "loss" in args[1] else False

# print(f"{bcolors.HEADER}Running with the following arguments:",
#       args[1], f"{bcolors.ENDC}")

config = {
    "learning_rate": 0.00002,
    "epochs": 100,
    "batch_size": 20,
    "loss_function": "mse",
    "optimizer": "adam",
    "dropout": 0.5,
    "units": 100,
    "past": 39,
    "future": 16,
    "preprocess": "simple",
    "bidirectional": True,
    "activation_function": "tanh",
    "normalize": True,
    "l_pass": 0.7,
    "h_pass": 0.01,
    "train_split": 0.7,
    "raw_path": "data/2021-09-22_004.snirf"
}

raw_path = config.get("raw_path")

paths = ["data/2021-09-22_004.snirf",
         "data/21-10-13_experiment_test.snirf",
         "data/21-10-13_pretrain_5.snirf",
         "data/21-09-15_pretrain_1.snirf",
         "data/21-09-22_pretrain_3.snirf"]

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

ic(nrows_test)
ic(idx)
ic(df_nrows[idx])
ic(df_nrows)

if v:
    ic("Normalise the data with a basis in the train split so there's no data leakage")

test_features = normalize_and_remove_time(
    full_df[idx])

train_features = [normalize_and_remove_time(df) for df in full_df]
train_features = pd.concat(train_features)

if p:
    if v:
        ic("Plot this stuff.")

    test_features.plot(use_index=True, alpha=0.2, title="Test data").get_figure().savefig(
        "output/test_data.png")
    train_features.plot(use_index=True, alpha=0.2, title="Train data").get_figure(
    ).savefig("output/train_data.png")

# Show heatmap of all column correlations (channel correlations)
if p:
    show_heatmap(train_features)

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

if v:
    ic("Select the training data from 0 to the train split")
train_data = train_features

if v:
    ic("Select the test data from train_split to end")
val_data = test_features

if v:
    ic("Select the start and end indices for the training data")
start = past + future
end = start + len(train_data)

if v:
    ic("Select all train data from start to end as one big array of arrays")
# y = one value per array in x

x_train = train_data.values
y_train = train_features.iloc[start:end][["S1_D1 hbo"]]

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
    ic("Set the max end index for the validation X data")
x_end = len(val_data) - past - future

if v:
    ic("Set the label start")
label_start = past + future

if v:
    ic("Make a validation dataset from like the train dataset")
x_val = val_data.iloc[:x_end].values
y_val = val_data.iloc[label_start:][["S1_D1 hbo"]]

if v:
    ic("Calculate the chance level with simple same-value prediction")
y_list = np.array(y_val.iloc[:-40]).ravel()
x_list = x_val[40:, 0]

chance_df = pd.DataFrame(
    data={
        "y": y_list,
        "x": x_list
    })

chance_df["guess"] = [np.mean(y_list) for i in range(len(chance_df))]
chance_df["gauss"] = [np.random.normal(0) for i in range(len(chance_df))]

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


chance_gauss = chance_df["diff_gauss"].mean()
chance_mean = chance_df["diff_mean"].mean()
chance_last = chance_df["diff_last"].mean()

print(f"{bcolors.HEADER}Mean value performance:",
      chance_mean, f"{bcolors.ENDC}")
print(f"{bcolors.HEADER}Last value chance performance:",
      chance_last, f"{bcolors.ENDC}")
print(f"{bcolors.HEADER}Gaussian chance performance:",
      chance_gauss, f"{bcolors.ENDC}")

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

if wb:
    wandb.init(project="fnirs_ml", entity="esbenkran",
               name=f"ubdt{int(random.random() * 1000)}", config=config, tags=["ALL"])
    config = wandb.config

if train:
    if v:
        ic("Define the model architecture")
    inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
    if config.get("bidirectional"):
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
    else:
        lstm_out = keras.layers.LSTM(
            config.get("units"),
            activation=config.get("activation_function"),
            dropout=config.get("dropout"))(inputs)

    dense_out = keras.layers.Dense(128, activation="relu")(lstm_out)
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

index = 0
# for x, y in dataset_val:
#     index += 1
#     predictions.append(model.predict(x)[0])
#     diff.append(model.predict(x)[0] - y[0])
#     printProgressBar(index, len(dataset_val), length=50,
#                      prefix=f"{bcolors.HEADER}Progress:", suffix=f"{index}/{len(dataset_val)}{bcolors.ENDC}")


# print(
#     f"{bcolors.HEADER}Mean delta (distance from real): {bcolors.WARN}{np.mean(np.abs(diff))}{bcolors.ENDC}\n{bcolors.HEADER}Chance delta (distance from real): {bcolors.WARN}{chance} {bcolors.ENDC}\n{bcolors.HEADER}Standard deviation: {bcolors.WARN}1.0{bcolors.ENDC}\n{bcolors.HEADER}Mean: {bcolors.WARN}0.0{bcolors.ENDC}")

for batch in dataset_val.take(1):
    inputs, targets = batch
# ic(inputs.shape)

predictions = [model.predict(i[None, ...]) for i in inputs]
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
