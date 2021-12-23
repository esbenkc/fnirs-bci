import mne
import mne_nirs
import pandas as pd
import numpy as np
from mne_nirs.experimental_design import make_first_level_design_matrix
from mne_nirs.statistics import run_glm
from mne_nirs.channels import (get_long_channels,
                               get_short_channels,
                               picks_pair_to_idx)
from mne.preprocessing.nirs import optical_density, beer_lambert_law
from icecream import ic
from keras import backend as K


def preprocess(path, l_pass=0.7, h_pass=0.01, bandpass=True, short_ch_reg=False, tddr=True, negative_correlation=False, verbose=False, return_all=False):
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


def load_and_process(path):
    """
    Load raw data and preprocess
    """
    raw = preprocess(path,
                     verbose=False,
                     tddr=True,
                     l_pass=0.7,
                     h_pass=0.01,
                     bandpass=True)
    return raw.to_data_frame()


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


def normalize(df, df_ref=None):
    """
    Normalize all numerical values in dataframe
    :param df: dataframe
    :param df_ref: reference dataframe
    """
    if df_ref is None:
        df_ref = df
    df_norm = (df - df_ref.mean()) / df_ref.std()
    return df_norm


def f1(y_true, y_pred):
    """
    F1 metric
    """
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def custom_binary_accuracy(y_true, y_pred):
    """
    Custom binary accuracy metric
    """
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))
