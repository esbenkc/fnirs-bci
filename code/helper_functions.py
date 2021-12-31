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
import tensorflow as tf
import random


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


def create_dataset(path, n_aug=0, output_path=None, split_fraction=0.6):
    """
    Creates a dataset from the raw data.

    :param str path: path to the raw data
    :return: x_train, y_train, x_test, y_test
    """
    filter_haemo = load_and_process_raw(path)
    ic("Extract epochs from raw data...")

    task_1 = re.findall(r'(?<=\d_).*(?=_)', path)[0]
    task_2 = re.findall(r'(?<=_)[a-z]{3,10}(?=.snirf)', path)[0]

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

    # samples, 39, 200
    print(f"{bcolors.ITALIC}Augmenting data from shape {x_train.shape}.{bcolors.ENDC}")

    # Augmenting data
    x_train_aug = x_train.copy()
    x_test_aug = x_test.copy()
    y_train_aug = y_train.copy()
    y_test_aug = y_test.copy()
    for i in range(n_aug):
        x_train = np.append(x_train, augment_data(x_train_aug), axis=0)
        x_test = np.append(x_test, augment_data(x_test_aug), axis=0)
        y_train = np.append(y_train, y_train_aug)
        y_test = np.append(y_test, y_test_aug)

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print(f"{bcolors.ITALIC}Input shape after augmentation: {x_train.shape}.\nTarget shape after augmentation: {y_train.shape}.{bcolors.ENDC}")

    print(f"{bcolors.OK}Test input shape after augmentation: {x_test.shape}.\nTest target shape after augmentation: {y_test.shape}.{bcolors.ENDC}")

    if output_path is not None:
        print(f"{bcolors.OK}Saving dataset to {output_path}.{bcolors.ENDC}")
        np.save(output_path + str(n_aug) + "_x_train.npy", x_train)
        np.save(output_path + str(n_aug) + "_y_train.npy", y_train)
        np.save(output_path + str(n_aug) + "_x_test.npy", x_test)
        np.save(output_path + str(n_aug) + "_y_test.npy", y_test)

    return x_train, y_train, x_test, y_test


def load_and_process(path):
    """
    Load raw data and preprocess
    """
    raw = preprocess(path,
                     verbose=False,
                     tddr=True,
                     l_pass=0.7,
                     h_pass=0.01,
                     bandpass=True,
                     short_ch_reg=False)
    return raw.to_data_frame()


def load_and_process_raw(path):
    """
    Load raw data and preprocess
    """
    raw = preprocess(path,
                     verbose=False,
                     tddr=True,
                     l_pass=0.7,
                     h_pass=0.01,
                     bandpass=True,
                     short_ch_reg=False)
    return raw


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


def augment_data(df, past):

    df_new = df.copy()
    for i in range(1, past):
        df_new = df_new.append(df.shift(i))
    return df_new

    return df


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
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))


def jitter(np_array, mu, sigma):
    """
    Add gaussian noise to each column
    """
    return np_array + np.random.normal(mu, sigma, np_array.shape)


def scale(np_array, min, max):
    """
    Make a random uniform variable between min and max and scale the axis 2 array
    """
    np_aug = np_array.copy()
    k = random.uniform(min, max)
    for i in range(np_aug.shape[1]):
        np_aug[:, i] = np_aug[:, i] * k
    return np_aug


def sample_gaussian_pdf(x, mu, sigma):
    """
    Sample a gaussian pdf
    """
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.))) / np.sqrt(2 * np.pi * np.power(sigma, 2.))


def gaussian_random_walk(length, x, y):
    """
    Generate a gaussian random walk
    """
    walk = np.zeros(length)
    for i in range(1, length):
        epsilon = np.random.normal(0, x)
        walk[i] = walk[i-1] + \
            sample_gaussian_pdf(walk[i] + epsilon, 0, y) * epsilon
    return walk


def add_random_gaussian_walk(np_array, x, y):
    """
    Add a gaussian random walk to the dataframe
    """
    np_aug = np_array.copy()
    for i in range(np_aug.shape[1]):
        np_aug[:, i] = np_aug[:, i] + \
            gaussian_random_walk(np_aug.shape[0], x, y)
    return np_aug


def augment_data(np_array, gaussian_walk=True, gaussian_jitter=True, scale_aug=True, jitter_sigma=0.05, x=0.05, y=0.1, min=0.9, max=1.1):
    """
    Augment the dataframe
    """
    np_aug = np_array.copy()
    if gaussian_walk:
        np_aug = add_random_gaussian_walk(np_aug, x, y)
    if gaussian_jitter:
        np_aug = jitter(np_aug, 0, jitter_sigma)
    if scale_aug:
        np_aug = scale(np_aug, min, max)
    return np_aug


def reset_weights(ix, model, weights_path="data/weights-lstm-layer.npy"):
    """
    Reset weights of a layer
    """
    new_weights = np.load(weights_path, allow_pickle=True)
    ix.set_weights(new_weights)
