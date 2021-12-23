import mne
import mne_nirs
import pandas as pd
import numpy as np
from helper_functions import *
from keras.models import load_model
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
import bcolors


def generate_predict_data(path, start_time, end_time, model_paths, past=39, futures=[16]):
    """
    Creates a dataframe with index and real values along with two columns for each model:
    1. The predicted values from the real data
    2. The predicted values from the self-generated data

    :param str path: path to the raw data
    :param int start_time: start index of the prediction
    :param int end_time: end index of the prediction
    :param list model_paths: paths to the models
    :param int past: length of the past
    :param list futures: list of future lengths (as long as the model_paths)
    :return: dataframe with index and real values along with two columns for each model
    """
    nirs = load_and_process(path)
    nirs = normalize_and_remove_time(df=nirs).values
    nirs = nirs[start_time:end_time, :]

    X = nirs

    df = pd.DataFrame()
    df["Real"] = nirs[:, 0]

    for idx, model_path in enumerate(model_paths):
        print(
            f"{bcolors.OK}Loading model {model_path} with future value of {futures[idx]}. Model number: {idx+1}.{bcolors.ENDC}")
        start = past + futures[idx]
        sequence_length = past

        X = nirs

        Y = X[start:, [0]]
        X = X[:-futures[idx], :]

        batch_size = len(X)

        dataset = timeseries_dataset_from_array(
            X,
            Y,
            sequence_length=sequence_length,
            sampling_rate=1,
            batch_size=batch_size,
            sequence_stride=1,
        )

        for batch in dataset.take(1):
            inputs, targets = batch

        if "model" in model_path:
            model = load_model(model_path)
            predictions = [model.predict(i[None, ...]) for i in inputs]
        elif model_path == "mean":
            predictions = [np.mean(inputs.values) for i in inputs]
        elif model_path == "last_value":
            predictions = [inputs[:, -1, 0]]
        elif model_path == "gaussian_random":
            predictions = [np.random.normal(0, 1, len(inputs))]

        print(
            f"{bcolors.HEADER}Start index: {start}\nStart padding shape: {np.array([nirs[i, 0] for i in range(start)]).shape}\nEvaulation shape: {np.array(np.concatenate(predictions)).flatten().shape}{bcolors.ENDC}")

        df["Prediction_" + str(idx) + "_" + str(futures[idx])
           ] = np.concatenate(([nirs[i, 0] for i in range(start)], np.array(np.concatenate(predictions)).flatten()))

        # Use the predictions as input
        X = nirs
        X[:, 0] = np.array(
            df["Prediction_" + str(idx) + "_" + str(futures[idx])])
        X = X[:-futures[idx], :]

        dataset = timeseries_dataset_from_array(
            X,
            Y,
            sequence_length=sequence_length,
            sampling_rate=1,
            batch_size=batch_size,
            sequence_stride=1,
        )

        for batch in dataset.take(1):
            inputs, targets = batch

        if "model" in model_path:
            predictions = [model.predict(i[None, ...]) for i in inputs]
        elif model_path == "mean":
            predictions = [np.mean(inputs, axis=1)]
        elif model_path == "last_value":
            predictions = [inputs[:, -1, 0]]
        elif model_path == "gaussian_random":
            predictions = [np.random.normal(0, 1, len(inputs))]

        df["Prediction_self_" + str(idx) + "_" + str(futures[idx])
           ] = np.concatenate(([nirs[i, 0] for i in range(start)], np.array(np.concatenate(predictions)).flatten()))

        print(f"{bcolors.OK}Finished with this loop.{bcolors.ENDC}")

    df["Index"] = range(len(df))
    df.plot(x="Index", y=np.concatenate(
        (
            ["Real"],
            ["Prediction_self_" +
             str(i) + "_" + str(futures[i]) for i in range(len(futures))],
            ["Prediction_" + str(i) + "_" + str(futures[i]) for i in range(len(futures))]))).get_figure().savefig("media/prediction.png")
    df.to_csv("data/prediction_example.csv")

    return df


if __name__ == "__main__":
    generate_predict_data(
        "data/snirf/pretrain_3.snirf",
        1200, 1400,
        model_paths=[
            "models/model-16.h5",
            "models/model-3-stack-16.h5",
            "models/model-4.h5",
            "gaussian_random",
            "mean",
            "last_value", ],
        futures=[16, 16, 4, 16, 16, 16])
