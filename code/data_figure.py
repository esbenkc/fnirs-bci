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
            f"{bcolors.HEADER}Loading model {model_path} with future value of {futures[idx]}. Model number: {idx+1}.{bcolors.ENDC}")

        start = past + futures[idx]
        sequence_length = past

        X = nirs

        print(f"X shape: {X.shape}")

        if "model" in model_path:
            # This seems to give an error of the length of input: 129 instead of 145
            X = X[:-futures[idx], :]
            Y = X[start:, [0]]
            batch_size = len(Y)
            dataset = timeseries_dataset_from_array(
                X,
                Y,
                sequence_length=sequence_length,
                sampling_rate=1,
                batch_size=batch_size,
                sequence_stride=1,
            )
        else:
            X = X
            Y = X[start:, [0]]
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

        print("Input shape:", inputs.numpy().shape)
        print("Target shape:", targets.numpy().shape)

        if "model" in model_path:
            model = load_model(model_path)
            if "dense" in model_path:
                print("Dense model predictions")
                print(inputs.numpy()[0][:, 0].shape)
                predictions = model.predict(inputs.numpy()[:, :, 0])
                # predictions = [model.predict(i[:, 0])
                #                for i in inputs]
            elif "lstm" in model_path:
                print("LSTM model predictions")
                predictions = [model.predict(i[None, ...]) for i in inputs]
        elif model_path == "mean":
            predictions = np.array([[np.array(X).mean() for i in inputs]])
        elif model_path == "last_value":
            predictions = [inputs[:, -1, 0]]
        elif model_path == "gaussian_random":
            predictions = [np.random.normal(0, 1, len(inputs))]

        print(
            f"{bcolors.HEADER}Start index: {start}\nStart padding shape: {np.array([nirs[i, 0] for i in range(start)]).shape}\nEvaluation shape: {np.array(np.concatenate(predictions)).flatten().shape}{bcolors.ENDC}")

        df[f"Prediction_{idx}_{futures[idx]}"] = np.concatenate(
            ([np.NaN for i in range(start)], np.array(np.concatenate(predictions)).flatten()))

        print(f"{bcolors.OK}Finished with this loop.{bcolors.ENDC}")

    df["Index"] = range(len(df))
    df.plot(x="Index", y=np.concatenate(
        (
            ["Real"],
            ["Prediction_" + str(i) + "_" + str(futures[i]) for i in range(len(futures))]))).get_figure().savefig("media/prediction.png")
    df.to_csv("data/visualization/prediction_plot.csv")

    return df


if __name__ == "__main__":
    generate_predict_data(
        "data/snirf/pretrain_3.snirf",
        2000, 2200,
        model_paths=[
            "last_value",
            "models/model-dense.h5",
            "models/model-lstm.h5",
            "models/model-lstm-3.h5",
            "mean",
        ],
        futures=[16, 16, 16, 16, 16])
