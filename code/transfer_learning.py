from train_st_esben import preprocess, visualize_loss, show_plot
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
    import time

    import wandb
    from wandb.keras import WandbCallback

    from absl import flags
    from absl import app


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


# FLAGS = flags.FLAGS

# flags.DEFINE_string('name', 'Jane Random', 'Your name.')
# flags.DEFINE_integer('age', None, 'Your age in years.', lower_bound=0)
# flags.DEFINE_enum('job', 'running', ['running', 'stopped'], 'Job status.')

# flags.DEFINE_string('path', 'data/21-10-13_bci_task_1.snirf',
#                     'Path to BCI task.')
# flags.DEFINE_boolean('verbose', False, 'Produces debugging output.')
# flags.DEFINE_boolean('trainable', False, 'Is the LSTM layer trainable?')
# flags.DEFINE_boolean('random_init', False,
#                      'Should we randomly initialize a new LSTM?')

def main(argv):
    v = True
    # v = FLAGS.verbose
    # raw_path = FLAGS.path
    # trainable = FLAGS.trainable
    # random_init = FLAGS.random_init

    config = {
        'dropout': 0.1,
        'dropout_2': 0.5,
        'train_split': 0.6,
        'preprocess': "medium",
        'batch_size': 20,
        'epochs': 100,
        'trainable': False,
        'pretrained': False,
        'dense_units': 256,
    }

    wandb.init(
        project="fnirs_transfer", entity="esbenkran",
        tags=["middle", "fourth", "16"], config=config)

    config = wandb.config

    raw_path = "data/21-10-13_bci_task_1.snirf"

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

    filter_haemo.annotations.rename(
        {
            '0': 'Nothing',
            '1': 'Waving arms',
            '2': 'Talking'
        })

    events, event_dict = mne.events_from_annotations(
        filter_haemo, verbose=False)

    epochs = mne.Epochs(filter_haemo, events=events, event_id=event_dict,
                        tmin=0.0, tmax=10.0, baseline=(0, 0.5),
                        preload=True,
                        verbose=False)

    df = epochs[['Waving arms', 'Talking']].to_data_frame()

    # Creating the training and test set
    talking_epochs = df.groupby("condition")["epoch"].unique().Talking
    arms_epochs = df.groupby("condition")["epoch"].unique()["Waving arms"]

    talking_train_split = int(len(talking_epochs) * split_fraction)
    amrs_train_split = int(split_fraction * len(arms_epochs))

    talking_train_epochs = talking_epochs[:talking_train_split]
    arms_train_epochs = arms_epochs[:amrs_train_split]

    talking_test_epochs = talking_epochs[talking_train_split:]
    arms_test_epochs = arms_epochs[amrs_train_split:]

    talking_train_data = df.loc[df["epoch"].isin(talking_train_epochs)]
    arms_train_data = df.loc[df["epoch"].isin(arms_train_epochs)]

    talking_test_data = df.loc[df["epoch"].isin(talking_test_epochs)]
    arms_test_data = df.loc[df["epoch"].isin(arms_test_epochs)]

    train_df = pd.concat([talking_train_data, arms_train_data])

    x_train = train_df.drop(["condition", "epoch", "time"], axis=1).values
    y_train = train_df.groupby("epoch").first()["condition"]
    y_train = [1 if y == "Talking" else 0 for y in y_train]

    x_train = normalize(x_train)

    batch_size = config.get("batch_size")
    dense_units = config.get("dense_units")

    dataset_train = keras.preprocessing.timeseries_dataset_from_array(
        x_train,
        y_train,
        shuffle=True,
        batch_size=batch_size,
        sequence_length=past,
        sequence_stride=past)

    test_df = pd.concat([talking_test_data, arms_test_data])

    x_test = test_df.drop(["condition", "epoch", "time"], axis=1).values
    y_test = test_df.groupby("epoch").first()["condition"]
    y_test = [1 if y == "Talking" else 0 for y in y_test]

    x_test = normalize(x_test)

    dataset_val = keras.preprocessing.timeseries_dataset_from_array(
        x_test,
        y_test,
        shuffle=True,
        batch_size=batch_size,
        sequence_length=past,
        sequence_stride=past)

    if v:
        print(
            f"Take batches out of the training dataset (currently {batch_size} samples)")
    for batch in dataset_train.take(1):
        inputs, targets = batch

    print("Input shape:", inputs.numpy().shape)
    print("Target shape:", targets.numpy().shape)

    # model_path = "wandb\\run-20211214_112845-tmz11bej\\files\\model-best.h5"
    model_path = "wandb\\run-20211215_145437-3qngff05\\files\\model-best.h5"
    path_checkpoint = "model_checkpoint.h5"

    source_model = keras.models.load_model(model_path)

    units = source_model.layers[1].layer.units
    dropout = source_model.layers[1].layer.dropout

    ic(units, dropout)

    model = keras.Sequential()
    for layer in source_model.layers[:-1 if config.get("pretrained") else -2]:
        # Go through until last layer to remove the Dense 1-neuron output layer
        model.add(layer)

    if not config.get("pretrained"):
        model.add(keras.layers.Bidirectional(
            keras.layers.LSTM(units=units,
                              activation='tanh',
                              dropout=dropout
                              )))

    model.add(keras.layers.Dropout(config.get("dropout_2")))
    model.add(keras.layers.Dense(dense_units, activation="relu"))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    model.summary()
    ic(config.get("trainable"))
    ic(config.get("pretrained"))

    opt = keras.optimizers.Adam()
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['binary_crossentropy'])

    model.get_layer("bidirectional").trainable = config.get("trainable")
    ic("Model loaded and LSTM.trainable set to...",
       model.get_layer("bidirectional").trainable)

    es_callback = keras.callbacks.EarlyStopping(
        monitor="binary_crossentropy", min_delta=0, patience=50, verbose=1, mode="min")

    modelckpt_callback = keras.callbacks.ModelCheckpoint(
        monitor="binary_crossentropy",
        filepath=path_checkpoint,
        verbose=1,
        save_weights_only=True,
        save_best_only=True,)

    # for x, y in dataset_train.take(1):
    #     print(x)
    #     print(y)
    # for x, y in dataset_val.take(1):
    #     print(x)
    #     print(y)

    history = model.fit(
        dataset_train,
        epochs=config.get("epochs"),
        validation_data=dataset_val,
        callbacks=[es_callback, modelckpt_callback,
                   WandbCallback(data_type="time series")],
    )

    # visualize_loss(history, "Training and Validation Loss")


# if __name__ == '__main__':
#     app.run(main)

main("me")
