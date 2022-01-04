import pandas as pd
import wandb
import bcolors
api = wandb.Api(timeout=19)


def save_run(run, path):
    """
    Save a run to a file
    :param run: run to save
    :param path: path to save to
    :return: run history
    """
    run = api.run(run)
    append = run.history()
    append["run_id"] = run.id
    append["batch_size"] = run.config.get("batch_size")
    append["future"] = run.config.get("future")
    append["past"] = run.config.get("past")
    append["units"] = run.config.get("units")
    append["dropout"] = run.config.get("dropout")
    append["runtime"] = run.summary.get("runtime")
    append["epoch"] = run.summary.get("epoch")
    append["best_val_loss"] = run.summary.get("best_val_loss")
    append["best_epoch"] = run.summary.get("best_epoch")
    append["pretrained"] = run.config.get("pretrained")
    append["trainable"] = run.config.get("trainable")
    append["preprocess"] = run.config.get("preprocess")
    append["optimizer"] = run.config.get("optimizer")
    append["loss"] = run.config.get("loss")
    append["train_split"] = run.config.get("train_split")
    append["h_pass"] = run.config.get("h_pass")
    append["l_pass"] = run.config.get("l_pass")
    append["layers_transferred"] = run.config.get("layers_transferred")
    append["bci_task"] = run.config.get("bci_task")
    append["repetitions"] = run.config.get("repetitions")
    append["architecture"] = run.config.get("architecture")
    append["pretrain_dense_units"] = run.config.get("pretrain_dense_units")
    append["test_channel"] = run.config.get("test_channel")
    append["model"] = run.config.get("model")
    append["n_augmentations"] = run.config.get("n_augmentations")
    append["dense_units"] = run.config.get("dense_units")

    print(f"{bcolors.HEADER}Saving run history {run.id} to {path}.{bcolors.ENDC}")
    append.to_csv(path)
    return run.history()


def save_runs(runs, path, filter={}, ):
    """
    Save a list of runs to a file
    :param runs: list of runs to save
    :param path: path to save to
    :param filter: filter for the runs
    :return: runs history
    """
    runs_wandb = api.runs(runs, filters=filter)
    print(f"{bcolors.HEADER}Saving {len(runs_wandb)} runs' history to {path}.{bcolors.ENDC}")
    df = pd.DataFrame()
    for run in runs_wandb:
        append = run.history()
        append["run_id"] = run.id
        append["batch_size"] = run.config.get("batch_size")
        append["future"] = run.config.get("future")
        append["past"] = run.config.get("past")
        append["units"] = run.config.get("units")
        append["dropout"] = run.config.get("dropout")
        append["runtime"] = run.summary.get("runtime")
        append["epoch"] = run.summary.get("epoch")
        append["best_val_loss"] = run.summary.get("best_val_loss")
        append["best_epoch"] = run.summary.get("best_epoch")
        append["pretrained"] = run.config.get("pretrained")
        append["trainable"] = run.config.get("trainable")
        append["preprocess"] = run.config.get("preprocess")
        append["optimizer"] = run.config.get("optimizer")
        append["loss"] = run.config.get("loss")
        append["train_split"] = run.config.get("train_split")
        append["h_pass"] = run.config.get("h_pass")
        append["l_pass"] = run.config.get("l_pass")
        append["layers_transferred"] = run.config.get("layers_transferred")
        append["bci_task"] = run.config.get("bci_task")
        append["repetitions"] = run.config.get("repetitions")
        append["architecture"] = run.config.get("architecture")
        append["pretrain_dense_units"] = run.config.get("pretrain_dense_units")
        append["test_channel"] = run.config.get("test_channel")
        append["model"] = run.config.get("model")
        append["n_augmentations"] = run.config.get("n_augmentations")
        append["dense_units"] = run.config.get("dense_units")
        df = df.append(append)
    df.to_csv(path)
    return df


if __name__ == "__main__":
    # save_run("esbenkran/fnirs_ml/3065xozb", "data/analysis/stack_lstm.csv")
    # save_run("esbenkran/fnirs_ml/3qngff05", "data/analysis/lstm.csv")
    # save_run("esbenkran/fnirs_ml/2e2s0nnz", "data/analysis/lstm-4.csv")
    # save_runs("esbenkran/fnirs_sweep", "data/analysis/4_sweep.csv",
    #           {"config.future": 4, "sweep": "v6x95luq"})
    # save_runs("esbenkran/fnirs_sweep",
    #           "data/analysis/16_sweep.csv", {"config.future": 16})
    # save_runs("esbenkran/fnirs_transfer", "data/analysis/16_transfer_sweep.csv",
    #           {"tags": "fourth", })
    # # Other sweeps that run the same config as the below: ["36v7trv1", "v7gbfmpo"]
    # save_runs("esbenkran/fnirs_transfer",
    #           "data/analysis/stack_transfer_layer_freeze.csv",
    #           {"sweep": {"$in": ["fx3h66q6"]}})
    # save_runs("esbenkran/thought_classification",
    #           "data/analysis/thought_classification.csv",
    #           {"sweep": {"$in": ["cp3p80yp"]}})
    # save_runs("esbenkran/thought_classification", "data/analysis/pretraining.csv",
    #           {"sweep": {"$in": ["8qp1jtu0", "1iubvmc2"]}})

    # save_runs("esbenkran/thought_classification", "data/analysis/transfer_learning.csv",
    #           {"sweep": {"$in": ["g3zov98e", "sb135301"]}})

    save_run("esbenkran/thought_classification/18sb2xx0",
             "data/analysis/dense.csv")
    save_run("esbenkran/thought_classification/2x3e0v0x",
             "data/analysis/lstm.csv")
    save_run("esbenkran/thought_classification/24orv839",
             "data/analysis/lstm-3.csv")
