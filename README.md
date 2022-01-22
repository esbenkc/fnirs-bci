# 🧠 fNIRS BCI
![badge](https://img.shields.io/badge/thesis-work-informational) 
![badge](https://img.shields.io/badge/yes-reproducibility-brightgreen)
![badge](https://img.shields.io/badge/ready-status-yellow)

Assessing the benefit of pre-training a thought classification model using neural prediction with an LSTM. [Read the paper](Lights%20in%20the%20Brain.pdf).

I perform self-supervised training (LeCun & Misra, 2021) to pre-train a machine learning model using the LSTM architecture (Hochreiter & Schmidhuber, 1997) on functional near-infrared spectroscopic (fNIRS) neuroimaging data (Naseer & Hong, 2015) from the NIRx NIRSport2 system (NIRx, 2021) and transfer and fine-tune it for a BCI thought classification task (Yoo et al., 2018) as is done with language models (C. Sun et al., 2019). As far as I am aware, this is the first example of such work.

Accompanies a [YouTube series](https://www.youtube.com/channel/UCvgUdk8C-PGobbY6o6eoKkA).

### Table of contents
- [🧠 fNIRS BCI Voyage](#-fnirs-bci-voyage)
    - [Table of contents](#table-of-contents)
  - [Results](#results)
  - [Reproduce this work](#reproduce-this-work)
  - [Structure](#structure)
  - [Meta](#meta)
  - [About Esben Kran](#about-esben-kran)

## Results

The 1-layer LSTM, 3-layer LSTM and dense pre-trained models were trained to predict the brain activity in channel 1 4.2 seconds in the future given 10 seconds of data. These were highly succesful and LSTMs performed much better than the fully-connected and the designed baseline (fig. 3).

The model weights were transferred and the last layer was replaced with a 256 dense layer and a sigmoid binary classifier. These underfit horribly but the pre-training avoided extreme overfitting (fig. 4).

|Figure 3 |Figure 4 |
|-|-|
|![figure 3](media/figures/figure%203.png) A) Predictions based on continuous real data for the three pre-trained models, B) Performance of the different models and the last-value baseline on all validation data, C) the three models’ mean absolute error on the test dataset (notice that the LSTMs learn more than the Dense model)|![figure 4](media/figures/figure%204.png) A) Validation loss at the last (250th) subtracted by the first step’s validation loss at different amounts of layers transferred with the LSTM-3 model. Positive numbers mean the model has become worse. B) The performance on the training and test set for the different models either pre-trained or not|

## Reproduce this work
![badge](https://img.shields.io/badge/7\/10-ease%20of%20use-informational)

| File name                                         | Description                                                                                                                                                       |
|---------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`exp_train_st_all.py`](code/exp_train_st_all.py) | 👩‍🔬 Trains the neural prediction models with basis in a configuration `dictionary` in the script. To run this, you need to **connect [W&B](https://wandb.ai)**. |
| [`exp_bci_task.py`](code/exp_bci_task.py)                               | 👩‍🔬 Trains the classification models. Also has configurations and need a **login to [W&B](https://wandb.ai)**. **Run [`generate_augmented_datasets.py](code/generate_augmented_datasets.py)** to generate augmented datasets in [`data/datasets`](data/datasets) before running this.          |
| [`experiment_bci.py`](code/experiment_bci.py)                           | 👩‍🔬 Code for running the terminal experimental paradigm. Starts an [LSL stream](https://labstreaminglayer.readthedocs.io/info/intro.html) that logs triggers in the [`.snirf`](data/snirf) fNIRS output files.          |
| [`helper_functions.py`](code/helper_functions.py)                       | 👩‍💻 An extensive selection of helper functions generally referred to by `.py` code in this directory.          |
| [`generate_augmented_datasets.py`](code/generate_augmented_datasets.py) | 👩‍💻 Generates `.npy` train/test datasets with/without augmentation to use for [`exp_bci_task.py`](code/exp_bci_task.py).           |
| [`3_data_figure.py`](code/3_data_figure.py)                             | 📊 Generates prediction data for figure 3A.           |
| [`4_brain_plot.py`](code/4_brain_plot.py)                               | 📊 Generates contrast brain plot in figure 4C.           |
| [`data_wandb.py`](code/data_wandb.py)                                   | 📊 Collects data from [W&B](https://wandb.ai) using their [api](https://docs.wandb.ai/guides/track/public-api-guide). Also requires you to **login**.           |
| [`figures.Rmd`](code/figures.Rmd)                                       | 📊 Generates figures from the data collected from the above scripts. Each figure can be run isolated in their own code chunk and outputs to [`media/figures`](media/figures).           |
| [`analysis.Rmd`](code/analysis.Rmd)                                     | ✍ Simple analyses and unstructured code.           |
| [`pipeline_math.Rmd`](code/pipeline_math.Rmd)                           | ✍ Goes through an unstructured explanation of the math implemented in R.           |

## Structure
- [Data](/data)
  - [Analysis](/data/analysis): Datasets used in [`analysis.rmd`](code/analysis.Rmd), [`figures.rmd`](code/figures.Rmd) and [`pipeline_math.rmd`](code/pipeline_math.Rmd)
  - [Datasets](/data/datasets): Generated train/test datasets as `.npy` from [`generate_augmented_datasets.py`](code/generate_augmented_datasets.py) (excluded by `.gitignore`)
  - [Snirf](/data/snirf): The raw fNIRS data files (`.snirf`)
  - [Visualization](data/visualization): Two datasets exclusively used for visualization in [`figures.rmd`](code/figures.Rmd)
  - [Weights](data/weights): Randomly initialized model weights to replace layer weights when loading pre-trained models
- [Media](media): Misc. image ouputs and showcases
  - [Figures](media/figures/): All figures and subfigures used in the paper, editable with [`figures.xd`](media/figures/figures.xd)
- [Models](models): Contains the pre-trained models used in the paper's transfer learning part
