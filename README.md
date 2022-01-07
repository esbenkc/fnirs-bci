# 🧠 fNIRS BCI Voyage
![badge](https://img.shields.io/badge/thesis-work-informational) 
![badge](https://img.shields.io/badge/reproducible-reproducibility-brightgreen)
![badge](https://img.shields.io/badge/in%20progress-status-yellow)

Assessing the benefit of pre-training with an LSTM. Specifically, I pre-train a machine learning model self-supervised (LeCun & Misra, 2021) using the LSTM architecture (Hochreiter & Schmidhuber, 1997) on functional near-infrared spectroscopic (fNIRS) neuroimaging data (Naseer & Hong, 2015) from the NIRx NIRSport2 system (NIRx, 2021) and transfer and fine-tune it for a BCI thought classification task (Yoo et al., 2018) as is done with language models (C. Sun et al., 2019). As far as I know, this is the first example of such work.

Accompanies a [YouTube series](https://www.youtube.com/channel/UCvgUdk8C-PGobbY6o6eoKkA).
## Reproduce this work
![badge](https://img.shields.io/badge/reproducible-status-brightgreen)

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

- [Data](/data)
  - [Analysis](/data/analysis): Datasets used in [`analysis.rmd`](code/analysis.Rmd), [`figures.rmd`](code/figures.Rmd) and [`pipeline_math.rmd`](code/pipeline_math.Rmd)
  - [Datasets](/data/datasets): Generated train/test datasets as `.npy` from [`generate_augmented_datasets.py`](code/generate_augmented_datasets.py) (excluded by `.gitignore`)
  - [Snirf](/data/snirf): The raw fNIRS data files (`.snirf`)
  - [Visualization](data/visualization): Two datasets exclusively used for visualization in [`figures.rmd`](code/figures.Rmd)
  - [Weights](data/weights): Randomly initialized model weights to replace layer weights when loading pre-trained models
- [Media](media): Misc. image ouputs and showcases
  - [Figures](media/figures/): All figures and subfigures used in the paper, editable with [`figures.xd`](media/figures/figures.xd)
- [Models](models): Contains the pre-trained models used in the paper's transfer learning part

## Meta
- 📜 Is part of the thesis project for [Esben Kran](https://kran.ai)
- 🕸 Scientific interest is to alleviate problems of signal instability, low signal/noise ratio and long initial adaptation time in brain-computer interfaces
  - Basic improvements in BCI thought classification accuracy can lead to downstream improvements in all parts of the BCI pipeline for clinical and retail use cases
- 🔨 The practical interest is to learn how to build a non-invasive brain-computer interface (BCI) using the [NIRSport2](https://nirx.net/nirsport) with deep learning 


## About Esben Kran
See [my website](https://kran.ai). I study Cognitive Science at Aarhus University in Denmark and lead NeuroTechX Denmark, work with fNIRS in general, am interested in scientific ethics and love entrepreneurship. I teach the course Cognition and Communication.

