
import os
import numpy as np
import matplotlib.pyplot as plt

import mne
import mne_nirs

from mne_nirs.experimental_design import make_first_level_design_matrix
from mne_nirs.statistics import run_glm
from mne_nirs.channels import (get_long_channels,
                               get_short_channels,
                               picks_pair_to_idx)

from nilearn.plotting import plot_design_matrix

fnirs_data_folder = "C:\\Users\\esben\\Desktop\\study\\5th semester\\fnirs-bci\\data\\snirf\\bci_task_1_arms_talk.snirf"
raw_intensity = mne.io.read_raw_snirf(fnirs_data_folder, preload=True)

raw_intensity.annotations.rename({'1': 'Task 1',
                                  '2': 'Task 2',
                                  '0': 'None'})
raw_intensity.annotations.set_durations(10)

raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=6)

short_chs = get_short_channels(raw_haemo)
raw_haemo = get_long_channels(raw_haemo)

events, event_dict = mne.events_from_annotations(raw_haemo, verbose=False)
mne.viz.plot_events(events, event_id=event_dict, sfreq=raw_haemo.info['sfreq'])

design_matrix = make_first_level_design_matrix(raw_haemo,
                                               drift_model='cosine',
                                               high_pass=0.005,  # Must be specified per experiment
                                               hrf_model='spm',
                                               stim_dur=10.0)

design_matrix["ShortHbO"] = np.mean(short_chs.copy().pick(
                                    picks="hbo").get_data(), axis=0)

design_matrix["ShortHbR"] = np.mean(short_chs.copy().pick(
                                    picks="hbr").get_data(), axis=0)

fig, ax1 = plt.subplots(figsize=(10, 6), nrows=1, ncols=1)
fig = plot_design_matrix(design_matrix, ax=ax1)

glm_est = run_glm(raw_haemo, design_matrix)
glm_est.plot_topo(conditions=['Task 1', 'Task 2']).savefig(
    "C:\\Users\\esben\\Desktop\\study\\5th semester\\fnirs-bci\\media\\figures\\glm_hbo_con.png")

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(
    10, 6), gridspec_kw=dict(width_ratios=[0.92, 1]))

glm_hbo = glm_est.copy().pick(picks="hbo")
conditions = ['Task 1']

glm_hbo.plot_topo(axes=axes[0], colorbar=False, conditions=conditions)

glm_hbo.copy().pick(picks=range(10)).plot_topo(
    conditions=conditions, axes=axes[1], colorbar=False, vmin=-16, vmax=16)
glm_hbo.copy().pick(picks=range(10, 20)).plot_topo(
    conditions=conditions, axes=axes[1], colorbar=False, vmin=-16, vmax=16)

axes[0].set_title("Smoothed across hemispheres")
axes[1].set_title("Hemispheres plotted independently")

fig.savefig(
    "C:\\Users\\esben\\Desktop\\study\\5th semester\\fnirs-bci\\media\\figures\\glm_hbo.png")

contrast_matrix = np.eye(design_matrix.shape[1])
basic_conts = dict([(column, contrast_matrix[i])
                   for i, column in enumerate(design_matrix.columns)])
contrast_LvR = basic_conts['Task 1'] - basic_conts['Task 2']

contrast = glm_est.compute_contrast(contrast_LvR)
contrast.plot_topo().savefig(
    "C:\\Users\\esben\\Desktop\\study\\5th semester\\fnirs-bci\\media\\figures\\3_con_arms_talk.png")
