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
