import mne
import mne_nirs
import pandas as pd
import numpy as np
import time

# raw_snirf = mne.io.read_raw_snirf("data/21-10-13_bci_task_1.snirf")
# print(raw_snirf.info)

# od = mne.preprocessing.nirs.optical_density(raw_snirf)
# haemo = mne.preprocessing.nirs.beer_lambert_law(od, ppf=6)
# short_chs = mne_nirs.channels.get_short_channels(haemo)
# haemo = mne_nirs.channels.get_long_channels(haemo)

# df = haemo.to_data_frame()

# print(raw_snirf.to_data_frame())
# print(od.to_data_frame())
# print(haemo.to_data_frame())


raw = mne.io.read_raw_snirf("data/snirf/bci_task_3_arithmetic_rotation.snirf")
raw = mne.io.read_raw_snirf("data/snirf/bci_task_2_arithmetic_audiobook.snirf")
events, event_dict = mne.events_from_annotations(
    raw, verbose=False)
epochs = mne.Epochs(raw, events=events, event_id=event_dict,
                    tmin=0.0, tmax=10.0, baseline=(0, 0.5),
                    preload=True,
                    verbose=False)

print(epochs)
print(events)
print(event_dict)
print(epochs[['1', '2']].to_data_frame())
