import numpy as np
import pandas as pd
import sys
import os
from os.path import join as ospj
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from DSOSD.model import NDD
from DSOSD.DSOSD_utils import load_config, get_iEEG_data, detect_bad_channels, remove_scalp_electrodes, preprocess_for_detection
# Load in config variables
usr,passpath,datapath,prodatapath,metapath,figpath = load_config(ospj('/mnt/leif/littlab/users/wojemann/NDD/code','config.json'),flag=None)
seizure_list = pd.read_csv(ospj(metapath,'example_seizure_sheet.csv'))
sz_annotations = {key: [] for key in ["patient","fname","onset","offset","onset_chs","spread_chs","spread_times"]}
for _, sz_row in seizure_list.iterrows():
    ## Initialize folder for saving down stream results
    os.makedirs(ospj(prodatapath,sz_row.patient),exist_ok=True)


    ## Loading the seizure
    # how much buffer before and after seiuzre onset to analyze (recommend 120 s and 120 s)
    onset_buffer, offset_buffer = 120,120
    # converting times for iEEG.org API
    start_time_us = (sz_row.onset - onset_buffer)*1e6
    stop_time_us = (sz_row.offset + offset_buffer)*1e6
    # pulling seizure from iEEG.org API
    print(f"Pulling seizure from {sz_row.patient}: {sz_row.fname} from {sz_row.onset} to {sz_row.offset}")
    sz_df, fs_raw = get_iEEG_data('wojemann',passpath,sz_row.fname,
                              start_time_us,stop_time_us,
                              force_pull=True)


    ## Preprocessing the seizure
    # remove scalp electrodes
    sz_df = sz_df.loc[:,remove_scalp_electrodes(sz_df.columns)]
    # find noisy channels based on preictal data
    bad_ch_mask,info = detect_bad_channels(sz_df.loc[:fs_raw*onset_buffer/2,:].to_numpy(),fs_raw)
    sz_df = sz_df.loc[:,bad_ch_mask]
    # preprocess seizure for detection algorithm
    sz_prep,fs = preprocess_for_detection(sz_df,fs_raw,'bipolar',128,pre_mask=[])
    

    ## Seizure detection
    # Initializing neural dynamic divergence model
    model = NDD(use_cuda=True,num_epochs=100,fs=fs)
    # fitting model to preictal window (-120 s to -60 seconds from seizure onset)
    model.fit(sz_prep.loc[:fs*onset_buffer/2,:])
    model.generate_loss(sz_prep)
    model.predict_gamma(confidence_threshold=.90,
                        pre_buffer=onset_buffer,
                        post_buffer=offset_buffer)
    print("Onset channels are: {model.sz_spread[model.sz_spread.iloc[0,:] <=3]}")
    print(model.sz_spread)
    