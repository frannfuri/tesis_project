import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mne
from collections import OrderedDict
from philistine.mne import savgol_iaf_without_plot

def IAF_computation(path_, panss, panss_posit):
    iaf_metrics = OrderedDict()
    new_raw = mne.io.read_raw_eeglab(path_, preload=True)
    new_raw_P8 = new_raw._data[9, :]
    new_raw_P7 = new_raw._data[10, :]
    new_raw_Pz = new_raw._data[11, :]
    new_raw_O1 = new_raw._data[15, :]
    new_raw_O2 = new_raw._data[16, :]
    post_chns_window = np.concatenate((np.expand_dims(new_raw_P8, 0),
                                       np.expand_dims(new_raw_P7, 0),
                                       np.expand_dims(new_raw_Pz, 0),
                                       np.expand_dims(new_raw_O1, 0),
                                       np.expand_dims(new_raw_O2, 0)), 0)
    post_chns_window_raw = mne.io.RawArray(post_chns_window, info=mne.create_info([new_raw.ch_names[9],
                                    new_raw.ch_names[10], new_raw.ch_names[11], new_raw.ch_names[15],
                                    new_raw.ch_names[16]], sfreq=new_raw.info['sfreq'], ch_types='eeg'))
    iaf_est = savgol_iaf_without_plot(post_chns_window_raw, picks=list(range(0, 5)))
    iaf_metrics['IAF_post_chns'] = iaf_est.PeakAlphaFrequency
    iaf_metrics['AlphaBand_low_bound'] = iaf_est.AlphaBand[0]
    iaf_metrics['AlphaBand_high_bound'] = iaf_est.AlphaBand[1]
    iaf_metrics['PANSS'] = panss
    iaf_metrics['PANSS_posit'] = panss_posit
    return iaf_metrics

SA007_files = []
SA010_files = []
SA014_files = []
SA017_files = []
SA047_files = []
file_target_score = {'SA010_day1_':85, 'SA010_day2_':94, 'SA010_day3_':100, 'SA010_day5_':84, 'SA010_day6_':83,
                     'SA010_day7_':77, 'SA010_day9_':74, 'SA010_day11':81, 'SA010_day12': 74,'SA010_day13':71,
                     'SA047_day1_': 96, 'SA047_day2_': 93, 'SA047_day3_': 84, 'SA047_day4_': 82, 'SA047_day5_': 85,
                     'SA047_day6_': 93, 'SA047_day7_': 98, 'SA047_day9_': 91, 'SA047_day11': 98, 'SA047_day12': 88,
                     'SA047_day13': 95, 'SA014_day1_':79, 'SA014_day2_':83, 'SA014_day5_':71, 'SA014_day6_':73,
                     'SA007_day1_':82, 'SA007_day2_': 78, 'SA007_day3_': 82, 'SA007_day4_': 77, 'SA007_day5_': 72,
                     'SA007_day6_': 78, 'SA007_day7_': 75, 'SA007_day8_': 78, 'SA017_day1_':89, 'SA017_day4_':74, 'SA017_day7_':75}
file_target_score2 = {'SA010_day1_':33, 'SA010_day2_':35, 'SA010_day3_':36, 'SA010_day5_':35, 'SA010_day6_':35,
                      'SA010_day7_':33, 'SA010_day9_':30, 'SA010_day11':30, 'SA010_day12':28, 'SA010_day13':27, 'SA047_day1_': 26,
                      'SA047_day2_': 21, 'SA047_day3_': 21, 'SA047_day4_': 20, 'SA047_day5_': 25, 'SA047_day6_': 24,
                      'SA047_day7_': 27, 'SA047_day9_': 25, 'SA047_day11': 26, 'SA047_day12': 24, 'SA047_day13': 26,
                      'SA014_day1_':28, 'SA014_day2_':29, 'SA014_day5_':28, 'SA014_day6_':30, 'SA007_day1_':28,
                      'SA007_day2_':25, 'SA007_day3_':22, 'SA007_day4_':22, 'SA007_day5_':21, 'SA007_day6_':21, 'SA007_day7_':25,
                      'SA007_day8_':23, 'SA017_day1_': 26, 'SA017_day4_': 26, 'SA017_day7_': 26}
for root, folders, _ in os.walk('./datasets/whole_data'):
    i = 0
    for fold_day in sorted(folders):
        for root_day, _, files in os.walk(os.path.join(root, fold_day)):
            for file in sorted(files):
                if file.endswith('Alpha.set'):
                    if file.startswith('SA007'):
                        SA007_files.append(os.path.join(root_day,file))
                    if file.startswith('SA010'):
                        SA010_files.append(os.path.join(root_day,file))
                    if file.startswith('SA014'):
                        SA014_files.append(os.path.join(root_day,file))
                    if file.startswith('SA017'):
                        SA017_files.append(os.path.join(root_day,file))
                    if file.startswith('SA047'):
                        SA047_files.append(os.path.join(root_day,file))
SA007_df_list = list()
for rec in SA007_files:
    rec_panss_score = file_target_score[rec.split('/')[-1][:11]]
    rec_panss_posit_score = file_target_score2[rec.split('/')[-1][:11]]
    rec_data = IAF_computation(rec, rec_panss_score, rec_panss_posit_score)
    SA007_df_list.append(rec_data)
df_all_windows_metrics = pd.DataFrame(SA007_df_list)
df_all_windows_metrics.to_csv('./SA007_iaf_metrics.csv')

SA010_df_list = list()
for rec in SA010_files:
    rec_panss_score = file_target_score[rec.split('/')[-1][:11]]
    rec_panss_posit_score = file_target_score2[rec.split('/')[-1][:11]]
    rec_data = IAF_computation(rec, rec_panss_score, rec_panss_posit_score)
    SA010_df_list.append(rec_data)
df_all_windows_metrics = pd.DataFrame(SA010_df_list)
df_all_windows_metrics.to_csv('./SA010_iaf_metrics.csv')

SA014_df_list = list()
for rec in SA014_files:
    rec_panss_score = file_target_score[rec.split('/')[-1][:11]]
    rec_panss_posit_score = file_target_score2[rec.split('/')[-1][:11]]
    rec_data = IAF_computation(rec, rec_panss_score, rec_panss_posit_score)
    SA014_df_list.append(rec_data)
df_all_windows_metrics = pd.DataFrame(SA014_df_list)
df_all_windows_metrics.to_csv('./SA014_iaf_metrics.csv')

SA017_df_list = list()
for rec in SA017_files:
    rec_panss_score = file_target_score[rec.split('/')[-1][:11]]
    rec_panss_posit_score = file_target_score2[rec.split('/')[-1][:11]]
    rec_data = IAF_computation(rec, rec_panss_score, rec_panss_posit_score)
    SA017_df_list.append(rec_data)
df_all_windows_metrics = pd.DataFrame(SA017_df_list)
df_all_windows_metrics.to_csv('./SA017_iaf_metrics.csv')

SA047_df_list = list()
for rec in SA047_files:
    rec_panss_score = file_target_score[rec.split('/')[-1][:11]]
    rec_panss_posit_score = file_target_score2[rec.split('/')[-1][:11]]
    rec_data = IAF_computation(rec, rec_panss_score, rec_panss_posit_score)
    SA047_df_list.append(rec_data)
df_all_windows_metrics = pd.DataFrame(SA047_df_list)
df_all_windows_metrics.to_csv('./SA047_iaf_metrics.csv')
