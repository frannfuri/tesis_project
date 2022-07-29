import os
import mne
import scipy.stats
import numpy as np
import pandas as pd
import multiprocessing
#num_windows_of_each_record = [35, 35, 34, 34, 34, 35, 34, 34, 34, 34] # w_len NN_av 10

from features.lzc_comp import KC_complexity_lempelziv_count
from features.features_comp import sliding_windowing
from features.binarization_methods import *
from collections import OrderedDict
from utils import MAD, mean_power
from philistine.mne import savgol_iaf_without_plot



if __name__ == '__main__':
    ##### PARAMETERS #####
    directory = './datasets/whole_data'
    fromat_type = 'set'
    record_name = 'SA010_day1_'
    ######################
    i=0
    for root, _, files in os.walk(os.path.join(directory, record_name)):
        i +=1
        all_band_metrics = []
        for file_band in sorted(files):
            if file_band.endswith('set'):
                band = file_band[-9:]
                if band == 'plete':
                    band == 'Complete'
                print('==============================Processing band ' + str(band) + '==============================')
                new_raw = mne.io.read_raw_eeglab(os.path.join(root, file_band), preload=True)
                new_raw_data_Cz = new_raw._data[8,:]
                new_raw_data_P8 = new_raw._data[9, :]
                new_raw_data_P7 = new_raw._data[10, :]
                new_raw_data_Pz = new_raw._data[11, :]
                new_raw_data_O1 = new_raw._data[15, :]
                new_raw_data_O2 = new_raw._data[16, :]
                band_metrics = OrderedDict()
                band_metrics['SUBJ'] = file_band[:5]
                if band == 'Complete':
                    # Subject name
                    band_metrics['LZC_median_Complete'] = KC_complexity_lempelziv_count(numpy_to_str_sequence(median_bin(new_raw_data_Cz)))
                    band_metrics['LZC_mean_Complete'] = KC_complexity_lempelziv_count(numpy_to_str_sequence(mean_bin(new_raw_data_Cz)))
                    band_metrics['LZC_H-env_Complete'] = KC_complexity_lempelziv_count(numpy_to_str_sequence(hilbert_envelop_bin(new_raw_data_Cz)))
                    band_metrics['LZC_H-pow_Complete'] = KC_complexity_lempelziv_count(numpy_to_str_sequence(hilbert_power_bin(new_raw_data_Cz)))
                    band_metrics['LZC_slope_Complete'] = KC_complexity_lempelziv_count(numpy_to_str_sequence(slope_sign_bin(new_raw_data_Cz)))
                    band_metrics['LZC_slope_H-env_Complete'] = KC_complexity_lempelziv_count(numpy_to_str_sequence(slope_hilb_envelop_bin(new_raw_data_Cz)))
                    band_metrics['LZC_slope_H-pow_Complete'] = KC_complexity_lempelziv_count(numpy_to_str_sequence(slope_hilb_power_bin(new_raw_data_Cz)))
                elif band == 'Gamma':
                    band_metrics['LZC_median_Gamma'] = KC_complexity_lempelziv_count(
                        numpy_to_str_sequence(median_bin(new_raw_data_Cz)))
                    band_metrics['LZC_mean_Gamma'] = KC_complexity_lempelziv_count(
                        numpy_to_str_sequence(mean_bin(new_raw_data_Cz)))
                    band_metrics['LZC_H-env_Gamma'] = KC_complexity_lempelziv_count(
                        numpy_to_str_sequence(hilbert_envelop_bin(new_raw_data_Cz)))
                    band_metrics['LZC_H-pow_Gamma'] = KC_complexity_lempelziv_count(
                        numpy_to_str_sequence(hilbert_power_bin(new_raw_data_Cz)))
                    band_metrics['LZC_slope_Gamma'] = KC_complexity_lempelziv_count(
                        numpy_to_str_sequence(slope_sign_bin(new_raw_data_Cz)))
                    band_metrics['LZC_slope_H-env_Gamma'] = KC_complexity_lempelziv_count(
                        numpy_to_str_sequence(slope_hilb_envelop_bin(new_raw_data_Cz)))
                    band_metrics['LZC_slope_H-pow_Gamma'] = KC_complexity_lempelziv_count(
                        numpy_to_str_sequence(slope_hilb_power_bin(new_raw_data_Cz)))
                else:
                    band_metrics['LZC_median_'+str(band)] = KC_complexity_lempelziv_count(
                        numpy_to_str_sequence(median_bin(new_raw_data_Cz)))
                    band_metrics['LZC_mean_'+str(band)] = KC_complexity_lempelziv_count(
                        numpy_to_str_sequence(mean_bin(new_raw_data_Cz)))
                    band_metrics['LZC_H-env_'+str(band)] = KC_complexity_lempelziv_count(
                        numpy_to_str_sequence(hilbert_envelop_bin(new_raw_data_Cz)))
                    band_metrics['LZC_H-pow_'+str(band)] = KC_complexity_lempelziv_count(
                        numpy_to_str_sequence(hilbert_power_bin(new_raw_data_Cz)))
                    band_metrics['LZC_slope_'+str(band)] = KC_complexity_lempelziv_count(
                        numpy_to_str_sequence(slope_sign_bin(new_raw_data_Cz)))
                    band_metrics['LZC_slope_H-env_'+str(band)] = KC_complexity_lempelziv_count(
                        numpy_to_str_sequence(slope_hilb_envelop_bin(new_raw_data_Cz)))
                    band_metrics['LZC_slope_H-pow_'+str(band)] = KC_complexity_lempelziv_count(
                        numpy_to_str_sequence(slope_hilb_power_bin(new_raw_data_Cz)))
                    band_metrics['mean_' + str(band)] = np.mean(new_raw_data_Cz)
                    band_metrics['median_' + str(band)] = np.median(new_raw_data_Cz)
                    band_metrics['std_' + str(band)] = np.std(new_raw_data_Cz)
                    band_metrics['kurt_' + str(band)] = scipy.stats.kurtosis(new_raw_data_Cz)
                    band_metrics['skew_' + str(band)] = scipy.stats.skew(new_raw_data_Cz)
                    band_metrics['IQR_' + str(band)] = scipy.stats.iqr(new_raw_data_Cz)
                    band_metrics['Mean_power_' + str(band)] = mean_power(new_raw_data_Cz)
                    band_metrics['MAD_' + str(band)] = 1.4826 * MAD(new_raw_data_Cz)
                    if band == 'Alpha':
                        post_chns = np.concatenate((np.expand_dims(new_raw_data_P8,0), np.expand_dims(new_raw_data_P7,0), np.expand_dims(new_raw_data_Pz,0),
                                                                              np.expand_dims(new_raw_data_O1,0), np.expand_dims(new_raw_data_O2,0)), 0)
                        post_chns_raw = mne.io.RawArray(post_chns, info=mne.create_info([new_raw.ch_names[9], new_raw.ch_names[10],
                                                                              new_raw.ch_names[11], new_raw.ch_names[15],
                                                                              new_raw.ch_names[16]], sfreq=new_raw.info['sfreq'], ch_types='eeg'))
                        iaf_est = savgol_iaf_without_plot(post_chns_raw, picks=list(range(0, 5)))
                        band_metrics['IAF_post_chns'] = iaf_est.PeakAlphaFrequency
                        band_metrics['CoG_post_chns'] = iaf_est.CenterOfGravity
                        band_metrics['AlphaBand_low_bound'] = iaf_est.AlphaBand[0]
                        band_metrics['AlphaBand_high_bound'] = iaf_est.AlphaBand[1]
            all_band_metrics.append(band_metrics)
        all_band_metrics[0].update(all_band_metrics[1])
        all_band_metrics[0].update(all_band_metrics[2])
        all_band_metrics[0].update(all_band_metrics[3])
        all_band_metrics[0].update(all_band_metrics[4])
        df_all_band_metrics = pd.DataFrame(all_band_metrics[0], index=[0])
        df_all_band_metrics.to_csv(
            './whole_record_features_{}.csv'.format(record_name))
    print(i)