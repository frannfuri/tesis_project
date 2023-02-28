import os
import mne
import scipy.stats
import pandas as pd
#num_windows_of_each_record = [35, 35, 34, 34, 34, 35, 34, 34, 34, 34] # w_len NN_av 10

from features.lzc_comp import KC_complexity_lempelziv_count
from features.features_comp import sliding_windowing
from features.binarization_methods import *
from collections import OrderedDict
from utils import MAD, mean_power
from philistine.mne import savgol_iaf_without_plot

def compute_features_on_windows(root, file, w_len, overlap, band, target_name,file_target_score):
    band_windows_metrics = list()
    if band == 'plete':
        band = 'Complete'
    new_raw = mne.io.read_raw_eeglab(os.path.join(root, file), preload=True)
    sliding_windows = sliding_windowing(new_raw._data[8, :], w_len=w_len, overlap=overlap, fs=new_raw.info['sfreq'])
    sliding_windows_P8 = sliding_windowing(new_raw._data[9, :], w_len=w_len, overlap=overlap, fs=new_raw.info['sfreq'])
    sliding_windows_P7 = sliding_windowing(new_raw._data[10, :], w_len=w_len, overlap=overlap, fs=new_raw.info['sfreq'])
    sliding_windows_Pz = sliding_windowing(new_raw._data[11, :], w_len=w_len, overlap=overlap, fs=new_raw.info['sfreq'])
    sliding_windows_O1 = sliding_windowing(new_raw._data[15, :], w_len=w_len, overlap=overlap, fs=new_raw.info['sfreq'])
    sliding_windows_O2 = sliding_windowing(new_raw._data[16, :], w_len=w_len, overlap=overlap, fs=new_raw.info['sfreq'])
    print('Band in process: {}'.format(band))
    for w_i in range(sliding_windows.shape[0]):
        if band == 'Complete':
            window_metrics = OrderedDict()
            # Subject name
            window_metrics['SUBJ'] = file[:5]
            # LZC METRICS
            #       Median
            sequence = numpy_to_str_sequence(median_bin(sliding_windows[w_i, :]))
            window_metrics['LZC_median_' + str(band)] = KC_complexity_lempelziv_count(sequence)
            #       Mean
            sequence = numpy_to_str_sequence(mean_bin(sliding_windows[w_i, :]))
            window_metrics['LZC_mean_' + str(band)] = KC_complexity_lempelziv_count(sequence)
            #       Hilbert envelop
            sequence = numpy_to_str_sequence(hilbert_envelop_bin(sliding_windows[w_i, :]))
            window_metrics['LZC_H-env_' + str(band)] = KC_complexity_lempelziv_count(sequence)
            #       Hilbert power
            sequence = numpy_to_str_sequence(hilbert_power_bin(sliding_windows[w_i, :]))
            window_metrics['LZC_H-pow_' + str(band)] = KC_complexity_lempelziv_count(sequence)
            #       Slope sign
            sequence = numpy_to_str_sequence(slope_sign_bin(sliding_windows[w_i, :]))
            window_metrics['LZC_slope_' + str(band)] = KC_complexity_lempelziv_count(sequence)
            #       Slope Hilbert env
            sequence = numpy_to_str_sequence(slope_hilb_envelop_bin(sliding_windows[w_i, :]))
            window_metrics['LZC_slope_H-env_' + str(band)] = KC_complexity_lempelziv_count(sequence)
            #       Slope Hilbert pow
            sequence = numpy_to_str_sequence(slope_hilb_power_bin(sliding_windows[w_i, :]))
            window_metrics['LZC_slope_H-pow_' + str(band)] = KC_complexity_lempelziv_count(sequence)
            window_metrics[target_name] = file_target_score
            band_windows_metrics.append(window_metrics)
        elif band == 'Gamma':
            window_metrics = OrderedDict()
            window_metrics['SUBJ'] = file[:5]
            # LZC METRICS
            #       Hilbert envelop
            sequence = numpy_to_str_sequence(hilbert_envelop_bin(sliding_windows[w_i, :]))
            window_metrics['LZC_H-env_' + str(band)] = KC_complexity_lempelziv_count(sequence)
            #       Hilbert power
            sequence = numpy_to_str_sequence(hilbert_power_bin(sliding_windows[w_i, :]))
            window_metrics['LZC_H-pow_' + str(band)] = KC_complexity_lempelziv_count(sequence)
            #       Slope sign
            sequence = numpy_to_str_sequence(slope_sign_bin(sliding_windows[w_i, :]))
            window_metrics['LZC_slope_' + str(band)] = KC_complexity_lempelziv_count(sequence)
            #       Slope Hilbert env
            sequence = numpy_to_str_sequence(slope_hilb_envelop_bin(sliding_windows[w_i, :]))
            window_metrics['LZC_slope_H-env_' + str(band)] = KC_complexity_lempelziv_count(sequence)
            #       Slope Hilbert pow
            sequence = numpy_to_str_sequence(slope_hilb_power_bin(sliding_windows[w_i, :]))
            window_metrics['LZC_slope_H-pow_' + str(band)] = KC_complexity_lempelziv_count(sequence)
            window_metrics[target_name] = file_target_score
            band_windows_metrics.append(window_metrics)
        else:
            window_metrics = OrderedDict()
            window_metrics['SUBJ'] = file[:5]
            # LZC METRICS
            #       Median
            sequence = numpy_to_str_sequence(median_bin(sliding_windows[w_i, :]))
            window_metrics['LZC_median_' + str(band)] = KC_complexity_lempelziv_count(sequence)
            #       Mean
            sequence = numpy_to_str_sequence(mean_bin(sliding_windows[w_i, :]))
            window_metrics['LZC_mean_' + str(band)] = KC_complexity_lempelziv_count(sequence)
            #       Hilbert envelop
            sequence = numpy_to_str_sequence(hilbert_envelop_bin(sliding_windows[w_i, :]))
            window_metrics['LZC_H-env_' + str(band)] = KC_complexity_lempelziv_count(sequence)
            #       Hilbert power
            sequence = numpy_to_str_sequence(hilbert_power_bin(sliding_windows[w_i, :]))
            window_metrics['LZC_H-pow_' + str(band)] = KC_complexity_lempelziv_count(sequence)
            #       Slope sign
            sequence = numpy_to_str_sequence(slope_sign_bin(sliding_windows[w_i, :]))
            window_metrics['LZC_slope_' + str(band)] = KC_complexity_lempelziv_count(sequence)
            #       Slope Hilbert env
            sequence = numpy_to_str_sequence(slope_hilb_envelop_bin(sliding_windows[w_i, :]))
            window_metrics['LZC_slope_H-env_' + str(band)] = KC_complexity_lempelziv_count(sequence)
            #       Slope Hilbert pow
            sequence = numpy_to_str_sequence(slope_hilb_power_bin(sliding_windows[w_i, :]))
            window_metrics['LZC_slope_H-pow_' + str(band)] = KC_complexity_lempelziv_count(sequence)
            # OTHERS
            window_metrics['mean_' + str(band)] = np.mean(sliding_windows[w_i, :])
            window_metrics['median_' + str(band)] = np.median(sliding_windows[w_i, :])
            window_metrics['std_' + str(band)] = np.std(sliding_windows[w_i, :])
            window_metrics['kurt_' + str(band)] = scipy.stats.kurtosis(sliding_windows[w_i, :])
            window_metrics['skew_' + str(band)] = scipy.stats.skew(sliding_windows[w_i, :])
            window_metrics['IQR_' + str(band)] = scipy.stats.iqr(sliding_windows[w_i, :])
            window_metrics['Mean_power_' + str(band)] = mean_power(sliding_windows[w_i, :])
            window_metrics['MAD_' + str(band)] = 1.4826 * MAD(sliding_windows[w_i, :])
            if band == 'Alpha' and Alpha_features==True:
                #post_indices = list(range(9, 12)) + list(range(15, 17))  # P8, P7, Pz, O1, O2
                post_chns_window = np.concatenate((np.expand_dims(sliding_windows_P8[w_i,:],0), np.expand_dims(sliding_windows_P7[w_i,:],0), np.expand_dims(sliding_windows_Pz[w_i,:],0),
                                                   np.expand_dims(sliding_windows_O1[w_i,:],0), np.expand_dims(sliding_windows_O2[w_i,:],0)), 0)
                post_chns_window_raw = mne.io.RawArray(post_chns_window, info=mne.create_info([new_raw.ch_names[9], new_raw.ch_names[10],
                                                                                               new_raw.ch_names[11], new_raw.ch_names[15],
                                                                                               new_raw.ch_names[16]], sfreq=new_raw.info['sfreq'], ch_types='eeg'))
                iaf_est = savgol_iaf_without_plot(post_chns_window_raw, picks=list(range(0,5)))
                window_metrics['IAF_post_chns'] = iaf_est.PeakAlphaFrequency
                window_metrics['CoG_post_chns'] = iaf_est.CenterOfGravity
                window_metrics['AlphaBand_low_bound'] = iaf_est.AlphaBand[0]
                window_metrics['AlphaBand_high_bound'] = iaf_est.AlphaBand[1]
            window_metrics[target_name] = file_target_score
            band_windows_metrics.append(window_metrics)
    n = sliding_windows.shape[0]
    return band_windows_metrics, n

if __name__ == '__main__':
    ##### PARAMETERS #####
    directory = '../datasets_per_band/train_subjs'
    fromat_type = 'set'
    target_name = 'real_and_pred_SAPS'
    w_len = 40
    overlap = 0.75
    Alpha_features = True
    subjs = ['SA007', 'SA010', 'SA014', 'SA025', 'SA026', 'SA039', 'SA047']
    ######################
    # Load labels
    all_targets_info = OrderedDict()
    for subj_ in subjs:
        target_info = pd.read_csv('../BENDR_datasets/labels/{}_labels.csv'.format(subj_),
                                  index_col=0, decimal=',')
        target_info = target_info.to_dict()
        target_info = target_info[target_name]
        all_targets_info.update(target_info)
    n_w = 0
    all_windows_metrics_condensed = list()
    n_windows_per_record = []
    record_names = []
    for root, folders, _ in os.walk(directory):
        #i = 0
        f_ = 0
        n_folders = len(folders)
        for fold_day in sorted(folders):
            print('==============================Processing record of ' + str(fold_day[:11]) + '==============================')
            #i += 1
            f_ += 1
            file_label_ = all_targets_info[fold_day[:11]]
            all_windows_all_bands_metrics = list()  # len of 5 (Alpha(19),  Complete(7), Delta(15), Theta(15), Gamma(5))
            #############################
            ############################

            for root_day, _, files in os.walk(os.path.join(root, fold_day)):
                for file in sorted(files):
                    if file.endswith('set'):
                        one_band_windows_metrics, n_windows = compute_features_on_windows(root_day, file, w_len, overlap, file[-9:-4],
                                                                                          target_name, file_label_)
                        all_windows_all_bands_metrics.append(one_band_windows_metrics)
                        print('Band {} --> ready.'.format(file[-9:-4]))
            print('{} windows generated for {}.'.format(n_windows, fold_day[:11]))
            for j in range(n_windows):
                all_windows_all_bands_metrics[0][j].update(all_windows_all_bands_metrics[1][j])
                all_windows_all_bands_metrics[0][j].update(all_windows_all_bands_metrics[2][j])
                all_windows_all_bands_metrics[0][j].update(all_windows_all_bands_metrics[3][j])
                all_windows_all_bands_metrics[0][j].update(all_windows_all_bands_metrics[4][j])
            for win_m in all_windows_all_bands_metrics[0]:
                all_windows_metrics_condensed.append(win_m)
            n_w += n_windows
            n_windows_per_record.append(n_windows)
            record_names.append(fold_day)
            print('Processed {} folders from a total of {} folders.'.format(f_, n_folders))


    df_all_windows_metrics = pd.DataFrame(all_windows_metrics_condensed)
    print('A total of {} windows have been obtained.'.format(n_w))
    df_all_windows_metrics.to_csv(
        './{}_features_{}_{}.csv'.format(directory.split('/')[-1][:5], directory.split('/')[-1][6:],
                                                  str(w_len) + 'sec'))
    with open('./{}_n_windows_{}_{}.txt'.format(directory.split('/')[-1][:5], directory.split('/')[-1][6:],
                                                  str(w_len) + 'sec'), 'w') as f:
        for i_ in range(len(n_windows_per_record)):
            f.write('{},{}\n'.format(record_names[i_], n_windows_per_record[i_]))
    f.close()
