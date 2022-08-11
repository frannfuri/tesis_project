import os
import mne
import scipy.stats
import pandas as pd
import multiprocessing
#num_windows_of_each_record = [35, 35, 34, 34, 34, 35, 34, 34, 34, 34] # w_len NN_av 10

from features.lzc_comp import KC_complexity_lempelziv_count
from features.features_comp import sliding_windowing, sliding_conc_samples_of_windows
from features.binarization_methods import *
from collections import OrderedDict
from utils import MAD, mean_power
from philistine.mne import savgol_iaf_without_plot

def compute_features_on_conc_samples_(root, file, w_len, conc_samples_len, conc_samples_overlap, band, target_name,
                                file_target_score, target_name2, file_target_score2, pool):
    band_conc_sample_metrics = list()
    if band == 'plete':
        band = 'Complete'
    elif band == '_Beta':
        band = 'Beta'
    new_raw = mne.io.read_raw_eeglab(os.path.join(root,file), preload=True)

    # dim -->  ( numb_conc_samples(e.g.132),  num_windows_per_conc_samples(e.g.4), len_of_each_window(e.g.10) )
    all_windows = sliding_conc_samples_of_windows(new_raw._data[8,:], w_len, conc_samples_len, conc_samples_overlap,
                                                  new_raw.info['sfreq'])
    print('Band in process: {}'.format(band))
    for conc_s_i in range(all_windows.shape[0]):
        metrics_windows_of_the_sample_one_band = list()
        for w_i in range(all_windows.shape[1]):
            metrics_of_the_window = OrderedDict()
            # Subject name
            metrics_of_the_window['SUBJ'] = file[:5]
            if band == 'Complete':
                # LZC METRICS
                a, b, c, d, e, f, g = zip(
                    pool.map(KC_complexity_lempelziv_count, [numpy_to_str_sequence(median_bin(all_windows[conc_s_i, w_i, :])),
                                                             numpy_to_str_sequence(mean_bin(all_windows[conc_s_i, w_i, :])),
                                                             numpy_to_str_sequence(
                                                                 hilbert_envelop_bin(all_windows[conc_s_i, w_i, :])),
                                                             numpy_to_str_sequence(
                                                                 hilbert_power_bin(all_windows[conc_s_i, w_i, :])),
                                                             numpy_to_str_sequence(
                                                                 slope_sign_bin(all_windows[conc_s_i, w_i, :])),
                                                             numpy_to_str_sequence(
                                                                 slope_hilb_envelop_bin(all_windows[conc_s_i, w_i, :])),
                                                             numpy_to_str_sequence(
                                                                 slope_hilb_power_bin(all_windows[conc_s_i, w_i, :]))]))
                #       Median
                metrics_of_the_window['LZC_median_' + str(band)] = a
                #       Mean
                metrics_of_the_window['LZC_mean_' + str(band)] = b
                #       Hilbert envelop
                metrics_of_the_window['LZC_H-env_' + str(band)] = c
                #       Hilbert power
                metrics_of_the_window['LZC_H-pow_' + str(band)] = d
                #       Slope sign
                metrics_of_the_window['LZC_slope_' + str(band)] = e
                #       Slope Hilbert env
                metrics_of_the_window['LZC_slope_H-env_' + str(band)] = f
                #       Slope Hilbert pow
                metrics_of_the_window['LZC_slope_H-pow_' + str(band)] = g

            elif band == 'Gamma':
                a, b, c, d, e, f, g = zip(
                    pool.map(KC_complexity_lempelziv_count, [numpy_to_str_sequence(median_bin(all_windows[conc_s_i, w_i, :])),
                                                             numpy_to_str_sequence(mean_bin(all_windows[conc_s_i, w_i, :])),
                                                             numpy_to_str_sequence(
                                                                 hilbert_envelop_bin(all_windows[conc_s_i, w_i, :])),
                                                             numpy_to_str_sequence(
                                                                 hilbert_power_bin(all_windows[conc_s_i, w_i, :])),
                                                             numpy_to_str_sequence(
                                                                 slope_sign_bin(all_windows[conc_s_i, w_i, :])),
                                                             numpy_to_str_sequence(
                                                                 slope_hilb_envelop_bin(all_windows[conc_s_i, w_i, :])),
                                                             numpy_to_str_sequence(
                                                                 slope_hilb_power_bin(all_windows[conc_s_i, w_i, :]))]))
                #       Median
                metrics_of_the_window['LZC_median_' + str(band)] = a
                #       Mean
                metrics_of_the_window['LZC_mean_' + str(band)] = b
                #       Hilbert envelop
                metrics_of_the_window['LZC_H-env_' + str(band)] = c
                #       Hilbert power
                metrics_of_the_window['LZC_H-pow_' + str(band)] = d
                #       Slope sign
                metrics_of_the_window['LZC_slope_' + str(band)] = e
                #       Slope Hilbert env
                metrics_of_the_window['LZC_slope_H-env_' + str(band)] = f
                #       Slope Hilbert pow
                metrics_of_the_window['LZC_slope_H-pow_' + str(band)] = g

            else:
                a, b, c, d, e, f, g = zip(
                    pool.map(KC_complexity_lempelziv_count, [numpy_to_str_sequence(median_bin(all_windows[conc_s_i, w_i, :])),
                                                             numpy_to_str_sequence(mean_bin(all_windows[conc_s_i, w_i, :])),
                                                             numpy_to_str_sequence(
                                                                 hilbert_envelop_bin(all_windows[conc_s_i, w_i, :])),
                                                             numpy_to_str_sequence(
                                                                 hilbert_power_bin(all_windows[conc_s_i, w_i, :])),
                                                             numpy_to_str_sequence(
                                                                 slope_sign_bin(all_windows[conc_s_i, w_i, :])),
                                                             numpy_to_str_sequence(
                                                                 slope_hilb_envelop_bin(all_windows[conc_s_i, w_i, :])),
                                                             numpy_to_str_sequence(
                                                                 slope_hilb_power_bin(all_windows[conc_s_i, w_i, :]))]))
                #       Median
                metrics_of_the_window['LZC_median_' + str(band)] = a
                #       Mean
                metrics_of_the_window['LZC_mean_' + str(band)] = b
                #       Hilbert envelop
                metrics_of_the_window['LZC_H-env_' + str(band)] = c
                #       Hilbert power
                metrics_of_the_window['LZC_H-pow_' + str(band)] = d
                #       Slope sign
                metrics_of_the_window['LZC_slope_' + str(band)] = e
                #       Slope Hilbert env
                metrics_of_the_window['LZC_slope_H-env_' + str(band)] = f
                #       Slope Hilbert pow
                metrics_of_the_window['LZC_slope_H-pow_' + str(band)] = g
                # OTHERS
                metrics_of_the_window['mean_' + str(band)] = np.mean(all_windows[conc_s_i, w_i, :])
                metrics_of_the_window['median_' + str(band)] = np.median(all_windows[conc_s_i, w_i, :])
                metrics_of_the_window['std_' + str(band)] = np.std(all_windows[conc_s_i, w_i, :])
                metrics_of_the_window['kurt_' + str(band)] = scipy.stats.kurtosis(all_windows[conc_s_i, w_i, :])
                metrics_of_the_window['skew_' + str(band)] = scipy.stats.skew(all_windows[conc_s_i, w_i, :])
                metrics_of_the_window['IQR_' + str(band)] = scipy.stats.iqr(all_windows[conc_s_i, w_i, :])
                metrics_of_the_window['Mean_power_' + str(band)] = mean_power(all_windows[conc_s_i, w_i, :])
                metrics_of_the_window['MAD_' + str(band)] = 1.4826 * MAD(all_windows[conc_s_i, w_i, :])

            metrics_of_the_window[target_name] = file_target_score
            metrics_of_the_window[target_name2] = file_target_score2
            metrics_windows_of_the_sample_one_band.append(metrics_of_the_window)
        metrics_conc_sample_one_band = OrderedDict()
        for i in range(len(metrics_windows_of_the_sample_one_band)):
            for key in list(metrics_windows_of_the_sample_one_band[i].keys()):
                metrics_conc_sample_one_band[key + '_{}'.format(i)] = metrics_windows_of_the_sample_one_band[i][key]
        band_conc_sample_metrics.append(metrics_conc_sample_one_band)
    return band_conc_sample_metrics, all_windows.shape[0]

if __name__ == '__main__':
    ##### PARAMETERS #####
    name = 'prueba'
    directory = './datasets/whole_data'
    fromat_type = 'set'
    file_target_score = {'SA007_day0_':75, 'SA007_day1_':82, 'SA007_day2_': 78, 'SA007_day3_': 82, 'SA007_day4_': 77, 'SA007_day5_': 72,
                         'SA007_day6_': 78, 'SA007_day7_': 75, 'SA007_day8_': 78, 'SA010_day1_':85, 'SA010_day3_':100, 'SA010_day5_':84,
                         'SA010_day6_':83, 'SA010_day7_':77, 'SA010_day9_':74, 'SA010_day11':81, 'SA010_day12': 74,'SA010_day13':71,
                         'SA014_day1_':79, 'SA014_day2_':83, 'SA014_day3_':84, 'SA014_day5_':71, 'SA014_day6_':73,'SA014_day7_':81,
                         'SA014_day9_':117, 'SA017_day1_':89, 'SA017_day2_':76, 'SA017_day4_':74, 'SA017_day6_':74, 'SA017_day7_':75,
                         'SA017_day8_':76, 'SA047_day1_': 96, 'SA047_day2_': 93, 'SA047_day3_': 84, 'SA047_day4_': 82, 'SA047_day5_': 85,
                         'SA047_day6_': 93, 'SA047_day7_': 98, 'SA047_day9_': 91, 'SA047_day13': 95}
                         # REVIEWED (check)

    file_target_score2 = {'SA007_day0_': 29, 'SA007_day1_':28, 'SA007_day2_':25, 'SA007_day3_':22, 'SA007_day4_':22, 'SA007_day5_':21,
                         'SA007_day6_':21, 'SA007_day7_':25, 'SA007_day8_':23, 'SA010_day1_':33, 'SA010_day3_':36, 'SA010_day5_':35,
                         'SA010_day6_':35, 'SA010_day7_':33, 'SA010_day9_':30, 'SA010_day11':30, 'SA010_day12':28, 'SA010_day13':27,
                         'SA014_day1_':28, 'SA014_day2_':29, 'SA014_day3_':30, 'SA014_day5_':28, 'SA014_day6_':30, 'SA014_day7_':29,
                         'SA014_day9_':39, 'SA017_day1_': 26, 'SA017_day2_':25, 'SA017_day4_': 26, 'SA017_day6_':26, 'SA017_day7_': 26,
                         'SA017_day8_': 27, 'SA047_day1_': 26, 'SA047_day2_': 21, 'SA047_day3_': 21, 'SA047_day4_': 20, 'SA047_day5_': 25,
                         'SA047_day6_': 24, 'SA047_day7_': 27, 'SA047_day9_': 25, 'SA047_day13': 26}
                         # REVIEWED (check)
    target_name = 'PANSS'
    target_name2 = 'PANSS_posit'
    w_len = 10
    conc_samples_len = 40
    conc_samples_overlap = 30
    #Alpha_features =
    n_pools = 4
    ######################
    pool = multiprocessing.Pool(n_pools)
    n_conc_samples = 0
    all_conc_samples_metrics_condensed = list()
    n_conc_samples_per_record = []
    record_names = []
    for root, folders, _ in os.walk(directory):
        for fold_day in sorted(folders):
            print('===========Processing record of {}'.format(fold_day[:11]))
            file_PANSS = file_target_score[fold_day[:11]]
            file_PANSS2 = file_target_score2[fold_day[:11]]
            all_conc_samples_all_band_metrics = list() # len of 6 (Alpha, Beta, Complete, Delta, Theta, Gamma)

            for root_day, _, files in os.walk(os.path.join(root, fold_day)):
                for file in sorted(files):
                    if file.endswith('set'):
                        one_band_conc_samples_metrics, n_conc_samples_particular = compute_features_on_conc_samples_(
                            root_day, file, w_len, conc_samples_len, conc_samples_overlap, file[-9:-4], target_name,
                            file_PANSS, target_name2, file_PANSS2, pool)
                        all_conc_samples_all_band_metrics.append(one_band_conc_samples_metrics)
            print('{} concatenated samples generated for {}.'.format(n_conc_samples_particular, fold_day[6:11]))
            for j in range(n_conc_samples_particular):
                all_conc_samples_all_band_metrics[0][j].update(all_conc_samples_all_band_metrics[1][j])
                all_conc_samples_all_band_metrics[0][j].update(all_conc_samples_all_band_metrics[2][j])
                all_conc_samples_all_band_metrics[0][j].update(all_conc_samples_all_band_metrics[3][j])
                all_conc_samples_all_band_metrics[0][j].update(all_conc_samples_all_band_metrics[4][j])
                all_conc_samples_all_band_metrics[0][j].update(all_conc_samples_all_band_metrics[5][j])
            for conc_sample_m in all_conc_samples_all_band_metrics[0]:
                all_conc_samples_metrics_condensed.append(conc_sample_m)
            n_conc_samples +=  n_conc_samples_particular
            n_conc_samples_per_record.append(n_conc_samples_particular)
            record_names.append(fold_day)

    df_all_conc_samples_metrics = pd.DataFrame(all_conc_samples_metrics_condensed)
    print('A total of {} concatenated samples have been obtained.'.format(n_conc_samples))
    df_all_conc_samples_metrics.to_csv(
        './{}_CONCfeats_{}_w{}cs{}ov{}.csv'.format(name,
                                         str(w_len), str(conc_samples_len), str(conc_samples_overlap)))
    with open('./{}_CONCfeats_n_windows_{}_w{}cs{}ov{}.txt'.format(name,
                                                str(w_len), str(conc_samples_len), str(conc_samples_overlap)), 'w') as f:
        for i_ in range(len(n_conc_samples_per_record)):
            f.write('{},{}\n'.format(record_names[i_], n_conc_samples_per_record[i_]))
    f.close()
