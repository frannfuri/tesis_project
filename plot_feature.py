import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    path_csv = 'results_bef_13-07/SA010_selected_features_low_freqs_10sec.csv'
    feature_to_plot = 'LZC_slope_H-env_Complete'
    num_windows_of_each_record = [84, 85, 95, 124, 95, 106, 85, 85, 97]  # w_len NN_av 10 SA010
    #num_windows_of_each_record = [172, 174, 195, 253, 195, 217, 174, 174, 198] # w_len NN_av 5 SA010
    lw = 1
    #num_windows_of_each_record = [1757, 1781, 1989, 2574, 1993, 2210, 1781, 1780, 2016]  # w_len NN_av 0.5 SA010
    #lw=0.3
    '''
    #num_windows_of_each_record = [360, 361, 358, 352, 357, 368, 356, 357, 353, 352]  # w_len NN_av 1
    #num_windows_of_each_record = [119, 119, 118, 116, 118, 122, 118, 118, 117, 116]  # w_len NN_av 3
    #num_windows_of_each_record = [17, 17, 16, 16, 16, 17, 16, 16, 16, 16]  # w_len NN_av 20
    #num_windows_of_each_record = [71, 71, 70, 69, 70, 72, 70, 70, 69, 69]  # w_len NN_av 5
    #num_windows_of_each_record = [1454, 1458, 1449, 1424, 1444, 1490, 1438, 1444, 1428, 1423]  # w_len NN_av 0.25
    #num_windows_of_each_record = [721, 723, 718, 706, 715, 738, 713, 715, 707, 705]  # w_len NN_av 0.5
    #num_windows_of_each_record = [35, 35, 34, 34, 34, 35, 34, 34, 34, 34]  # w_len NN_av 10
    PANSS_total = [98, 95, 96, 93, 84, 82, 85, 93, 98, 91]
    PANSS_posit = [26, 26, 26, 21, 21, 20, 25, 24, 27, 25]
    PANSS_neg = [17, 24, 28, 18, 19, 19, 22, 22, 26, 22]
    PANSS_disorg = [37, 34, 29, 32, 30, 30, 30, 34, 30, 32]
    '''
    PANSS_total = [81, 71, 85, 94, 100, 84, 83, 77, 74]#, 81, 71]
    PANSS_posit = [30, 27, 33, 35, 36, 35, 35, 33, 30]#, 30, 27]
    #############
    df = pd.read_csv(path_csv, index_col=0)
    fig, ax = plt.subplots(5,2,figsize=(13,8))
    plt.suptitle('[w_len={}]'.format(path_csv[-10:-4]),y=0.995)
    labels = ['Day11', 'Day13', 'Day1', 'Day2', 'Day3', 'Day5', 'Day6', 'Day7', 'Day9']
    #labels = ['Day11', 'Day13', 'Day1', 'Day2', 'Day3', 'Day4', 'Day5', 'Day6', 'Day7', 'Day9']
    ylims = (min(df[feature_to_plot]), max(df[feature_to_plot]))
    w_marker1 = 0
    w_marker2 = num_windows_of_each_record[0]
    ax[0,0].plot(list(range(num_windows_of_each_record[0]))[:150], df[feature_to_plot][w_marker1:w_marker2][:150], ms=2, linewidth=lw, c='red')
    ax[0,0].set_title('Feature: {}; {} [Total:{}, P:{}]'.format(feature_to_plot, labels[0], PANSS_total[0], PANSS_posit[0]), fontsize=9)
    ax[0,0].set_ylim(ylims)
    ax[0,0].set_yticks(np.linspace(ylims[0],ylims[1],9))
    ax[0,0].tick_params(labelsize=7)
    ax[0,0].grid()
    ax_n = 0
    for i in range(1, len(num_windows_of_each_record)):
        w_marker1 = w_marker2
        w_marker2 += num_windows_of_each_record[i]
        ax_n += (i+1)%2
        ax[ax_n,i%2].plot(list(range(num_windows_of_each_record[i]))[:150], df[feature_to_plot][w_marker1:w_marker2][:150], ms=2, linewidth=lw, c='red')
        ax[ax_n,i%2].set_title('Feature: {}; {} [Total:{}, P:{}]'.format(feature_to_plot, labels[i], PANSS_total[i], PANSS_posit[i]), fontsize=9)
        ax[ax_n,i%2].set_ylim(ylims)
        ax[ax_n,i%2].set_yticks(np.linspace(ylims[0], ylims[1], 9))
        ax[ax_n, i % 2].tick_params(labelsize=7)
        ax[ax_n,i%2].grid()
        if i == 8 or i == 9 or i ==7:
            ax[ax_n,i%2].set_xlabel('Sliding windows on time', fontsize=9)
    plt.tight_layout()
    plt.show()