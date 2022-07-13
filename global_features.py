import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import MAD
import scipy.stats

if __name__ == '__main__':
    f_analysis = 'kurt_Theta'
    path_csv = 'results_bef_13-07/SA010_selected_features_low_freqs_10sec.csv'
    num_windows_of_each_record = [84, 85, 95, 124, 95, 106, 85, 85, 97] # w_len NN_av 10 SA010
    #num_windows_of_each_record = [172, 174, 195, 253, 195, 217, 174, 174, 198] # w_len NN_av 5 SA010
    #num_windows_of_each_record = [1757, 1781, 1989, 2574, 1993, 2210, 1781, 1780, 2016]  # w_len NN_av 0.5 SA010
    '''
    #num_windows_of_each_record = [360, 361, 358, 352, 357, 368, 356, 357, 353, 352]  # w_len NN_av 1
    #num_windows_of_each_record = [119, 119, 118, 116, 118, 122, 118, 118, 117, 116]  # w_len NN_av 3
    #num_windows_of_each_record = [17, 17, 16, 16, 16, 17, 16, 16, 16, 16]  # w_len NN_av 20
    #num_windows_of_each_record = [71, 71, 70, 69, 70, 72, 70, 70, 69, 69]  # w_len NN_av 5
    #num_windows_of_each_record = [1454, 1458, 1449, 1424, 1444, 1490, 1438, 1444, 1428, 1423]  # w_len NN_av 0.25
    #num_windows_of_each_record = [35, 35, 34, 34, 34, 35, 34, 34, 34, 34]  # w_len NN_av 10
    '''
    per_day = False
    discrim_f = 'PANSS_posit'
    #scores = [81, 71, 85, 94, 100, 84, 83, 77, 74]
    #scores = [71, 74, 77, 81, 83, 84, 85, 94, 100]
    ordered_scores = [30, 27, 33, 35, 36, 35, 35, 33, 30]
    scores = [27,30,33,35,36]
    ######################
    df = pd.read_csv(path_csv, index_col=0)
    min_val = min(df[f_analysis])
    max_val = max(df[f_analysis])
    if per_day:
        w_marker1 = 0
        w_marker2 = num_windows_of_each_record[0]
        print(f_analysis)
        medians = []
        mads = []
        kurts = []
        skews = []
        medians.append(np.median(df[f_analysis][w_marker1:w_marker2]))
        print('median: {}'.format(np.median(df[f_analysis][w_marker1:w_marker2])))
        mads.append(MAD(df[f_analysis][w_marker1:w_marker2]))
        print('MAD: {}'.format(MAD(df[f_analysis][w_marker1:w_marker2])))
        kurts.append(scipy.stats.kurtosis(df[f_analysis][w_marker1:w_marker2]))
        print('kurtosis: {}'.format(scipy.stats.kurtosis(df[f_analysis][w_marker1:w_marker2])))
        skews.append(scipy.stats.skew(df[f_analysis][w_marker1:w_marker2]))
        print('skewness: {}'.format(scipy.stats.skew(df[f_analysis][w_marker1:w_marker2])))
        for i in range(1, len(num_windows_of_each_record)):
            w_marker1 = w_marker2
            w_marker2 += num_windows_of_each_record[i]
            medians.append(np.median(df[f_analysis][w_marker1:w_marker2]))
            print('median: {}'.format(np.median(df[f_analysis][w_marker1:w_marker2])))
            mads.append(MAD(df[f_analysis][w_marker1:w_marker2]))
            print('MAD: {}'.format(MAD(df[f_analysis][w_marker1:w_marker2])))
            kurts.append(scipy.stats.kurtosis(df[f_analysis][w_marker1:w_marker2]))
            print('kurtosis: {}'.format(scipy.stats.kurtosis(df[f_analysis][w_marker1:w_marker2])))
            skews.append(scipy.stats.skew(df[f_analysis][w_marker1:w_marker2]))
            print('skewness: {}'.format(scipy.stats.skew(df[f_analysis][w_marker1:w_marker2])))
        fig, ax = plt.subplots()
        ax.set_title('{} [w_len={}]'.format(f_analysis, path_csv[-10:-4]))
        #ax.errorbar([0,1,2,3,4,5,6,7,8,9],medians, yerr=mads,label='median')
        #ax.plot(medians, label='Median')
        ax.errorbar([0,1,2,3,4,5,6,7,8 ], medians, yerr=mads, label='Median')
        #ax.set_ylim((min(medians), max(medians)))
        ax.set_ylabel('Median')
        ax.grid()
        ax.set_xlabel(discrim_f)
        plt.grid()
        plt.tight_layout()
        ax2 = ax.twinx()
        ax2.plot(kurts, label='kurtosis', c='r')
        ax2.plot(skews, label='skewness', c='g')
        ax2.set_ylabel('Kurtosis & Skewness')
        fig.legend()
        day_labels_ = ['day11', 'day13', 'day1', 'day2', 'day3', 'day5', 'day6', 'day7', 'day9']
        plt.xticks(list(range(0,len(num_windows_of_each_record))),
                   day_labels_)
        plt.grid()
        fig, axs = plt.subplots(3,3, figsize=(15,20))
        fig.suptitle('Histograms per day, of feature {}, window length {} seconds'.format(f_analysis, path_csv[-9:-7]),
                     fontsize=10)
        w_marker1 = 0
        w_marker2 = num_windows_of_each_record[0]
        axs[0,0].hist(df[f_analysis][w_marker1:w_marker2], 15)
        axs[0,0].set_title('{}, score={}'.format(day_labels_[0], ordered_scores[0]), fontsize=8)
        axs[0,0].set_xlim((min_val, max_val))
        ax_n = 0
        for i in range(1,len(num_windows_of_each_record)):
            w_marker1 = w_marker2
            w_marker2 += num_windows_of_each_record[i]
            if i%3 == 0:
                ax_n += 1
            axs[i%3, ax_n].hist(df[f_analysis][w_marker1:w_marker2], 15)
            axs[i%3, ax_n].set_title('{}, score={}'.format(day_labels_[i], ordered_scores[i]), fontsize=8)
            axs[i%3, ax_n].set_xlim((min_val, max_val))
        plt.subplots_adjust(hspace=0.35)
    else:
        #if discrim_f == 'PANSS':
        #    scores = [82, 84, 85, 91, 93, 95, 96, 98]
        #elif discrim_f == 'PANSS_posit':
        #    scores = [20, 21, 24, 25, 26, 27]
        fig,ax = plt.subplots()
        ax2 = ax.twinx()
        medians=[]
        kurts=[]
        skews=[]
        mads=[]
        for i in range(len(scores)):
            medians.append(np.median(df[f_analysis][df[discrim_f]==scores[i]]))
            mads.append(MAD(df[f_analysis][df[discrim_f]==scores[i]]))
            kurts.append(scipy.stats.kurtosis(df[f_analysis][df[discrim_f] == scores[i]]))
            skews.append(scipy.stats.skew(df[f_analysis][df[discrim_f] == scores[i]]))
        ax.errorbar(scores, medians, yerr=mads, label='median',elinewidth=0.8)
        #ax.plot(scores, medians, label='median')
        ax2.plot(scores, kurts, label='kurtosis', c='r')
        ax2.plot(scores, skews ,label='skewness', c='g')
        ax.set_ylabel('Median')
        ax.set_title('{} [w_len={}]'.format(f_analysis, path_csv[-10:-4]))
        ax.set_xlabel(discrim_f)
        ax2.set_ylabel('Kurtosis & Skewness')
        fig.legend()
        plt.grid()
        plt.tight_layout()
        fig, axs = plt.subplots(3,2, figsize=(15,15))
        ax_n = 0
        fig.suptitle('Histograms per {}, of feature {}, window length {} seconds'.format(discrim_f, f_analysis,
                      path_csv[-9:-7]), fontsize=10)
        for i in range(len(scores)):
            if i!=0 and i%2==0:
                ax_n += 1
            axs[ax_n, i%2].hist(df[f_analysis][df[discrim_f]==scores[i]], 15)
            axs[ax_n, i%2].set_xlim((min_val, max_val))
            axs[ax_n, i%2].set_title('{} = {}'.format(discrim_f, scores[i]), fontsize=8)
        plt.subplots_adjust(hspace=0.35)
    plt.show()
    a = 0
