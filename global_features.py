import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import MAD
import scipy.stats
from utils import value

if __name__ == '__main__':
    f_analysis = 'LZC_H-env_Complete'
    #path_csv = './whole_features_data_0.5sec.csv'
    path_csv = '/home/jack/cluster_results/whole_features_data_15sec_fixed.csv'
    subj_name = 'SA047'
    discrim_f = 'PANSS'
    n_bins = 25 #for the histogram
    ###########################
    df = pd.read_csv(path_csv, index_col=0)
    df = df[df['SUBJ']==subj_name]
    scores = np.sort(df[discrim_f].unique())
    min_val = min(df[f_analysis])
    max_val = max(df[f_analysis])
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
    ax.errorbar(scores, medians, yerr=mads, label='median',elinewidth=0.8)#, marker='.', markersize=7)
    ax2.plot(scores, kurts, label='kurtosis', c='r')
    ax2.plot(scores, skews ,label='skewness', c='g')
    ax.set_ylabel('Median')
    ax.set_title('{} {} [w_len={}]'.format(f_analysis, subj_name, path_csv[-10:-4]))
    ax.set_xlabel(discrim_f)
    ax2.set_ylabel('Kurtosis & Skewness')
    fig.legend()
    plt.grid()
    plt.tight_layout()
    fig, axs = plt.subplots(int(np.ceil(len(scores)/3)),3, figsize=(15,15))
    ax_n = 0
    fig.suptitle('Histograms per {}, of feature {}, {} window length {} seconds'.format(discrim_f, f_analysis, subj_name,
                  path_csv[-9:-7]), fontsize=10)
    for i in range(len(scores)):
        if np.ceil(len(scores) / 3) < 4:
            axs[ax_n].hist(df[f_analysis][df[discrim_f] == scores[i]], n_bins)
            axs[ax_n].set_xlim((min_val, max_val))
            axs[ax_n].set_title('{} = {}'.format(discrim_f, scores[i]), fontsize=8)
            ax_n += 1
        else:
            if i != 0 and i % 3 == 0:
                ax_n += 1
            axs[ax_n, i%3].hist(df[f_analysis][df[discrim_f]==scores[i]], n_bins)
            axs[ax_n, i%3].set_xlim((min_val, max_val))
            axs[ax_n, i%3].set_title('{} = {}'.format(discrim_f, scores[i]), fontsize=8)
    plt.subplots_adjust(hspace=0.35)
    plt.show()
    a = 0
