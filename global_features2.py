import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import MAD
import scipy.stats

if __name__ == '__main__':
    f_analysis = 'LZC_median'
    path_csv = './whole_features_data_0.5sec.csv'

    discrim_f = 'PANSS'
    ######################
    df = pd.read_csv(path_csv, index_col=0)
    min_val = min(df[f_analysis])
    max_val = max(df[f_analysis])
    fig,ax = plt.subplots()
    ax2 = ax.twinx()
    df
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
