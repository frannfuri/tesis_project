import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    path_csv = 'results_bef_13-07/SA010_selected_features_low_freqs_5sec.csv'
    feature_to_plot = 'LZC_slope_H-pow'
    discrim_feature = 'PANSS_posit'
    alpha_ = 0.2
    #num_windows_of_each_record = [84, 85, 95, 124, 95, 106, 85, 85, 97]  # w_len NN_av 10 SA010
    num_windows_of_each_record = [172, 174, 195, 253, 195, 217, 174, 174, 198] # w_len NN_av 5 SA010
    #lw = 1
    # num_windows_of_each_record = [1757, 1781, 1989, 2574, 1993, 2210, 1781, 1780, 2016]  # w_len NN_av 0.5 SA010
    # lw=0.3
    '''
    ## SA047 ##
    #num_windows_of_each_record = [360, 361, 358, 352, 357, 368, 356, 357, 353, 352]  # w_len NN_av 1
    #num_windows_of_each_record = [119, 119, 118, 116, 118, 122, 118, 118, 117, 116]  # w_len NN_av 3
    #num_windows_of_each_record = [17, 17, 16, 16, 16, 17, 16, 16, 16, 16]  # w_len NN_av 20
    #num_windows_of_each_record = [71, 71, 70, 69, 70, 72, 70, 70, 69, 69]  # w_len NN_av 5
    #num_windows_of_each_record = [1454, 1458, 1449, 1424, 1444, 1490, 1438, 1444, 1428, 1423]  # w_len NN_av 0.25
    #num_windows_of_each_record = [721, 723, 718, 706, 715, 738, 713, 715, 707, 705]  # w_len NN_av 0.5
    num_windows_of_each_record = [35, 35, 34, 34, 34, 35, 34, 34, 34, 34]  # w_len NN_av 10
    '''
    #############

    df = pd.read_csv(path_csv, index_col=0)
    labels = ['Day11', 'Day13', 'Day1', 'Day2', 'Day3', 'Day5', 'Day6', 'Day7', 'Day9']
    bands_list = ['Delta', 'Theta', 'Alpha', 'Gamma', 'Complete']
    if not (feature_to_plot=='IAF_post_chns' or feature_to_plot=='CoG_post_chns' or feature_to_plot=='AlphaBand_low_bound' or feature_to_plot=='AlphaBand_high_bound'):
        fig, ax = plt.subplots(1, 5, figsize=(13,8))
        w_marker1 = 0
        w_marker2 = num_windows_of_each_record[0]
        ax[0].scatter(df[discrim_feature][w_marker1:w_marker2], df[feature_to_plot+'_Delta'][w_marker1:w_marker2], label=labels[0], alpha=alpha_)
        delta_y = max(df[feature_to_plot+'_Delta']) - min(df[feature_to_plot+'_Delta'])
        ax[0].set_ylim((min(df[feature_to_plot+'_Delta'])-delta_y*0.05, max(df[feature_to_plot+'_Delta'])+delta_y*0.05))
        ax[1].scatter(df[discrim_feature][w_marker1:w_marker2], df[feature_to_plot+'_Theta'][w_marker1:w_marker2], label=labels[0], alpha=alpha_)
        delta_y = max(df[feature_to_plot + '_Theta']) - min(df[feature_to_plot + '_Theta'])
        ax[1].set_ylim((min(df[feature_to_plot + '_Theta']) - delta_y * 0.05, max(df[feature_to_plot + '_Theta']) + delta_y * 0.05))
        ax[2].scatter(df[discrim_feature][w_marker1:w_marker2], df[feature_to_plot+'_Alpha'][w_marker1:w_marker2], label=labels[0], alpha=alpha_)
        delta_y = max(df[feature_to_plot + '_Alpha']) - min(df[feature_to_plot + '_Alpha'])
        ax[2].set_ylim((min(df[feature_to_plot + '_Alpha']) - delta_y * 0.05, max(df[feature_to_plot + '_Alpha']) + delta_y * 0.05))
        if feature_to_plot+'_Gamma' in df.keys():
            ax[3].scatter(df[discrim_feature][w_marker1:w_marker2], df[feature_to_plot+'_Gamma'][w_marker1:w_marker2], label=labels[0], alpha=alpha_)
            delta_y = max(df[feature_to_plot + '_Gamma']) - min(df[feature_to_plot + '_Gamma'])
            ax[3].set_ylim((min(df[feature_to_plot + '_Gamma']) - delta_y * 0.05,
                            max(df[feature_to_plot + '_Gamma']) + delta_y * 0.05))
        if feature_to_plot + '_Complete' in df.keys():
            ax[4].scatter(df[discrim_feature][w_marker1:w_marker2], df[feature_to_plot+'_Complete'][w_marker1:w_marker2], label=labels[0], alpha=alpha_)
            delta_y = max(df[feature_to_plot + '_Complete']) - min(df[feature_to_plot + '_Complete'])
            ax[4].set_ylim((min(df[feature_to_plot + '_Complete']) - delta_y * 0.05,
                            max(df[feature_to_plot + '_Complete']) + delta_y * 0.05))
        for i in range(0,5):
            ax[i].set_xlabel(discrim_feature + ' score', fontsize=7)
            ax[i].set_ylabel(feature_to_plot, fontsize=7)
            ax[i].set_title(feature_to_plot + ' (' + bands_list[i] + ') [w_len=' + path_csv[-10:-4]+']', fontsize=10)
            ax[i].set_xlim((min(df[discrim_feature])-1, max(df[discrim_feature])+1))
        for i in range(1, len(num_windows_of_each_record)):
            w_marker1 = w_marker2
            w_marker2 += num_windows_of_each_record[i]
            ax[0].scatter(df[discrim_feature][w_marker1:w_marker2], df[feature_to_plot + '_Delta'][w_marker1:w_marker2],
                          label=labels[i], alpha=alpha_)
            ax[1].scatter(df[discrim_feature][w_marker1:w_marker2], df[feature_to_plot + '_Theta'][w_marker1:w_marker2],
                          label=labels[i], alpha=alpha_)
            ax[2].scatter(df[discrim_feature][w_marker1:w_marker2], df[feature_to_plot + '_Alpha'][w_marker1:w_marker2],
                          label=labels[i], alpha=alpha_)
            if feature_to_plot + '_Gamma' in df.keys():
                ax[3].scatter(df[discrim_feature][w_marker1:w_marker2], df[feature_to_plot + '_Gamma'][w_marker1:w_marker2],
                              label=labels[i], alpha=alpha_)
            if feature_to_plot + '_Complete' in df.keys():
                ax[4].scatter(df[discrim_feature][w_marker1:w_marker2], df[feature_to_plot + '_Complete'][w_marker1:w_marker2],
                              label=labels[i], alpha=alpha_)
        handles, labels = ax[0].get_legend_handles_labels()
        leg = fig.legend(handles, labels, loc='lower right', prop={'size':8}, ncol=len(df.columns))
        leg.get_frame().set_edgecolor('black')
    else:
        w_marker1 = 0
        w_marker2 = num_windows_of_each_record[0]
        plt.scatter(df[discrim_feature][w_marker1:w_marker2], df[feature_to_plot][w_marker1:w_marker2],
                      label=labels[0], alpha=alpha_)
        delta_y = max(df[feature_to_plot]) - min(df[feature_to_plot])
        plt.ylim((min(df[feature_to_plot]) - delta_y * 0.05,
                        max(df[feature_to_plot]) + delta_y * 0.05))
        plt.xlabel(discrim_feature + ' score', fontsize=7)
        plt.ylabel(feature_to_plot, fontsize=7)
        plt.title(feature_to_plot + ' (Alpha) [w_len=' + path_csv[-10:-4]+']', fontsize=10)
        plt.xlim((min(df[discrim_feature])-1, max(df[discrim_feature])+1))
        for i in range(1, len(num_windows_of_each_record)):
            w_marker1 = w_marker2
            w_marker2 += num_windows_of_each_record[i]
            plt.scatter(df[discrim_feature][w_marker1:w_marker2], df[feature_to_plot][w_marker1:w_marker2],
                          label=labels[i], alpha=alpha_)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()