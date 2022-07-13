import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils import MAD, robust_z_score_norm
import umap
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats
#num_windows_of_each_record = [17, 17, 16, 16, 16, 17, 16, 16, 16, 16]  # w_len NN_av 20
#num_windows_of_each_record = [71, 71, 70, 69, 70, 72, 70, 70, 69, 69]  # w_len NN_av 5
# num_windows_of_each_record = [1454, 1458, 1449, 1424, 1444, 1490, 1438, 1444, 1428, 1423]  # w_len NN_av 0.25
#num_windows_of_each_record=[119, 119, 118, 116, 118, 122, 118, 118, 117, 116]
def UMAP_plot_general(path=None, features_to_use=None, num_windows=None , use_scale=None,
                      random_state=None, n_components=2, transform_seed=42, alpha_=0.4):
    plt.ion()
    df_features = pd.read_csv(path, index_col=0)
    labels=['day11', 'day13', 'day1', 'day2', 'day3', 'day4', 'day5','day6' ,'day7', 'day9']
    #features_to_use.append(discriminator)
    data = df_features[features_to_use].to_numpy()
    if use_scale == 'standard':
        print('Using standard scaler normalization!')
        scaled_data = StandardScaler().fit_transform(data[:,:-1])
    elif use_scale == 'robustzscore':
        print('Using robust z-score normalization!')
        scaled_data = []
        for f in range(data.shape[1]-1):
            scaled_feature_vector = robust_z_score_norm(data[:,f])
            scaled_data.append(scaled_feature_vector)
        scaled_data = np.transpose(np.array(scaled_data))
    else:
        print('Working without normalization...')
        scaled_data = data[:,:-1]
    reducer = umap.UMAP(random_state=random_state, transform_seed=transform_seed, n_components=n_components)
    embedding = reducer.fit_transform(scaled_data)
    np.save('./embedding.npy', embedding)
    np.save('./data.npy', data)
    print('Embedding shape: ' + str(embedding.shape))
    w_marker1 = 0
    w_marker2 = num_windows[0]
    cos = ['blue', 'red', 'saddlebrown', 'darkslategray', 'indigo', 'green', 'dimgray', 'darkorange',
           'cyan', 'olive']
    if n_components==2:
        plt.scatter(embedding[:, 0][w_marker1:w_marker2], embedding[:, 1][w_marker1:w_marker2], c=cos[0], alpha=alpha_,
                    label=labels[0])
        plt.title('UMAP; {};\nNormalization: {};  Features:\n{}'.format(
            path.split('/')[-1], use_scale, features_to_use), fontsize=7)
        for i in range(1,len(num_windows)):
            w_marker1=w_marker2
            w_marker2 += num_windows[i]
            plt.scatter(embedding[:, 0][w_marker1:w_marker2], embedding[:, 1][w_marker1:w_marker2], c=cos[i], alpha=alpha_,
                        label=labels[i])
        plt.legend(loc='best')
    else:
        fig=plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title('UMAP; {};\nNormalization: {};  Features:\n{}'.format(
            path.split('/')[-1], use_scale, features_to_use), fontsize=7)
        ax.scatter(embedding[:, 0][w_marker1:w_marker2], embedding[:, 1][w_marker1:w_marker2], embedding[:, 2][w_marker1:w_marker2], c=cos[0], alpha=alpha_,
                    label=labels[0])
        for i in range(1, len(num_windows)):
            w_marker1 = w_marker2
            w_marker2 += num_windows[i]
            ax.scatter(embedding[:, 0][w_marker1:w_marker2], embedding[:, 1][w_marker1:w_marker2], embedding[:, 2][w_marker1:w_marker2], c=cos[i], alpha=alpha_,
                        label=labels[i])
        ax.legend(loc='best')
    plt.tight_layout()
    plt.show()

def pair_of_features(path=None, feature1=None, feature2=None, num_windows=None , use_scale=None,
                     alpha_=0.6, discrim_feat_ax=None):
    df = pd.read_csv(path, index_col=0)
    cos = ['blue', 'red', 'saddlebrown', 'darkslategray', 'indigo', 'green', 'dimgray', 'darkorange',
           'cyan', 'olive']
    labels= ['day11', 'day13', 'day1', 'day2', 'day3', 'day4', 'day5', 'day6', 'day7', 'day9']
    w_marker1=0
    w_marker2=num_windows[0]
    if discrim_feat_ax is None:
        plt.scatter(df[feature1][w_marker1:w_marker2], df[feature2][w_marker1:w_marker2], c=cos[0], label=labels[0], alpha=alpha_)
        for i in range(1,len(num_windows)):
            w_marker1 = w_marker2
            w_marker2 += num_windows[i]
            plt.scatter(df[feature1][w_marker1:w_marker2], df[feature2][w_marker1:w_marker2], c=cos[i], label=labels[i], alpha=alpha_)
        plt.legend(loc='best')
        plt.xlabel(feature1)
        plt.xlim((min(df[feature1]), max(df[feature1])))
        plt.ylabel(feature2)
        plt.ylim((min(df[feature2]), max(df[feature2])))
        plt.title('Windows of length = {}'.format(path[-10:-4]), fontsize=9)
    else:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(df[discrim_feat_ax][w_marker1:w_marker2], df[feature1][w_marker1:w_marker2], df[feature2][w_marker1:w_marker2], c=cos[0], label=labels[0], alpha=alpha_)
        for i in range(1, len(num_windows)):
            w_marker1 = w_marker2
            w_marker2 += num_windows[i]
            ax.scatter(df[discrim_feat_ax][w_marker1:w_marker2], df[feature1][w_marker1:w_marker2], df[feature2][w_marker1:w_marker2], c=cos[i], label=labels[i], alpha=alpha_)
        ax.legend(loc='best')
        ax.set_xlabel(discrim_feat_ax)
        ax.set_ylabel(feature1)
        ax.set_zlabel(feature2)
        ax.set_title('Windows of length = {}'.format(path[-10:-4]), fontsize=9)
    plt.show()
    plt.tight_layout()