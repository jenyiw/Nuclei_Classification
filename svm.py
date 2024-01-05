import shutil
import h5py
import numpy as np
import matplotlib.pyplot as plt

#SVM packages
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree


def run_svm_feature(train_arr,
                    test_arr,
                    feature_num:int=-1):
    """
    Run SVM from sklearn.

    Parameters:
    train_arr: N x num. features + 1, ndarray of training data
    test_arr: N x num. features + 1, ndarray of testing data
    feature_num: number of features, int

    Return:
    clf: sklearn classifier
    acc: accuracy score
    f1: F1 score
    """
    # get training data and labels
    x_train = train_arr[:, :feature_num]
    y_train = train_arr[:, -1]

    # train classifier
    clf = SVC()
    clf.fit(x_train, y_train)

    #get test data and labels
    x_test = test_arr[:, :feature_num]
    y_test = test_arr[:, -1]

    #predict
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_pred.astype(np.uint8), y_test.astype(np.uint8))
    f1 = f1_score(y_pred.astype(np.uint8), y_test.astype(np.uint8))

    return clf, acc, f1

def feature_permutation(data_arr,
                        num_feature:int):

    """
    Implement feature permutation.

    Parameters:
    data_arr: N x num. features, ndarray
    num_feature: int, index of feature to permutate

    Return:
    data_arr: N x num. features, ndarray, shuffled data array
    """


    np.random.shuffle(data_arr[:, num_feature])

    return data_arr

def sample_data(train_data):

    """
    Returns training data with an equal number of positive and negative samples.
    Parameters:
    train_data: N x num. features, ndarray, data array to subset. Assumes last column is the label.

    Return:
    sampled_train_data: M x num. features, ndarray, dataset with same number of positive and negative samples
    """

    num_neg = len(train_data[train_data[:, -1]==0])
    num_pos = len(train_data[train_data[:, -1]==1])
    sampled_train = np.random.choice(range(num_neg, num_neg+num_pos), size=num_neg)
    valid_indices = np.concatenate([sampled_train, np.array(range(num_neg))])
    sampled_train_data = train_data[valid_indices, :]

    return sampled_train_data

