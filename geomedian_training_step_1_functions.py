import numpy as np
from numpy import genfromtxt

# from spectral import envi

import urban_module as ubm

def bin_entropy(distarr, c1, c2, num_bins):
    """
    Calculate entropy based on bins and class occurences
    """
    enp = np.zeros(num_bins)
    for i in np.arange(num_bins):
        p1=np.double(distarr[i, c1])
        p2=np.double(distarr[i, c2])
        r1=(p1+1)/(p1+p2+2)
        r2=(p2+1)/(p1+p2+2)
        if (p1+p2!=0):
            enp[i]=-(p1*np.log2(r1)+p2*np.log2(r2))/(p1+p2)
        else:
            enp[i]=1

    return enp


def hist_distribution(tgtband, bandnames, oneimg, clsarr, numcls, num_bins):
    """
    Returns:
    distarr: number of occurences of each class in each bin
    """
    ftidx = bandnames.index(tgtband)
    ftvals = oneimg[ftidx, :]

    xlmin = ftvals.min()
    xrmax = ftvals.max()
    pidxlist = []

    distarr = np.zeros((num_bins, numcls), dtype=np.uint64)
    binsize = (xrmax - xlmin) / num_bins
    bounds = np.zeros((num_bins, 2))

    for i in np.arange(num_bins):
        bounds[i, 0] = xlmin + i * binsize
        if (i != num_bins - 1):
            bounds[i, 1] = xlmin + (i + 1) * binsize
        else:
            bounds[i, 1] = xrmax

    for i in np.arange(num_bins):
        if (i != num_bins - 1):
            fbidx = np.where(np.logical_and(ftvals >= bounds[i, 0], ftvals < bounds[i, 1]))[0]
        else:
            fbidx = np.where(np.logical_and(ftvals >= bounds[i, 0], ftvals <= bounds[i, 1]))[0]

        pidxlist.append(fbidx)
        items = clsarr[fbidx]
        for j in np.arange(numcls):
            distarr[i, j] = np.count_nonzero(items == j)

    return bounds, distarr, pidxlist


def create_training_sets(enp, thd, pidxlist, distarr, clsarr, num_bins, c1, c2):
    c1_train = []
    c2_train = []
    binlist = np.where(enp < thd)[0]
    for binidx in binlist:
        fx = pidxlist[binidx]
        if distarr[binidx, c1] > distarr[binidx, c2]:
            c1_idx = fx[clsarr[fx] == c1]
            c1_train.append(c1_idx)

        else:
            c2_idx = fx[clsarr[fx] == c2]
            c2_train.append(c2_idx)

    return c1_train, c2_train


def conclistarr(listarr):
    ntr = len(listarr)
    if ntr > 0:
        arr = listarr[0]
        for i in np.arange(1, ntr):
            arr = np.concatenate((arr, listarr[i]), axis=None)
    else:
        arr = np.asarray(listarr)

    return arr


def sets_by_sampling(idxset, ds):
    abc = np.random.choice(idxset, 3 * ds, replace='False')
    train_set = abc[:ds]
    val_set = abc[ds:2 * ds]
    test_set = abc[2 * ds:]

    return train_set, val_set, test_set

def shuffle_array(ar1):
    ntr=ar1.shape[0]
    shfx=np.random.permutation(ntr)
    ar1=ar1[shfx]
    return ar1

def shuffle_pair(ar1, ar2):
    ntr=ar1.shape[0]
    shfx=np.random.permutation(ntr)
    ar1=ar1[shfx]
    ar2=ar2[shfx]
    return ar1, ar2


def find_feature_index(bandnames, feature_list):
    ntr = len(feature_list)
    flsarr = np.zeros(ntr, dtype=np.int32)
    for i, feature in enumerate(feature_list):
        ftidx = bandnames.index(feature)
        flsarr[i] = ftidx

    return flsarr


def create_sets(input_feature_list, wholedata, c1p_index, c1_tra_num, c1_tst_num, c2p_index, c2_tra_num, c2_tst_num,
                setname, path):
    """
    Saves out training data
    """

    # Output the names of the input features to a text file

    feature_arr = np.asarray(input_feature_list)
    filename = path + '/' + setname + '_feature_list'
    np.savetxt(filename, feature_arr, delimiter=',', fmt='%s')

    # prepare the index of training and testing data set

    c1p_index = shuffle_array(c1p_index)
    c1_tra_index = c1p_index[:c1_tra_num]
    c1_tst_index = c1p_index[c1_tra_num:c1_tra_num + c1_tst_num]

    c2p_index = shuffle_array(c2p_index)
    c2_tra_index = c2p_index[:c2_tra_num]
    c2_tst_index = c2p_index[c2_tra_num:c2_tra_num + c2_tst_num]

    # combine class #1 and class #2 pixels to form training and testing set

    tra_index = np.concatenate((c1_tra_index, c2_tra_index), axis=None)
    tst_index = np.concatenate((c1_tst_index, c2_tst_index), axis=None)

    # Generate class labels
    y_train = np.zeros(c1_tra_num + c2_tra_num, dtype=np.float32)
    y_train[c1_tra_num:] = 1.0
    y_test = np.zeros(c1_tst_num + c2_tst_num, dtype=np.float32)
    y_test[c1_tst_num:] = 1.0

    # Form training and tests data sets
    tra_index, y_train = shuffle_pair(tra_index, y_train)
    tst_index, y_test = shuffle_pair(tst_index, y_test)

    x_train = wholedata[tra_index, :]
    x_test = wholedata[tst_index, :]

    # Output training and testing data as binary files
    filename = setname + '_train_features'
    fileoutput(path, filename, x_train)

    filename = setname + '_train_labels'
    fileoutput(path, filename, y_train)

    filename = setname + '_test_features'
    fileoutput(path, filename, x_test)

    filename = setname + '_test_labels'
    fileoutput(path, filename, y_test)

def fileoutput(path, filestem, data):
    """
    Creates a binary file output
    """
    filename=path+'/'+filestem
    data.tofile(filename)