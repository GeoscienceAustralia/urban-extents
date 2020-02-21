import numpy as np

def load_datafile_pair(path, feature_filename, label_filename, numfte):
    filename = path + '/' + feature_filename
    features = np.fromfile(filename, dtype=np.float32)
    irow = int(features.shape[0] / numfte)
    features = features.reshape((irow, numfte))
    filename = path + '/' + label_filename
    labels = np.fromfile(filename, dtype=np.float32)
    return features, labels


def hotcode_categorical(data, numcls):
    ntr = data.shape[0]
    hc_data = np.zeros((ntr, numcls), dtype=np.float32)
    for i in np.arange(ntr):
        hc_data[i, int(data[i])] = 1.0

    return hc_data


def calc_std_paras(data):
    ntr = data.shape[1]
    paras = np.zeros(ntr * 2, dtype=np.float32)
    for i in np.arange(ntr):
        clm = data[:, i]
        mu = clm.mean()
        std = clm.std()
        paras[i] = mu
        paras[i + ntr] = std

    return paras


def std_datasets(data, rs):
    ntr = data.shape[1]
    paras = np.zeros(ntr * 2, dtype=np.float32)
    for i in np.arange(ntr):
        clm = data[:, i]
        mu = clm.mean()
        std = clm.std()
        clm = (clm - mu) / (rs * std)
        paras[i] = mu
        paras[i + ntr] = std
        data[:, i] = clm

    return data, paras


def std_by_paramters(data, rs, msarr):
    ntr = data.shape[1]
    for i in np.arange(ntr):
        clm = data[:, i]
        mu = msarr[i]
        std = msarr[i + ntr]
        clm = (clm - mu) / (rs * std)
        data[:, i] = clm

    return data


def find_feature_index(bandnames, feature_list):
    ntr = len(feature_list)
    flsarr = np.zeros(ntr, dtype=np.int32)
    for i, feature in enumerate(feature_list):
        ftidx = bandnames.index(feature)
        flsarr[i] = ftidx

    return flsarr
