# This program reads 3 Landsat surface reflectance-derived data, applies k-mean clustering algorithm to generate M
# clusters, which will be combined to form
# urban and non-urban classes in the next step in the work flow


import os
import sys
import sklearn as sk
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
from spectral import *
import urban_module as ubm

# directory where the input indices data can be found
path = sys.argv[1]

# file name of the brightness index for the clusters image 
brightnessfile = sys.argv[2]

# file name of the greenness index for the clusters image 
greennessfile = sys.argv[3]

# file name of the wetness index for the clusters image 
wetnessfile = sys.argv[4]

#TODO: Add path to smad here

# the number of clusters to generate 
numcls = int(sys.argv[5])

filelist = [brightnessfile, greennessfile, wetnessfile]

hdrfile = path + '/' + brightnessfile + '.hdr'
h = envi.read_envi_header(hdrfile)

nrow = np.int32(h['lines'])
ncol = np.int32(h['samples'])
pnum = nrow * ncol

#TODO: Change number of dimensions to 4
data = np.zeros([pnum, 3], dtype=np.float32)

# a function to read indices data file, extract the relevant band in the file
# put the data in the columns of an array as inputs for the clustering algorithm 

col = 0
for filename in filelist:
    h, oneimage, pnum = ubm.load_envi_data_float(path, filename)
    data[:, col] = oneimage
    col = col + 1

#TODO: modify to another covariate
def classifyoneblock_spectral(data, nrow, ncol, rp, numcls):
    """
    This function calls the clustering function and masks water pixels
    :param data: input array
    :param nrow: dimensions
    :param ncol: dimensions
    :param rp:
    :param numcls: Number of classes
    :return:
    """
    pnum = nrow * ncol

    greenness = data[:, 1]
    brightness = data[:, 0]
    wetness = data[:, 2]

    waterthd = 0.08

    # exclude water pixels from input data

    wateridx = np.where(wetness >= waterthd)[0]
    nonwateridx = np.where(wetness < waterthd)[0]
    stidx = np.argsort(greenness)

    # load non-water pixels data as input for clustering algorithm
    sprst = data
    sprst = sprst[nonwateridx]
    sprst = np.asarray(sprst)

    # re-scale the data
    sprst_scale = preprocessing.scale(sprst)

    width = 2

    clscount = np.zeros([pnum, numcls], dtype=np.int32)

    for i in range(rp):
        clsimg = clusterclassifier(sprst_scale, pnum, numcls, wateridx, nonwateridx, brightness, greenness, wetness)
        for k in range(pnum):
            m = clsimg[k]
            clscount[k, m] = clscount[k, m] + 1

    bbclsimg = clscount.argmax(axis=1)
    bbclsimg = bbclsimg.astype(np.int8)

    return bbclsimg

# TODO: modify to add another covariate
def clusterclassifier(scaled_data, pnum, numcls, wateridx, nonwateridx, brightness, greenness, wetness):
    clsimg = np.zeros(pnum, dtype=np.int8)
    clsimg[wateridx] = -1
    clustering = KMeans(init='k-means++', precompute_distances=True, algorithm="full", n_clusters=numcls - 1)
    clustering.n_jobs = 6
    labels = clustering.fit(scaled_data)
    clsimg[nonwateridx] = labels.labels_
    clsimg[:] = clsimg[:] + 1

    cc = np.zeros(numcls)
    greencc = np.zeros(numcls)
    brightcc = np.zeros(numcls)
    wetcc = np.zeros(numcls)

    for i in range(pnum):
        clab = clsimg[i]
        greencc[clab] = greencc[clab] + greenness[i]
        brightcc[clab] = brightcc[clab] + brightness[i]
        wetcc[clab] = wetcc[clab] + wetness[i]
        cc[clab] = cc[clab] + 1

    for i in range(numcls):
        if (cc[i] > 0):
            greencc[i] = greencc[i] / cc[i]
            brightcc[i] = brightcc[i] / cc[i]
            wetcc[i] = wetcc[i] / cc[i]

    greencc[0] = 1000
    stidx = np.argsort(greencc)

    maps = np.zeros(numcls, dtype=np.int32)
    for i in range(numcls):
        k = numcls - 1 - i
        maps[stidx[k]] = i

    for i in range(pnum):
        clsimg[i] = maps[clsimg[i]]

    return clsimg


rp = 5

clsimg = classifyoneblock_spectral(data, nrow, ncol, rp, numcls)
clsimg = clsimg.astype(np.int8)

sprst = preprocessing.scale(data[:, 0:3])

cores = np.zeros([numcls, 3])
dist = np.zeros([numcls, 2])

for i in range(numcls):
    classidx = np.where(clsimg == i)[0]
    apm = sprst[classidx]
    cores[i, :] = apm.mean(axis=0)

for i in range(1, numcls):
    dist[i, 0] = np.sqrt(((cores[i, :] - cores[1, :]) ** 2).sum(0))
    dist[i, 1] = np.sqrt(((cores[i, :] - cores[numcls - 1, :]) ** 2).sum(0))

mapcls = np.zeros(numcls)
mapcls = mapcls.astype(np.int8)

midpp = int(numcls / 2) + 1

for i in range(1, numcls):
    if (i < midpp):
        mapcls[i] = 1
    else:
        mapcls[i] = 3

fstem = 'urban_spec_5c_raw'
bandnames = ['land cover class raw']
bands = 1
datatype = 1
description = 'Land cover classification raw'
ubm.outputenvifile(clsimg, path, h, fstem, bandnames, bands, datatype, description)

bbclsimg = mapcls[clsimg]

fstem = 'urban_spec_5c'
bandnames = ['land cover class']
bands = 1
datatype = 1
description = 'Land cover classification'
ubm.outputenvifile(bbclsimg, path, h, fstem, bandnames, bands, datatype, description)
