import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
from tensorflow.keras.datasets import mnist
import numpy as np

from numpy import genfromtxt
from spectral import *

import urban_module as ubm
import sys
import os


def load_datafile_pair(path, feature_filename, label_filename, numfte):
    filename=path+'/'+feature_filename
    features=np.fromfile(filename, dtype=np.float32)
    irow = int(features.shape[0]/numfte)
    features=features.reshape((irow, numfte))
    filename=path+'/'+label_filename
    labels=np.fromfile(filename, dtype=np.float32)
    return features, labels

def hotcode_categorical(data, numcls):
    ntr=data.shape[0]
    hc_data=np.zeros((ntr,numcls), dtype=np.float32)
    for i in np.arange(ntr):
        hc_data[i, int(data[i])]=1.0

    return hc_data


def calc_std_paras(data):
    ntr=data.shape[1]
    paras=np.zeros(ntr*2, dtype=np.float32)
    for i in np.arange(ntr):
        clm=data[:, i]
        mu=clm.mean()
        std=clm.std()
        paras[i]=mu
        paras[i+ntr]=std
      
    return paras

def std_datasets(data, rs):
    ntr=data.shape[1]
    paras=np.zeros(ntr*2, dtype=np.float32)
    for i in np.arange(ntr):
        clm=data[:, i]
        mu=clm.mean()
        std=clm.std()
        clm=(clm-mu)/(rs*std)
        paras[i]=mu
        paras[i+ntr]=std
        data[:,i]=clm
      
    return data, paras

def std_by_paramters(data, rs, msarr):
    ntr=data.shape[1]
    for i in np.arange(ntr):
        clm=data[:, i]
        mu=msarr[i]
        std=msarr[i+ntr]
        clm=(clm-mu)/(rs*std)
        data[:,i]=clm
        
    return data


def find_feature_index(bandnames, feature_list):
    ntr=len(feature_list)
    flsarr=np.zeros(ntr, dtype=np.int32)
    for i, feature in enumerate(feature_list):
        ftidx=bandnames.index(feature)
        flsarr[i]=ftidx
        
    return flsarr


param=sys.argv

path=param[1]
setname=param[2]

#dirc='/g/data1/u46/pjt554/urban_change_s2_full_sites/brisbane'
#path=dirc+'/2018'

#h, oneimg, pnum, bandnames, clsarr = ubm.load_data(path)

filename=path+'/'+setname+'_feature_list'
featurelist=np.loadtxt(filename, delimiter=',', dtype='str')

numfte = featurelist.shape[0]
print('number of features =', numfte)


hdrfile=path+'/urban_spec_5c.hdr'
h=envi.read_envi_header(hdrfile)

irow=np.int32(h['lines'])
icol=np.int32(h['samples'])
pnum = irow*icol

bandnames=featurelist
numbands=len(bandnames)
allpixels=np.zeros((pnum, numbands), dtype=np.float32)


j=0
for tgtband in bandnames:
    filename=tgtband
    h, oneband, pnum = ubm.load_envi_data_float(path, filename)
    oneband=oneband[0]
    allpixels[:, j]=oneband
    j=j+1
        





#flsarr=find_feature_index(bandnames, featurelist)

x_train_filename = setname+'_train_features'
y_train_filename = setname+'_train_labels'

train_features, train_labels = load_datafile_pair(path, x_train_filename, y_train_filename, numfte)


x_test_filename = setname+'_test_features'
y_test_filename = setname+'_test_labels'

test_features, test_labels = load_datafile_pair(path, x_test_filename, y_test_filename, numfte)


#norm_paras =calc_std_paras(allpixels)
#train_features = std_by_paramters(train_features, 2, norm_paras)

train_features, norm_paras =std_datasets(train_features, 2)

filename=path+'/'+setname+'_standarise_parameters'
np.savetxt(filename, norm_paras,  delimiter=',', fmt='%f')

test_features = std_by_paramters(test_features, 2, norm_paras)


hc_train_labels=hotcode_categorical(train_labels, 2)
hc_test_labels=hotcode_categorical(test_labels, 2)


model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(train_features.shape[1],)))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(train_features, hc_train_labels, batch_size=200, epochs=10)

score = model.evaluate(test_features, hc_test_labels)

print('\n', score)
print('\n', 'Test accuracy: ', score[1])


filename=path+'/'+setname+'_classification.h5'
model.save(filename)




allpixels = std_by_paramters(allpixels, 2, norm_paras)

#allpixels, norm_paras =std_datasets(allpixels, 2)
#filename=path+'/'+setname+'_standarise_parameters'
#np.savetxt(filename, norm_paras,  delimiter=',', fmt='%f')



print(allpixels.shape)

predictions=model.predict(allpixels)


print(predictions.shape, pnum, predictions.dtype)


outimage=predictions[:,1]+1


filename='MVAUI'
h, mvaui, pnum = ubm.load_envi_data_float(path, filename)
mvaui=mvaui[0]
waterpixels = np.where(mvaui>0.05)[0]



print(waterpixels, waterpixels.shape)
outimage[waterpixels]=0


filename = setname+'_by_dnn'
ubm.outputclsfile(outimage, path, h, filename, 4)
