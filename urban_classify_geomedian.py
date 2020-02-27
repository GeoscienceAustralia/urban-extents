import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
from tensorflow.keras.datasets import mnist
import numpy as np
import sys
import os

from numpy import genfromtxt
from spectral import *

import urban_module as ubm


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


def combineclassify(clsstack, newcls):
    urban_arr=clsstack[0]
    bld_bare_arr=clsstack[1]
    bare_suburb_arr=clsstack[2]
    for i in np.arange(pnum):
        ux = urban_arr[i]
        bx = bld_bare_arr[i]
        sx = bare_suburb_arr[i] 
        if (ux==0):
            cls=0
        elif (ux<1.5):
            cls=ux
        elif (bx<1.5):
            cls=4
        elif (sx<1.5):
            cls=1+sx
        else:
            cls=1+sx 
        newcls[i]=cls

    return newcls
	    

def classifyoneimage(oneimage, modelstack, normstack, newcls):
    clsstack=[]
    ss=len(modelstack)

    scale=10000.0
    blue=oneimage[0, : ]
    green=oneimage[1, :]
    red=oneimage[2, :]
    nir=oneimage[3, :]
    swir1=oneimage[4, :]
    swir2=oneimage[5, :]

    print("Start calculating MVAUI")
    mvaui=cal_mvaui(blue, green, red, nir, swir1, swir2)

    print(mvaui)
    print(mvaui.shape)
    print("Start classifying ")
    for i in np.arange(ss):
        allpixels=np.copy(oneimage)
        allpixels=allpixels.transpose()
        model=modelstack[i]
        norm_paras=normstack[i]
        allpixels = std_by_paramters(allpixels, 2, norm_paras)
        predictions=model.predict(allpixels)
        outimage=predictions[:,1]+1
        clsstack.append(outimage)

    print("Start combining results from 3 classifiers")
	
    newcls = combineclassify(clsstack, newcls)
    waterpixels = np.where(mvaui>0.05)[0]
    print(waterpixels.shape)
    newcls[waterpixels]=0


    return newcls


    
def cal_mvaui(blue, green, red, nir, swir1, swir2):
    msavi = (2*nir+1-np.sqrt((2*nir+1)*(2*nir+1)-8*(nir-red)))/2
    mndwi = (green-swir1)/(green+swir1)
    mvaui = (red-swir1)/(red+swir1) + (swir2 - swir1)/(swir2+swir1) - msavi + mndwi

    return mvaui

def cal_dbsi(blue, green, red, nir, swir1, swir2):
    msavi = (2 * nir + 1 - np.sqrt((2 * nir + 1) * (2 * nir + 1) - 8 * (nir - red))) / 2
    dbsi = (swir1 - green) / (swir1 + green) - msavi	
    return dbsi

    
def cal_ndti(blue, green, red, nir, swir1, swir2):
    ndti = (swir1 - swir2) / (swir1 + swir2)
    return ndti
    
param=sys.argv



#dirc='/g/data1/u46/pjt554/urban_change_s2_full_sites/gold_coast'
modeldirc=param[1]
path=param[2]
setname=param[3]
outfilename=param[4]

hdrfile=path+'/NBAR_blue.hdr'
h=envi.read_envi_header(hdrfile)


irow=np.int32(h['lines'])
icol=np.int32(h['samples'])
pnum = irow*icol




bandnames=['NBAR_blue','NBAR_green', 'NBAR_red', 'NBAR_nir', 'NBAR_swir1', 'NBAR_swir2']
numbands=len(bandnames)
allpixels=np.zeros((pnum, numbands), dtype=np.float32)


for j, tgtband in enumerate(bandnames):
    filename=tgtband
    h, oneband, pnum = ubm.load_envi_data_float(path, filename)
    oneband=oneband[0]
    allpixels[:, j]=oneband


oneimage=allpixels.transpose()

blue=oneimage[0, : ]
green=oneimage[1, :]
red=oneimage[2, :]
nir=oneimage[3, :]
swir1=oneimage[4, :]
swir2=oneimage[5, :]

print("Start calculating MVAUI")
mvaui=cal_mvaui(blue, green, red, nir, swir1, swir2)
print("Start calculating DBSI")
dbsi=cal_dbsi(blue, green, red, nir, swir1, swir2)
print("Start calculating NDTI")
ndti=cal_ndti(blue, green, red, nir, swir1, swir2)



newcls=np.zeros(pnum, dtype=np.float32)




filename=modeldirc+'/'+setname+'_standarise_parameters'
norm_paras = np.loadtxt(filename)
modelfilename=modeldirc+'/'+setname+'_classification.h5'
model = models.load_model(modelfilename)

allpixels = std_by_paramters(allpixels, 2, norm_paras)
mixtures=model.predict(allpixels)



newcls=ubm.classify_by_mixture(mixtures, 'geomedian_v2')

waterpixels = np.where(mvaui>0.05)[0]
newcls[waterpixels]=0

#newcls.astype(np.uint8)

#identify false suburban classification 
fx=np.where(np.logical_and(newcls==3.0, dbsi>0.28))[0]

newcls[fx]=5

fx=np.where(np.logical_and(newcls==3.0, dbsi+ndti>0.40))[0]

newcls[fx]=6

filename=outfilename+'_urban_map_raw'
ubm.outputclsfile(newcls, path, h, filename, 4)

#remove false suburban classification
newcls[newcls==5]=1
newcls[newcls==6]=1


filename=outfilename+'_urban_map'
ubm.outputclsfile(newcls, path, h, filename, 4)


filename=outfilename+'_mixture'
mixtures=mixtures.transpose()

bandnames='{veg, bare, sub, bld}'
description='fraction of ground objects'
ubm.outputenvifile(mixtures, path, h, filename, bandnames, 4, 4, description)


#outimage=mixtures.transpose()
#print(outimage.shape)
#print(outimage[0, :])

#filename = setname+'_by_dnn'

#if setname=='5classes':
#    outbandnames=['water', 'vegetation', 'bare soil', 'suburban mixtures', 'buildings']

#if setname=='4classes':
#    outbandnames=['water', 'vegetation', 'bare soil', 'buildings']

#if setname=='mixture_model':
#    outbandnames=['water', 'vegetation', 'bare soil', 'buildings', 'veg and bare', 'building mixture']

#if setname=='mixture_model_sm':
#    outbandnames=['water', 'vegetation', 'bare soil', 'buildings', 'veg and bare', 'bare and building', 'veg and buidling', 'building mixture']

#ubm.outputenvifile(outimage, path, h, filename, outbandnames, numcls, 4, 'Ground objects fraction')
        





