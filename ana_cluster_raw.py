
# This program reads 3 Landsat surface reflectance-derived data, applies k-mean clustering algorithm to generate M clusters, which will be combined to form
# urban and non-urban classes in the next step in the work flow


import os
import sys
import sklearn as sk
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans



# directory where the input indices data can be found
path=sys.argv[1]

# directory where the header file for the clusters image can be found
sourcehdr=sys.argv[2]

# the number of clusters to generate 
numcls=int(sys.argv[3])


# file name of the Tasseled cap brightness data
bridatafile=path+'/phg_bri.img'

# file name of the modified soil adjusted vegetation index data
msavidatafile=path+'/phg_msavi.img'


# file name of the modified normalised difference waterindex data
mndwidatafile=path+'/phg_mndwi.img'

# file name of a csv file which stores the spatial dimensions (number of rows, number of columns) of the data  
parafile=path+'/ts_irow_icol.csv'

filelist=[bridatafile, msavidatafile, mndwidatafile]






tmm = pd.read_csv(parafile, header =None)

nrow=tmm.values[1:3][0][0]
ncol=tmm.values[1:3][1][0]
print (nrow, ncol)
pnum = nrow*ncol

data=np.zeros([pnum, 5])

rowdiv=20
coldiv=16

rlen=8
clen=8



# a function to read indices data file, extract the relevant band in the file
# put the data in the columns of an array as inputs for the clustering algorithm 

def readimgfile(filename, pnum, tgt, data, col):
    imgdata=np.fromfile(filename, dtype=np.float32)
    oneblock=imgdata[tgt*pnum:(tgt+1)*pnum]
    data[:, col]=imgdata[tgt*pnum:(tgt+1)*pnum]
    




tgt=4
col=0
for filename in filelist:
    readimgfile(filename, pnum, tgt, data, col)
    col=col+1




def classifyoneblock_spectral(data, nrow, ncol, rp, numcls):
    
    pnum=nrow*ncol
    
    greenness = data[:, 1]
    brightness = data[:, 0]
    wetness = data[:, 2]
   

    #exclude water pixels from input data

    wateridx=np.where(wetness>=0.02)[0]
    nonwateridx=np.where(wetness<0.02)[0]
    stidx = np.argsort(greenness)
    
    
    #load non-water pixels data as input for clustering algorithm
    sprst=data
    sprst=sprst[nonwateridx]
    sprst=np.asarray(sprst)


    # re-scale the data 
    sprst_scale=preprocessing.scale(sprst)
    
    width=2
    

    clscount=np.zeros([pnum, numcls], dtype=np.int32)

    for i in range(rp):
        clsimg = clusterclassifier(sprst_scale, pnum, numcls, wateridx, nonwateridx, brightness, greenness, wetness)
        for k in range(pnum):
            m=clsimg[k]
            clscount[k, m]=clscount[k,m]+1




    bbclsimg=clscount.argmax(axis=1)
    bbclsimg=bbclsimg.astype(np.int8)

 
    return bbclsimg






def clusterclassifier(scaled_data, pnum, numcls, wateridx, nonwateridx, brightness, greenness, wetness):
    
    clsimg = np.zeros(pnum, dtype=np.int8)
    clsimg[wateridx]=-1
    clustering=KMeans(init='k-means++', precompute_distances=True, algorithm="full", n_clusters=numcls-1)
    clustering.n_jobs=6
    labels=clustering.fit(scaled_data)
    clsimg[nonwateridx]=labels.labels_
    clsimg[:]=clsimg[:]+1
   
    cc =np.zeros(numcls)
    greencc=np.zeros(numcls)
    brightcc=np.zeros(numcls)
    wetcc=np.zeros(numcls)

    for i in range(pnum):
        clab=clsimg[i]
        greencc[clab]=greencc[clab]+greenness[i]
        brightcc[clab]=brightcc[clab]+brightness[i]
        wetcc[clab]=wetcc[clab]+wetness[i]
        cc[clab]=cc[clab]+1
        
    for i in range(numcls):
        if (cc[i]>0):
            greencc[i]=greencc[i]/cc[i]
            brightcc[i]=brightcc[i]/cc[i]
            wetcc[i]=wetcc[i]/cc[i]
    
    
    greencc[0]=1000
    stidx = np.argsort(greencc)


    maps=np.zeros(numcls, dtype=np.int32)
    for i in range(numcls):
        k=numcls-1-i
        maps[stidx[k]]=i
    
    for i in range(pnum):
        clsimg[i]=maps[clsimg[i]]
    
    return clsimg








rp=5

clsimg=classifyoneblock_spectral(data, nrow, ncol, rp, numcls)
clsimg=clsimg.astype(np.int8)



sprst=preprocessing.scale(data[:, 0:3])

cores=np.zeros([numcls,3])
dist=np.zeros([numcls,2])

  
for i in range(numcls):
    classidx=np.where(clsimg==i)[0]
    apm=sprst[classidx]
    cores[i, :]=apm.mean(axis=0)
 
  
for i in range(1, numcls):
    dist[i, 0]=np.sqrt(((cores[i, :]-cores[1,:])**2).sum(0))
    dist[i, 1]=np.sqrt(((cores[i, :]-cores[numcls-1,:])**2).sum(0))
  
  
mapcls=np.zeros(numcls)
mapcls=mapcls.astype(np.int8)

midpp=int(numcls/2)+1

for i in range(1, numcls):
    if (i<midpp):
        mapcls[i]=1
    else:
        mapcls[i]=3
      
          
imgfile = path+'/urban_spec_5c_raw.img'
clsimg.tofile(imgfile)

hdrfile = path+'/urban_spec_5c_raw.hdr'

commstr='cp '+sourcehdr+' '+hdrfile
os.system(commstr)



bbclsimg=mapcls[clsimg]
imgfile = path+'/urban_spec_5c.img'
bbclsimg.tofile(imgfile)



hdrfile = path+'/urban_spec_5c.hdr'

commstr='cp '+sourcehdr+' '+hdrfile

os.system(commstr)


