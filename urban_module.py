#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numpy import genfromtxt
from spectral import envi
import matplotlib.pyplot as plt


def urbclassify(pnum, bandnames, oneimg, clsarr, app, thd):
    if app == 1:
        tgtband = 'NDBI'
        # thd=-0.15
        ftidx = bandnames.index(tgtband)
        ftvals = oneimg[ftidx, :]
        for i in np.arange(pnum):
            cc = clsarr[i]
            val = ftvals[i]
            if cc == 1 and val > thd:
                clsarr[i] = 3
            elif cc == 3 and val < thd:
                clsarr[i] = 1
    elif app == 2:
        tgtband = 'BUI'
        # thd=-0.60
        ftidx = bandnames.index(tgtband)
        ftvals = oneimg[ftidx, :]
        for i in np.arange(pnum):
            cc = clsarr[i]
            val = ftvals[i]
            if cc == 1 and val > thd:
                clsarr[i] = 3
            elif cc == 3 and val < thd:
                clsarr[i] = 1
    elif app == 3:
        tgtband = 'NBUI'
        # thd=-0.680
        ftidx = bandnames.index(tgtband)
        ftvals = oneimg[ftidx, :]
        for i in np.arange(pnum):
            cc = clsarr[i]
            val = ftvals[i]
            if cc > 0 and val > thd:
                clsarr[i] = 3
            elif cc > 0 and val < thd:
                clsarr[i] = 1
    elif app == 4:
        tgtband = 'VAUI'
        # thd=-0.680
        ftidx = bandnames.index(tgtband)
        ftvals = oneimg[ftidx, :]
        for i in np.arange(pnum):
            cc = clsarr[i]
            val = ftvals[i]
            if cc > 0 and val > thd:
                clsarr[i] = 3
            elif cc > 0 and val < thd:
                clsarr[i] = 1
    elif app == 5:
        tgtband = 'MVAUI'
        # thd=-0.680
        ftidx = bandnames.index(tgtband)
        ftvals = oneimg[ftidx, :]
        for i in np.arange(pnum):
            cc = clsarr[i]
            val = ftvals[i]
            if cc > 0 and val > thd:
                clsarr[i] = 3
            elif cc > 0 and val < thd:
                clsarr[i] = 1
    return tgtband


def outputclsfile(clsarr, path, h, fstem, dtt):
    newhdrfile = path + '/' + fstem + '.hdr'
    newimgfile = path + '/' + fstem + '.img'
    h['band names'] = ['LC class']
    h['bands'] = 1
    h['data type'] = dtt
    h['description'] = 'Landcover change detection'
    envi.write_envi_header(newhdrfile, h)
    clsarr.tofile(newimgfile)


def outputenvifile(clsarr, path, h, fstem, bandnames, bands, datatype, description):
    newhdrfile = path + '/' + fstem + '.hdr'
    newimgfile = path + '/' + fstem + '.img'
    h['band names'] = bandnames
    h['bands'] = bands
    h['data type'] = datatype
    h['description'] = description
    envi.write_envi_header(newhdrfile, h)
    clsarr.tofile(newimgfile)


def classifyoneyear(path, appidx, thd):
    h, oneimg, pnum, bandnames, clsarr = load_data(path)
    tgtband = urbclassify(pnum, bandnames, oneimg, clsarr, appidx, thd)
    labelbare(clsarr, bandnames, 'DBSI', oneimg, 0.22)
    fstem = tgtband + '_adj_cls'
    outputclsfile(clsarr, path, h, fstem)
    return clsarr


def load_data(path):
    imgfile = path + '/indice.img'
    hdrfile = path + '/indice.hdr'
    h = envi.read_envi_header(hdrfile)
    irow = np.int32(h['lines'])
    icol = np.int32(h['samples'])
    ftbands = np.int32(h['bands'])
    oneimg = np.fromfile(imgfile, dtype=np.float32)
    oneimg = oneimg.reshape([ftbands, irow * icol])
    pnum = irow * icol
    bandnames = h['band names']
    clsfile = path + '/urban_spec_5c.img'
    clsarr = np.fromfile(clsfile, dtype=np.uint8)
    return h, oneimg, pnum, bandnames, clsarr


def load_envi_data_short(path, filename):
    imgfile = path + '/' + filename + '.img'
    hdrfile = path + '/' + filename + '.hdr'
    h = envi.read_envi_header(hdrfile)
    irow = np.int32(h['lines'])
    icol = np.int32(h['samples'])
    ftbands = np.int32(h['bands'])
    oneimg = np.fromfile(imgfile, dtype=np.int16)
    oneimg = oneimg.reshape([ftbands, irow * icol])
    pnum = irow * icol
    return h, oneimg, pnum


def load_envi_data_float(path, filename):
    imgfile = path + '/' + filename + '.img'
    hdrfile = path + '/' + filename + '.hdr'
    h = envi.read_envi_header(hdrfile)
    irow = np.int32(h['lines'])
    icol = np.int32(h['samples'])
    ftbands = np.int32(h['bands'])
    oneimg = np.fromfile(imgfile, dtype=np.float32)
    oneimg = oneimg.reshape([ftbands, irow * icol])
    pnum = irow * icol
    return h, oneimg, pnum


def load_envi_data_char(path, filename):
    imgfile = path + '/' + filename + '.img'
    hdrfile = path + '/' + filename + '.hdr'
    h = envi.read_envi_header(hdrfile)
    irow = np.int32(h['lines'])
    icol = np.int32(h['samples'])
    ftbands = np.int32(h['bands'])
    oneimg = np.fromfile(imgfile, dtype=np.int8)
    oneimg = oneimg.reshape([ftbands, irow * icol])
    pnum = irow * icol
    return h, oneimg, pnum


def labelinf(clsarr, bandnames, tgtband, oneimg, thd):
    ftidx = bandnames.index(tgtband)
    ftvals = oneimg[ftidx, :]
    fx = np.where(np.logical_and(clsarr == 3, ftvals < thd))[0]
    clsarr[fx] = 4
    return clsarr


def labelbare(clsarr, bandnames, tgtband, oneimg, thd):
    ftidx = bandnames.index(tgtband)
    ftvals = oneimg[ftidx, :]
    fx = np.where(np.logical_and(clsarr == 3, ftvals > thd))[0]
    clsarr[fx] = 2
    return clsarr


def draw_index_hist_index(bandlabel, oneimg, clsarr, num_bins):
    ftvals = oneimg
    urbanvals = ftvals[clsarr == 3]
    vegvals = ftvals[clsarr == 1]
    xlmin = ftvals.min()
    xrmax = ftvals.max()

    if xrmax != np.inf and xlmin != np.NINF:
        nC1, binsC1, patches = plt.hist(vegvals, num_bins, density=1, facecolor='g', alpha=0.85)
        nC3, binsC3, patches = plt.hist(urbanvals, num_bins, density=1, facecolor='r', alpha=0.85)

        rgh = [nC1.max(), nC3.max()]
        rgh = np.array(rgh)
        yrmax = rgh.max()

        plt.xlabel(bandlabel)
        plt.ylabel('Index Values')
        plt.title('Histogram of ' + bandlabel)
        plt.axis([xlmin, xrmax, 0, yrmax + 0.1])
        plt.grid(True)
        plt.show()


def draw_index_hist(tgtband, bandnames, oneimg, clsarr, num_bins):
    ftidx = bandnames.index(tgtband)
    ftvals = oneimg[ftidx, :]
    urbanvals = ftvals[clsarr == 3]
    vegvals = ftvals[clsarr == 1]
    xlmin = ftvals.min()
    xrmax = ftvals.max()

    if xrmax != np.inf and xlmin != np.NINF:
        nC1, binsC1, patches = plt.hist(vegvals, num_bins, density=1, facecolor='g', alpha=0.85)
        nC3, binsC3, patches = plt.hist(urbanvals, num_bins, density=1, facecolor='r', alpha=0.85)

        rgh = [nC1.max(), nC3.max()]
        rgh = np.array(rgh)
        yrmax = rgh.max()

        plt.xlabel(tgtband)
        plt.ylabel('Index Values')
        plt.title('Histogram of ' + tgtband)
        plt.axis([xlmin, xrmax, 0, yrmax + 0.1])
        plt.grid(True)
        plt.show()


def hist_distribution(tgtband, bandnames, oneimg, clsarr, numcls, num_bins):
    ftidx = bandnames.index(tgtband)
    ftvals = oneimg[ftidx, :]

    xlmin = ftvals.min()
    xrmax = ftvals.max()

    distarr = np.zeros((numcls, num_bins))
    binsize = (xrmax - xlmin) / num_bins
    bounds = np.zeros((num_bins, 2))

    for i in np.arange(num_bins):
        bounds[i, 0] = xlmin + i * binsize
        if i != num_bins - 1:
            bounds[i, 1] = xlmin + (i + 1) * binsize
        else:
            bounds[i, 1] = xrmax

    for i in np.arange(num_bins):
        if i != num_bins - 1:
            fbidx = np.where(np.logical_and(ftvals >= bounds[i, 0], ftvals < bounds[i, 1]))[0]
        else:
            fbidx = np.where(np.logical_and(ftvals >= bounds[i, 0], ftvals <= bounds[i, 1]))[0]

        items = clsarr[fbidx]
        for j in np.arange(numcls):
            fx = np.where(items == j)[0]
            distarr[i, j] = fx.shape[0]

    return bounds, distarr


def classify_by_mixture(mixtures, setname):
    irows = mixtures.shape[0]
    icol = mixtures.shape[1]
    newcls = np.zeros(irows, dtype=np.float32)
    if setname == '5classes':
        for i in np.arange(irows):
            vec = mixtures[i, :]
            if vec[0] > 0.5:
                newcls[i] = 0
            elif vec[1] > 0.7:
                newcls[i] = 1
            elif vec[2] > 0.7:
                newcls[i] = 2
            elif vec[4] > 0.7:
                newcls[i] = 4
            else:
                newcls[i] = 3

    elif setname == 'geomedian':
        for i in np.arange(irows):
            vec = mixtures[i, :]
            if vec[0] > 0.7:
                newcls[i] = 1
            elif vec[1] > 0.7:
                newcls[i] = 2
            elif vec[3] > 0.7:
                newcls[i] = 4
            else:
                newcls[i] = 3

    elif setname == 'mixture_model_sm':
        for i in np.arange(irows):
            vec = mixtures[i, :]
            flag = 0
            for j in np.arange(icol):
                if vec[j] > 0.8:
                    newcls[i] = j
                    flag = 1
                    break

            if flag == 0:
                rbi_rate = np.zeros(3, dtype=np.float32)
                rbi_rate[0] = vec[1] + 0.5 * (vec[4] + vec[6])
                rbi_rate[1] = vec[2] + 0.5 * (vec[4] + vec[5])
                rbi_rate[2] = vec[3] + 0.5 * (vec[5] + vec[6])

                ss = rbi_rate.sum()
                rbi_rate = rbi_rate / ss
                for pp in np.arange(3):
                    if rbi_rate[pp] > 0.8:
                        newcls[i] = pp + 1
                        flag = 1

                if flag == 0:
                    mms = rbi_rate.min()
                    if mms > 0.2:
                        newcls[i] = 7
                    elif rbi_rate[0] + rbi_rate[1] >= 0.8:
                        newcls[i] = 4
                    elif rbi_rate[1] + rbi_rate[2] >= 0.8:
                        newcls[i] = 5
                    elif rbi_rate[0] + rbi_rate[2] >= 0.8:
                        newcls[i] = 6

    elif setname == 'mixture_model_geomedian':
        for i in np.arange(irows):
            vec = mixtures[i, :]
            flag = 0
            for j in np.arange(icol):
                if vec[j] > 0.8:
                    newcls[i] = j + 1
                    flag = 1
                    break

            if flag == 0:
                rbi_rate = np.zeros(3, dtype=np.float32)
                rbi_rate[0] = vec[0] + 0.5 * (vec[3] + vec[5])
                rbi_rate[1] = vec[1] + 0.5 * (vec[3] + vec[4])
                rbi_rate[2] = vec[2] + 0.5 * (vec[4] + vec[5])

                ss = rbi_rate.sum()
                rbi_rate = rbi_rate / ss
                for pp in np.arange(3):
                    if rbi_rate[pp] > 0.8:
                        newcls[i] = pp + 1
                        flag = 1

                if flag == 0:
                    mms = rbi_rate.min()
                    if mms > 0.2:
                        newcls[i] = 7
                    elif rbi_rate[0] + rbi_rate[1] >= 0.8:
                        newcls[i] = 4
                    elif rbi_rate[1] + rbi_rate[2] >= 0.8:
                        newcls[i] = 5
                    elif rbi_rate[0] + rbi_rate[2] >= 0.8:
                        newcls[i] = 6

    return newcls
