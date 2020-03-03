#!/usr/bin/env python

# Import modules
import datacube
import sys
import numpy as np
import time
import os
from spectral import envi
import rasterio
from dea_sartools import sarcube

argc = len(sys.argv)
print(argc)

if argc != 9:
    print(
        'Usage: python3 load_geomedian_data.py lat_top lat_bottom lon_left lon_right sensor year output_dir '
        'load_sar_data_or_not')
    sys.exit()

# Set up spatial and temporal query; note that 'output_crs' and 'resolution' need to be set
param = sys.argv

lat_top = float(param[1])
lat_bottom = float(param[2])
lon_left = float(param[3])
lon_right = float(param[4])
sensor = param[5]
year = param[6]
dirc = param[7]
sar = param[8]

comm = 'mkdir ' + dirc
os.system(comm)
start_of_epoch = year + '-01-01'
end_of_epoch = year + '-12-31'

newquery = {'x': (lon_left, lon_right),
            'y': (lat_top, lat_bottom),
            'time': (start_of_epoch, end_of_epoch),
            'output_crs': 'EPSG:3577',
            'resolution': (-25, 25)}

dc = datacube.Datacube(app="geomedian")
data = dc.load(product=sensor + '_nbart_geomedian_annual', **newquery)
sdevdata = dc.load(product=sensor + '_nbart_tmad_annual', **newquery)


def write_single_band_dataarray(timebandnaes, filename, dataarray, **profile_override):
    profile = {
        'width': len(dataarray[dataarray.crs.dimensions[1]]),
        'height': len(dataarray[dataarray.crs.dimensions[0]]),
        'transform': dataarray.affine,
        'crs': dataarray.crs.crs_str,
        'count': len(dataarray.time),
        'dtype': str(dataarray.dtype)
    }
    profile.update(profile_override)
    dest = rasterio.open(str(filename), 'w', **profile)
    dest.write(dataarray.data[0], 1)
    dest.close()


allbands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'sdev']
output_bandnames = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'sdev']

path = dirc + '/' + year
comm = 'mkdir ' + path
os.system(comm)

yearstr = ['year ' + year]
yearstr = np.asarray(yearstr)

meanstack = []
scale = np.float32(10000.0)

for cc, bandname in enumerate(allbands):
    if (bandname != 'sdev'):
        banddata = data[bandname]
    else:
        banddata = sdevdata[bandname]
    outbandname = output_bandnames[cc]
    filename = path + '/NBAR_' + outbandname + '.img'
    hdrfilename = path + '/NBAR_' + outbandname + '.hdr'
    write_single_band_dataarray(yearstr, filename, banddata, driver='ENVI')
    h = envi.read_envi_header(hdrfilename)
    h['band names'] = '{' + bandname + ' band year ' + year + '}'
    h['data type'] = 4
    envi.write_envi_header(hdrfilename, h)
    dev = banddata.data
    dev = dev[0]
    if (bandname != 'sdev'):
        dev = dev / scale
        meanstack.append(dev)

    dev.tofile(filename)


def cal_indices(indstr, datastack):
    blue = datastack[0]
    green = datastack[1]
    red = datastack[2]
    nir = datastack[3]
    swir1 = datastack[4]
    swir2 = datastack[5]

    if indstr == 'TSC_BRI':  # Taseeled Cap Brightness
        indval = 0.3037 * blue + 0.2793 * green + 0.4743 * red + 0.5585 * nir + 0.5082 * swir1 + 0.1863 * swir2
    elif indstr == 'MSAVI':  # Modified Soil Adjusted Vegetation Index
        indval = (2 * nir + 1 - np.sqrt((2 * nir + 1) * (2 * nir + 1) - 8 * (nir - red))) / 2
    elif indstr == 'MNDWI':  # Modified Normalised Difference Water Index
        indval = (green - swir1) / (green + swir1)
    elif indstr == 'NDBI':  # Normalised difference built-up index
        indval = (swir1 - nir) / (swir1 + swir2)
    elif indstr == "SAVI":  # Soil adjusted vegetation index
        L = np.float32(0.5)
        indval = ((nir - red) / (nir + red + L)) * (1 + L)
    elif indstr == 'NDTI':  # Normalised difference tillage index
        indval = (swir1 - swir2) / (swir1 + swir2)
    elif indstr == 'NDWI':  # Normalised difference water index
        indval = (green - nir) / (green + nir)
    elif indstr == 'MVAUI':  # MNDWI and Vegetation adjusted urban index
        msavi = (2 * nir + 1 - np.sqrt((2 * nir + 1) * (2 * nir + 1) - 8 * (nir - red))) / 2
        mndwi = (green - swir1) / (green + swir1)
        indval = (red - swir1) / (red + swir1) + (swir2 - swir1) / (swir2 + swir1) - msavi + mndwi
    elif indstr == 'WVAUI':  # NDWI and Vegetation adjusted urban index
        msavi = (2 * nir + 1 - np.sqrt((2 * nir + 1) * (2 * nir + 1) - 8 * (nir - red))) / 2
        ndwi = (green - nir) / (green + nir)
        indval = (red - swir1) / (red + swir1) + (swir2 - swir1) / (swir2 + swir1) - msavi + ndwi
    elif indstr == 'DBSI':  # NDWI and Vegetation adjusted urban index
        msavi = (2 * nir + 1 - np.sqrt((2 * nir + 1) * (2 * nir + 1) - 8 * (nir - red))) / 2
        indval = (swir1 - green) / (swir1 + green) - msavi
    elif indstr == 'BSI':  # Building urban index
        indval = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue))
    elif indstr == 'NDVI':  # Normalised difference vegetation index
        indval = (nir - red) / (nir + red)
    elif indstr == 'BUI':  # Built-up index
        ndbi = (swir1 - nir) / (swir1 + swir2)
        ndvi = (nir - red) / (nir + red)
        indval = ndbi - ndvi
    else:
        indval = (blue + green + red + nir + swir1 + swir2) / 6

    return indval


def write_image_envi(path, bandname, dev, h):
    hdrfile = path + '/' + bandname + '.hdr'
    imgfile = path + '/' + bandname + '.img'
    h['bands'] = 1
    h['description'] = '{' + bandname + '}'
    h['band names'] = '{' + bandname + '}'

    envi.write_envi_header(hdrfile, h)
    dev.tofile(imgfile)


h['data type'] = 4

indiceslist = ['TSC_BRI', 'MSAVI', 'MNDWI', 'NDVI', 'NDTI', 'DBSI', 'BUI', 'SAVI',
               'VAUI', 'NDWI', 'MVAUI', 'WVAUI', 'BSI']

for indstr in indiceslist:
    indval = cal_indices(indstr, meanstack)
    write_image_envi(path, indstr, indval, h)


# Load SAR data
def write_bandstats_envi(path, bandname, sp, dev, h):
    if sp == 1:
        mks = np.nanmean(dev, axis=0)
    elif sp == 2:
        mks = np.nanstd(dev, axis=0)
    elif sp == 3:
        mks = np.nanmax(dev, axis=0) - np.nanmin(dev, axis=0)

    hdrfile = path + '/' + bandname + '.hdr'
    imgfile = path + '/' + bandname + '.img'
    h['bands'] = 1
    h['description'] = bandname
    h['band names'] = '{' + bandname + '}'

    envi.write_envi_header(hdrfile, h)
    mks.tofile(imgfile)
    return mks


if (sar == '1'):
    dc_sar = sarcube(app="Sentinel_1", config='radar.conf')
    sarbands = ['vv', 'vh', 'lia']

    data = dc_sar.load(product='s1_gamma0_scene', group_by='solar_day', db=False, **newquery)

    vv = data['vv'].data
    vh = data['vh'].data

    vvbnames = ['vv_mean', 'vv_std', 'vv_range']
    vhbnames = ['vh_mean', 'vh_std', 'vh_range']

    for sp in np.arange(3):
        aa = write_bandstats_envi(path, vvbnames[sp], sp + 1, vv, h)
        aa = write_bandstats_envi(path, vhbnames[sp], sp + 1, vh, h)
