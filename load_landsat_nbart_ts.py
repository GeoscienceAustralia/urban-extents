# coding: utf-8
# Load sentinel time series data, save the data as ENVI images

import datacube
import sys
import numpy as np
import time
import os
import rasterio

# Import dea-notebooks functions using relative link to Scripts directory
# sys.path.append('/g/data1/u46/pjt554/dea_notebooks/dea-notebooks-master/10_Scripts')
import DEADataHandling

def write_multi_time_dataarray(filename, dataarray, **profile_override):
    profile = {
        'width': len(dataarray[dataarray.crs.dimensions[1]]),
        'height': len(dataarray[dataarray.crs.dimensions[0]]),
        'transform': dataarray.affine,
        'crs': dataarray.crs.crs_str,
        'count': len(dataarray.time),
        'dtype': str(dataarray.dtype)
    }
    profile.update(profile_override)

    with rasterio.open(str(filename), 'w', **profile) as dest:
        for time_idx in range(len(dataarray.time)):
            bandnum = time_idx + 1
            dest.write(dataarray.isel(time=time_idx).data, bandnum)


# Connect to a datacube
dc = datacube.Datacube(app='Clear Landsat')
argc = len(sys.argv)

if (argc != 8):
    print('Usage: python dea_fetch_data.py lat_top lat_bottom lon_left lon_right start_of_epoch(yyyy-mm-dd) end_of_epoch(yyyy-mm-dd) output_dir')
    sys.exit() 

# Set up spatial and temporal query; note that 'output_crs' and 'resolution' need to be set
param = sys.argv

lat_top = float(param[1]) # the latitude of the top left corner of the targeted area
lat_bottom = float(param[2]) # the latitude of the right bottom corner of the targeted area
lon_left = float(param[3]) # the longitude of the top left corner of the targeted area
lon_right = float(param[4])  # the longitude of the right bottom corner of the targeted area
start_of_epoch = param[5] # the earliest possible date of the time series query
end_of_epoch = param[6] # the latest possible date of the time series query
dirc = param[7] # the directory where the data will be saved

#create the directory where the data will be saved
comm = 'mkdir ' + dirc
os.system(comm)

query={'x': (lon_left, lon_right),
       'y': (lat_top, lat_bottom),
       'time': (start_of_epoch, end_of_epoch)
       }

allbands=['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
output_bandnames = ['blue','green', 'red', 'nir', 'swir1', 'swir2']

cc=0
# Load observations from both S2A and S2B as a single combined dataset
for bandname in allbands:
    
    inbandlist=[]
    inbandlist.append(bandname)
    print(inbandlist)
    landsat_ds = DEADataHandling.load_clearlandsat(dc=dc, query=query, sensors=('ls5', 'ls7', 'ls8'), product='nbart',
                                       bands_of_interest=inbandlist, ls7_slc_off=True,
                                       mask_pixel_quality=False, mask_invalid_data=False, masked_prop=0.0)
    
    timelist = []
    for t in landsat_ds['time']:
        timelist.append(time.strftime("%Y-%m-%d", time.gmtime(t.astype(int)/1000000000)))

    timebandnames = np.asarray(timelist)
    banddata = landsat_ds[bandname]
    outbandname = output_bandnames[cc]
    filename=dirc+'/NBAR_'+outbandname+'.img'
    write_multi_time_dataarray(filename, banddata, driver='ENVI')
    cc=cc+1

# function to create an ENVI header file
def create_envi_header(fname, icol, irow, ts, lon_left, lat_top, description, bandnames):
    hdr_file = open(fname, "w")
    hdr_file.write("ENVI\n")
    outstr = 'description = { ' + description + '}\n'
    hdr_file.write(outstr)
    outstr = 'samples = '+str(icol)+'\n'
    hdr_file.write(outstr)
    outstr = 'lines = '+str(irow)+'\n'
    hdr_file.write(outstr)
    outstr = 'bands = '+str(ts)+'\n'
    hdr_file.write(outstr)
    hdr_file.write("header offset = 0\n")
    hdr_file.write("file type = ENVI Standard\n")
    hdr_file.write("data type = 1\n")
    hdr_file.write("interleave = bsq\n")
    hdr_file.write("sensor type = Unknown\n")
    hdr_file.write("byte order = 0\n")
    outstr = 'map info = { Geographic Lat/Lon, 1, 1, '+str(format(lon_left, '.2f'))+ ', ' +str(format(lat_top, '.2f'))
    outstr = outstr + ', 0.00025, 0.00025, WGS-84 }\n'
    hdr_file.write(outstr)
    hdr_file.write("wavelength units =\n")
    hdr_file.write("band names ={\n")

    cc = 0
    for bandstr in bandnames:
        if (cc != ts - 1):
            outstr = bandstr + ',\n'
        else:
            outstr = bandstr + '\n'

        hdr_file.write(outstr)
        cc = cc + 1

    hdr_file.write('}\n')    
    hdr_file.close()


    
fname = dirc + '/clouds.hdr'

xs = landsat_ds['x'].size
ys = landsat_ds['y'].size
ts = landsat_ds['time'].size
des = 'cloud = 3, non_cloud = 0'
lon_left = 1.0
lat_top = 1.0

# create an ENVI header file with dates of time series as bandnames 
create_envi_header(fname, xs, ys, ts, lon_left, lat_top, des, timebandnames)



# write the number of time series bands, the number of rows and columns of the images into a text file  
amm = [ts, ys, xs]
fname=dirc+'/ts_irow_icol.csv'
np.savetxt(fname, amm, fmt='%d', delimiter=', ', newline='\n', header='', footer='', comments='# ')

