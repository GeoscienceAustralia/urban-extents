import os
import gdal
import numpy as np

import datacube
from datacube.storage import masking
from datacube.helpers import write_geotiff
from datacube.utils.geometry import CRS

dc = datacube.Datacube(app="geomedian")

# Kalgoorlie
x = (-1002450,  -972375)
y = (-3402100, -3366125)

# Perth
# x = (-1550000, -1450000)
# y = (-3650000, -3550000)

# 'Diamantina': 
# x =  (800000, 900000)
# y = (-2800000,-2700000)

# Hobart
x = (1200000, 1300000)
y = (-4800000, -4700000)

# Adelaide
# x = (550000, 650000)
# y = (-3850000, -3750000)

# Canberra
# x = (148.985, 149.284)
# y = (-35.144, -35.505) 

# Sydney

# x = (1680000, 1750000)
# y = (-3850000, -3780000)

# 'Ayr': 
# x = (1500000,1600000)
# y = (-2200000,-2100000)

res = (-25, 25)
crs = "EPSG:3577"
time = ("2015-01-01", "2015-12-31")
sensor = 'ls8'
query =({'x':x,
        'y':y,
        'crs':crs,
        'resolution':res})

data = dc.load(product=sensor + "_nbart_geomedian_annual", time=time, **query)
location = "Hobart"
year = "2015"
suffix = '.tif'
for var in data.data_vars:
    out = data[var]
    out = out.to_dataset(name=var)
#     out *= 1.0/10000
    out = out.astype(np.float32)
    out.attrs['crs']=CRS(data.crs)
    out = out.drop('time').isel(time=0)
    name = f'{location}_{year}_{var}{suffix}'
    write_geotiff(name, out)
    name_out = f'{location}_{year}_{var}.img'
    envi = gdal.Open(name)
    gdal.Translate(name_out, envi, format="ENVI")
