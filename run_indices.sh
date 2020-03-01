#!/bin/bash

# Output directory
dir='/g/data/u46/users/bt2744/work/data/urban_extent/tennant_ck'


# Change these to select a different area to train on
########################

# Alice Springs
#lat_top=-23.66
#lat_bottom=-23.86
#lon_left=149.75
#lon_right=149.95
#sensor='ls8'
#year='2016'

# Tennant Creek
lat_top=-19.5
lat_bottom=-19.8
lon_left=134
lon_right=134.3
sensor='ls8'
year='2016'


python3 ${pwd}"load_geomedian_data.py" $lat_top $lat_bottom $lon_left $lon_right $sensor $year $dir