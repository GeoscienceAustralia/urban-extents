#!/bin/bash

# Output directory
dir='/g/data/u46/users/sc0554/LCCS/urban/alice_springs'


# Change these to select a different area to train on
########################

# Alice Springs
lat_top=-23.66
lat_bottom=-23.86
lon_left=133.70
lon_right=133.9
sensor='ls8'
year='2018'

# Tennant Creek
#lat_top=-19.5
#lat_bottom=-19.8
#lon_left=134
#lon_right=134.3
#sensor='ls8'
#year='2016'

# Canberra
# lat_top=-35.11
# lat_bottom=-35.35
# lon_left=149.02
# lon_right=149.22
# sensor='ls8'
# year='2017'
# dirc='/g/data1a/u46/pjt554/urban_geomedian_data/canberra'

#test spot
#lat_top=-29
#lat_bottom=-29.5
#lon_left=131.5
#lon_right=132
#sensor='ls8'
#year='2018'


python3 ${pwd}"load_geomedian_data.py" $lat_top $lat_bottom $lon_left $lon_right $sensor $year $dir