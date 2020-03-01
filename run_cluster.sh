#!/bin/bash
# Indices available
#'MVAUI':    # MNDWI and Vegetation adjusted urban index
#'MSAVI':    # Modified Soil Adjusted Vegetation Index
#'MNDWI':    # Modified Normalised Difference Water Index

#'TSC_BRI':  # Tasseled Cap Brightness
#'NDBI':     # Normalised difference built-up index
#'SAVI':     # Soil adjusted vegetation index
#'NDTI':     # Normalised difference tillage index
#'NDWI':     # Normalised difference water index
#'WVAUI':    # NDWI and Vegetation adjusted urban index
#'DBSI':     # NDWI and Vegetation adjusted urban index
#'BSI':      # Building urban index
#'NDVI':     # Normalised difference vegetation index
#'BUI':      # Built-up index

# directory where the input indices data can be found
path='/g/data/u46/users/bt2744/work/data/urban_extent/tennant_ck/2016'

# file name of the brightness index for the clusters image
brightness='MVAUI'

# file name of the greenness index for the clusters image
greenness='MSAVI'

# file name of the wetness index for the clusters image
wetness='MNDWI'

# the number of clusters to generate
numclumps=6


python3 ${pwd}"indices_cluster_raw.py" $path $brightness $greenness $wetness $numclumps