# Urban extent and urban development detection using Landsat time series data

The codes in this repository will take user inputs from users, generate scripts to conduct urban extent and urban development detection for a user specified area. All pixels within the targeted area will be classified as one of water, non-urban d urban classes on an annual base.  

To compile the code:

clone the repo
cd code_dir && module load gsl && make all

This will compile the source code and create all executable programs. The code_dir is the directory where the source codes are saved. 
The next step is to create file structure to accommodate time series data and scripts. The file structures are organised as follow. 

data_dir/local_sub1
data_dir/local_sub2
…
data_dir/local_subx

The data and scripts for each targeted area are located at subdirectory data_dir/local_subx, all local_subx directories are under directory data_dir. As such, user should create directory data_dir and ensure write privilege is granted under data_dir before generating the scripts. 
The subdirectories are created automatically when user generate scripts for targeted areas. 

To generate bash shell scripts for urban extents detection for a targeted area, type 
 
code_dir/vdi_urban_scripts code_dir data_dir local_subx lat_top lat_bottom lon_left lon_right first_year last_year num_cls

The program will create all scripts for urban extents and urban development detection for the targeted area and save them to directory data_dir/local_subx.  The input parameters required to generate the scripts are described in table 1. 

_Table 1_

Parameter|Description
--- | ---
code_dir   | the directory where the programs locate, i.e., where the downloaded source codes are saved, it should be in form of full directory, e.g., /path_to/../../code_dir
data_dir	  | the directory where the data will be saved, , it should be in form of full directory, e.g., /path_to/../../code_dir
local_subx	| the name of the sub directory where the data and scripts will be saved
lat_top	| the latitude of the top left corner of the targeted area
lat_bottom	| the latitude of the right bottom corner of the targeted area
lon_left	| the longitude of the top left corner of the targeted area
lon_right	| the longitude of the right bottom corner of the targeted area
first_year	| the year of the start date of the time series 
last_year	| the year of the end date of the time series
num_cls	| number of classes for the k-mean clustering algorithm, normally should be set  to 6 


_Table 2_

Script name	| Description
--- | ---
load_landsat_data_xxx.sh	| Load NBAR-t time series of 6 Landsat spectral bands (‘blue’, ‘green’, ‘red’, ‘nir’,’swir1’,’swir2’
create_tsmask_xxx.sh	| Create cloud and cloud shadow mask using time series time series noise detection algorithm
create_indices_xxx.sh	| Calculate 3 spectral indices (Tasseled cap brightness, modified soil adjusted vegetation index, modified normalised difference water index) from NBAR-t data
create_clusters_xxx.sh	| Create clusters with 3 spectral indices as inputs
map_raw_class_xxx.sh	| Classify clusters into vegetation and non-vegetation classes
urban_change_xxx.sh	| Time series analysis for urban change detection
urban_detection_run_all_xxx.sh	| A script wrapping all above scripts in sequence, so user only need to run this script to complete the task

