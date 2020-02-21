# urban-extents
This repository uses data from the Digital Earth Australia Open Data Cube to classify urban extents using a combination unsupervised and supervised neural network approach.

# Training
`load_geomedian_data.py` - Extract data <br>
`indices_cluster_raw.py` - Unsupervised clustering <br>
`geomedian_training_step_1.ipynb` - Training data cleaning and preparation <br>
`train_binary_classifier_geomedian.py` - Train neural network <br>

# Classification
`get_geomedian_as_envi.py` - Runs a datacube query and saves the output as an ENVI file. <br>
`urban_classify_geomedian.py` - Apply trained model to classify, takes ENVI files as input <br>
`urban_classify_geomedian_mixture.py` - Apply model as above but using mixture approach <br>
