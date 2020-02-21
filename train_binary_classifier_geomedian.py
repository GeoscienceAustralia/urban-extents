import numpy as np
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
from spectral import *
import urban_module as ubm
import sys
# TODO: Many of these functions can be replaced by standard numpy and sklearn ones, some are unused
from train_binary_classifier_geomedian_functions import load_datafile_pair, hotcode_categorical, \
    std_datasets, std_by_paramters, find_feature_index, calc_std_paras

param = sys.argv

path = param[1]
setname = param[2]

filename = path + '/' + setname + '_feature_list'
featurelist = np.loadtxt(filename, delimiter=',', dtype='str')

numfte = featurelist.shape[0]
print('number of features =', numfte)

# Training section

x_train_filename = setname + '_train_features'
y_train_filename = setname + '_train_labels'

# Load data and create test train split
train_features, train_labels = load_datafile_pair(path, x_train_filename, y_train_filename, numfte)

x_test_filename = setname + '_test_features'
y_test_filename = setname + '_test_labels'

test_features, test_labels = load_datafile_pair(path, x_test_filename, y_test_filename, numfte)

# TODO: Confirm with Peter after running file that commented code can be removed

train_features, norm_paras = std_datasets(train_features, 2)

# Save out normalisation parameters
filename = path + '/' + setname + '_standarise_parameters'
np.savetxt(filename, norm_paras, delimiter=',', fmt='%f')

test_features = std_by_paramters(test_features, 2, norm_paras)

# TODO: Replace Peter's function with sklearn.prepocessing.OneHotEncoder
hc_train_labels = hotcode_categorical(train_labels, 2)
hc_test_labels = hotcode_categorical(test_labels, 2)

# Initialise model architecture
model = models.Sequential()
# Add layers
# Only the first layer needs information regarding the shape
# The following layers automatically infer shape
model.add(layers.Dense(256, activation='relu', input_shape=(train_features.shape[1],)))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))

# Configure the learning process through compilation
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# TODO: Use Sklearn gridsearchcv to explore parameters
# Fit model
model.fit(train_features, hc_train_labels, batch_size=200, epochs=10)

# Evaluate model and print scores
# TODO: Why is this giving a test accuracy of 1?
#  Model might not be overfitting could be due to 'easy to learn' test data
#
score = model.evaluate(test_features, hc_test_labels)
print('\n', score)
print('\n', 'Test accuracy: ', score[1])

# Save model file
filename = path + '/' + setname + '_classification.h5'
model.save(filename)

# Prediction

# Read in classification created by clustering
hdrfile = path + '/urban_spec_5c.hdr'
h = envi.read_envi_header(hdrfile)
irow = np.int32(h['lines'])
icol = np.int32(h['samples'])
# This is the important part
pnum = irow * icol

# Create empty array
bandnames = featurelist
numbands = len(bandnames)
allpixels = np.zeros((pnum, numbands), dtype=np.float32)

# Load ENVI data into numpy array, this data is not used in training
# TODO: This is for PREDICTION, move into seperate file
# TODO: Process will be redundant if using multiband format (geotiff) or keeping in memory
j = 0
for tgtband in bandnames:
    filename = tgtband
    h, oneband, pnum = ubm.load_envi_data_float(path, filename)
    oneband = oneband[0]
    allpixels[:, j] = oneband
    j = j + 1

# Normalise by previously saved parameters to prepare for prediction
allpixels = std_by_paramters(allpixels, 2, norm_paras)

print(allpixels.shape)

# Prediction on unseen data
predictions = model.predict(allpixels)

print(predictions.shape, pnum, predictions.dtype)

outimage = predictions[:, 1] + 1

filename = 'MVAUI'
h, mvaui, pnum = ubm.load_envi_data_float(path, filename)
mvaui = mvaui[0]
waterpixels = np.where(mvaui > 0.05)[0]

print(waterpixels, waterpixels.shape)
outimage[waterpixels] = 0

# Save predicted data
filename = setname + '_by_dnn'
ubm.outputclsfile(outimage, path, h, filename, 4)
