from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import numpy as np
from numpy import *
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import os
import gzip
import os
import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import MLexperiments.classes.DataSet as DS
import MLexperiments.config.parameters
SAMPLE_LEN = MLexperiments.config.parameters.SAMPLE_LEN
SAMPLE_HEIGHT = MLexperiments.config.parameters.SAMPLE_HEIGHT
base_path = MLexperiments.config.parameters.DATA_PATH

def read_data_sets(train_dir=None, fake_data=False, one_hot=False, test_size=0.33, validation_size = 0.1):
    class DataSets(object):
        pass

    data_sets = DataSets()

    if fake_data:
        data_sets.train = DataSet([], [], fake_data=True, one_hot=one_hot)
        data_sets.validation = DataSet([], [], fake_data=True, one_hot=one_hot)
        data_sets.test = DataSet([], [], fake_data=True, one_hot=one_hot)
        return data_sets




    original_path = os.path.join(base_path, "original")
    plots_path = os.path.join(base_path, "plots")
    csv_path = os.path.join(base_path, "csv")
    interfer_path = os.path.join(base_path, "interfer")
    label_path = os.path.join(base_path, "label")
    CNNinput_path = os.path.join(base_path, "CNNinput")

    save_name = os.path.join(CNNinput_path, "CNNinput.npy")

    np_input = np.load(save_name)

    np_input = np_input.reshape((int(np_input.size / (SAMPLE_LEN * SAMPLE_HEIGHT)), SAMPLE_HEIGHT, SAMPLE_LEN))

    save_name = os.path.join(label_path, "label.npy")
    labelData = np.load(save_name)
    labelData = np.transpose(np.array([labelData, ~labelData], dtype = int))



    print("read succesful")
    train_images, test_images, train_labels, test_labels = train_test_split(np_input, labelData, test_size=test_size,
                                                                            random_state=42)
    VALIDATION_SIZE = int(train_labels.size * validation_size)
    validation_images = train_images[:VALIDATION_SIZE]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_images = train_images[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]

    data_sets.train = DS.DataSet(train_images, train_labels)
    data_sets.validation = DS.DataSet(validation_images, validation_labels)
    data_sets.test = DS.DataSet(test_images, test_labels)

    return data_sets
