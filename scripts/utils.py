
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import scipy.io
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


def read_data_sets(train_dir=None, fake_data=False, one_hot=False, test_size=0.1, validation_size = 0.1, augment = False,mode = 1):
    """
    mode: 1 => 3 human + rain: 1 means high only, 2 high + mid, 3 high+mid+low , 
      4=>6 man + rain + wind:      4 means high only, 5 high + mid, 6 high+mid+low , 
    
    """
    class DataSets(object):
        pass
        
    def load_data(_path,key):
        mat = scipy.io.loadmat(_path)

        signals = mat[key][0]['data']
        label = mat[key][0]['label']
        m_signals = np.array([item.T for item in signals])
        m_label = np.array([item[0][0] for item in label])

        return m_signals, m_label
    #print(" m_label shape:  "+ str(m_label.shape) + " signal shape:  "+str(m_signals.shape))
    
    
    np_input, labelData = [],[]
    
    data_sets = DataSets()
    
    man_input, man_label = load_data('../data/DataSet_10x100_HumanHighQty.mat',"DataSetHuman_Hqty")
    man_input2, man_label2 = load_data('../data/DataSet_10x100_HumanMidQty.mat',"DataSetHuman_Mqty")
    man_input3, man_label3 = load_data('../data/DataSet_10x100_HumanLowQty.mat',"DataSetHuman_Lqty")
    
    rain_input, rain_label = load_data('../data/DataSet_10x100_DSZRainiHighQty.mat',"DataSetDSZRain_Hqty")
    rain_input2, rain_label2 = load_data('../data/DataSet_10x100_DSZRainiMidQty.mat',"DataSetDSZRain_Mqty")
    rain_input3, rain_label3 = load_data('../data/DataSet_10x100_DSZRainiLowQty.mat',"DataSetDSZRain_Lqty")
    
    np_input, labelData = np.concatenate((man_input, rain_input,man_input2, rain_input2)), np.concatenate((man_label, rain_label,man_label2, rain_label2))
    
    
    # if mode ==1:
    #     np_input, labelData = np.concatenate((man_input, rain_input)), np.concatenate((man_label, rain_label))
#     else if mode == 2:
#         np_input, labelData = np.concatenate((man_input, rain_input,man_input2, rain_input2)), np.concatenate((man_label, rain_label,man_label2, rain_label2))
#     else if mode == 3:
        
#     else if mode == 4:
        
#     else if mode == 5:
        
#     else if mode == 6:
        
        
        
    print(" np_input shape:  "+ str(np_input.shape) + " labelData shape:  "+str(labelData.shape))
    
    if augment:
        pass #todo[][]
    
    
    
    if fake_data:
        data_sets.train = DataSet([], [], fake_data=True, one_hot=one_hot)
        data_sets.validation = DataSet([], [], fake_data=True, one_hot=one_hot)
        data_sets.test = DataSet([], [], fake_data=True, one_hot=one_hot)
        return data_sets
    
    
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


    