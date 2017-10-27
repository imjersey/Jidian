
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
import os
import sys
pwd = os.getcwd()
sys.path.append("/root/Gan/jidian/MLexperiments")
sys.path.append("/root/Gan/jidian")
sys.path.append(pwd)
sys.path.append(os.path.dirname(os.getcwd()))

SAMPLE_LEN = MLexperiments.config.parameters.SAMPLE_LEN
SAMPLE_HEIGHT = MLexperiments.config.parameters.SAMPLE_HEIGHT
base_path = MLexperiments.config.parameters.DATA_PATH


def balanced_subsample(x,y,subsample_size=1.0):

    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)
    
    print("number of each type (balanced):" )
    print(use_elems)
    
    
    xs = []
    ys = []

    for ci,this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    print("total (balanced):" )
    print(len(ys))
    return xs,ys


def read_data_sets(train_dir=None, fake_data=False, one_hot=False, test_size=0.1, validation_size = 0.1, augment = False,mode = 1, data_balance = False):
    """
    mode: 1 => 3 human + rain: 1 means high only, 2 high + mid, 3 high+mid+low , 
      4=>6 man + rain + wind:      4 means high only, 5 high + mid, 6 high+mid+low , 
      7 => qinshang + factory for testing
    
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
    #factory
    man_input, man_label = load_data('../data/DataSet_10x100_HumanHighQty.mat',"DataSetHuman_Hqty")
    man_input2, man_label2 = load_data('../data/DataSet_10x100_HumanMidQty.mat',"DataSetHuman_Mqty")
    man_input3, man_label3 = load_data('../data/DataSet_10x100_HumanLowQty.mat',"DataSetHuman_Lqty")
    
    rain_input, rain_label = load_data('../data/DataSet_10x100_DSZRainiHighQty.mat',"DataSetDSZRain_Hqty")
    rain_input2, rain_label2 = load_data('../data/DataSet_10x100_DSZRainiMidQty.mat',"DataSetDSZRain_Mqty")
    rain_input3, rain_label3 = load_data('../data/DataSet_10x100_DSZRainiLowQty.mat',"DataSetDSZRain_Lqty")
    
    #qinshang
    qinshang_man_input, qinshang_man_label = load_data('../data/DataSet_10x100_HumanQinShanHighQty.mat',"DataSetHuman_Hqty")
    qinshang_man_input2, qinshang_man_label2 = load_data('../data/DataSet_10x100_HumanQinShanMidQty.mat',"DataSetHuman_Mqty")
    qinshang_man_input3, qinshang_man_label3 = load_data('../data/DataSet_10x100_HumanQinShanLowQty.mat',"DataSetHuman_Lqty")
    #rain_input, rain_label = load_data('../data/DataSet_10x100_DSZRainiHighQty.mat',"DataSetDSZRain_Hqty")
    #rain_input2, rain_label2 = load_data('../data/DataSet_10x100_DSZRainiMidQty.mat',"DataSetDSZRain_Mqty")
    #rain_input3, rain_label3 = load_data('../data/DataSet_10x100_DSZRainiLowQty.mat',"DataSetDSZRain_Lqty")
    
    
    #np_input, labelData = np.concatenate((man_input3, rain_input3)), np.concatenate((man_label3, rain_label3))
    np_input, labelData = rain_input, rain_label
    
#     if mode ==1:
#         np_input, labelData = np.concatenate((man_input, rain_input)), np.concatenate((man_label, rain_label))
#     else if mode == 2:
#         np_input, labelData = np.concatenate((man_input, rain_input,man_input2, rain_input2)), np.concatenate((man_label, rain_label,man_label2, rain_label2))
#     else if mode == 3:
#         np_input, labelData = np.concatenate((man_input, rain_input,man_input2, rain_input2,man_input3, rain_input3)), np.concatenate((man_label, rain_label,man_label2, rain_label2,man_label3, rain_label3))
#     else if mode == 4:
        
#     else if mode == 5:
        
#     else if mode == 6:

#     else if mode == 5:
        
#     else if mode == 6:

#     else if mode == 7:
        #no transfer, bad
        #np_input, labelData = np.concatenate((qinshang_man_input)), np.concatenate((qinshang_man_label))
#     else if mode == 8:
        
#     else if mode == 9:

#     else if mode == 10:
        
#     else if mode == 11:

#     else if mode == 12:
        
#     else if mode == 13:
        
#     else if mode == 14:

#     else if mode == 15:
        
#     else if mode == 16:

#     else if mode == 17:
        
#     else if mode == 18:
        
#     else if mode == 19:

#     else if mode == 20:
        
#     else if mode == 21:
        
        
    if data_balance:
        np_input , labelData = balanced_subsample(np_input,labelData,subsample_size=1.0)
    
    
    print(" np_input shape:  "+ str(np_input.shape) + " labelData shape:  "+str(labelData.shape))
    
    if augment:
        pass #todo[][]
    
    
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
    
    if fake_data:
        data_sets.train = DS.DataSet(train_images, train_labels, fake_data=False, one_hot=one_hot)
        data_sets.validation = DS.DataSet(validation_images, validation_labels, fake_data=True, one_hot=one_hot)
        data_sets.test = DS.DataSet(test_images, test_labels, fake_data=True, one_hot=one_hot)
        
    return data_sets


    