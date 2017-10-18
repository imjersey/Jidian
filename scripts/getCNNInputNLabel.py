"""
===================================================
auto generate data segment and label based on interference

===================================================
"""


import getopt
import sys
import os
import math
import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import copy

sample_len = 200
sample_height = 6
gate = 2000


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def get_immediate_files(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isfile(os.path.join(a_dir, name))]


base_path = MLexperiments.config.parameters.DATA_PATH
original_path = os.path.join(base_path, "original")
plots_path = os.path.join(base_path, "plots")
csv_path = os.path.join(base_path, "csv")
interfer_path = os.path.join(base_path, "interfer")
label_path = os.path.join(base_path, "label")
CNNinput_path = os.path.join(base_path, "CNNinput")


def getCNNInput():
    file_names = get_immediate_files(csv_path)
    input_data = []
    for file_name in file_names:
        file_path = os.path.join(csv_path, file_name)
        data = np.genfromtxt(file_path, dtype=float, delimiter=',')
        n = data.shape[0]
        m = data.shape[1]
        n1 = int(n/sample_height)
        m1 = int(m/sample_len)
        for i in range(n1):
            for j in range(m1):
                tmp_data = np.zeros((sample_height, sample_len))
                for k in range((i) * sample_height, (i + 1) * sample_height):
                    tmp_data[k % 6] = data[k][(j) * sample_len: (j + 1) * sample_len]
                input_data.append(tmp_data)


    np_input = np.array(input_data)
    #np_input.reshape(1, np_input.size)
    save_name = os.path.join(CNNinput_path, "CNNinput")
    np.save(save_name, np_input)



def getLabel():
    file_names = get_immediate_files(interfer_path)
    labelData = []
    for file_name in file_names:
        file_path = os.path.join(interfer_path, file_name)
        data = np.genfromtxt(file_path, dtype=float, delimiter=',')
        n = data.shape[0]
        m = data.shape[1]
        n1 = int(n / sample_height)
        m1 = int(m / sample_len)
        for i in range(n1):
            for j in range(m1):
                tmp_data = np.zeros((sample_height, sample_len))
                for k in range((i) * sample_height, (i + 1) * sample_height):
                    tmp_data[k % 6] = data[k][(j) * sample_len: (j + 1) * sample_len]


                labelData.append(np.amax(tmp_data))

    nplabelData = np.array(labelData)
    nplabelData = nplabelData > gate
    save_name = os.path.join(label_path, "label")
    np.save(save_name, nplabelData)



getLabel()


