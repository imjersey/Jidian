"""
==================================================
Read rssi data
==================================================
"""

import getopt
import sys
import os
import math

import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.animation as animation

import interfer_detector as detector

def rssi(path):
    version = "o1"
    show = False
    stdev_threshold = 150.0


    def is_rssi_file(_file):
        _ext = _file[len(_file) - 4:len(_file)]
        return _ext == ".txt" or _ext == ".dat" or _ext == "data"


    def read_log(_parent, _file, _index):
        interfer_detector = detector.InterferDetector()
        interfer_detector.set_params(100, 200, 100, 200, 2, 4, 20, 2, 5, 20)
        file_object = open(_parent + '\\' + _file)
        i = 0
        try:
            for line in file_object:
                interfer_detector.push_sample(int(line))
                interfer = interfer_detector.get_interfer()
                interfer_mat[_index][i] = interfer
                i = i + 1
        finally:
            file_object.close()

    def get_lines(_file):
        file_object = open(_file)
        try:
            all_the_text = file_object.read()
        finally:
            file_object.close()
        return all_the_text.count('-') + 1

    def get_max_lines(_path):
        lines = []
        for parent, dirs, files in os.walk(_path):
            for file in files:
                if is_rssi_file(file):
                    lines.append(get_lines(parent + "\\" + file))
        if len(lines) == 0:
            print(_path, " is empty")
            return 0
        return max(lines) + 1

    def read_logs(_path):
        _index = 0
        for parent, dirs, files in os.walk(_path):
            for file in files:
                if is_rssi_file(file):
                    read_log(parent, file, _index)
                    _index += 1

    max_lines = get_max_lines(path)
    len1 = len(os.listdir(path))
    len2 = max_lines

    interfer_mat = np.zeros(len1, len2)

    y_scale = 1
    plot = []
    read_logs(path)


    print("完成")


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


# short_args = "p:s"
# long_args = ["path=", "show"]
# opts, args = getopt.getopt(sys.argv[1:], short_args, long_args)
# for opt, val in opts:
#     if opt in ("-p", "--path"):
#         path = val
#     if opt in ("-s", "--show"):
#         show = True

base_path = "D:\\data"
original_path = os.path.join(base_path, "original")
label_path = os.path.join(base_path, "label")
sub_dirs = get_immediate_subdirectories(original_path)

sub_dirs = get_immediate_subdirectories(original_path)

for sub_dir in sub_dirs:

    path = os.path.join(original_path, sub_dir)
    rssi(path)


