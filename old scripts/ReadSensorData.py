# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
# import getopt
# import sys
# import os
# ###init
#
# rssi_data = [][]
#
#
# def is_rssi_file(_file):
#     _ext = _file[len(_file) - 4:len(_file)] #[][]
#     return _ext == ".txt" or _ext == ".dat" or _ext == "data"
#
# def read_logs(_path):
#     for parent, dirs, files in os.walk(_path):
#         if parent != _path:
#             continue
#         for file in files:
#             if is_rssi_file(file):
#                 read_log(parent, file)
#
# def read_log(_parent, _file):
#     _sender = get_sender_id(_file)
#     _receiver = get_receiver_id(_file)
#     file_object = open(_parent + '\\' + _file)
#     i = 0
#     try:
#         for line in file_object:
#             rssi_data[_index][i] = int(line)
#             i = i + 1
#     finally:
#         file_object.close()
#
# def get_sender_id(_file):
#     if _file[0] == 'o':
#         return int(_file[1:4])
#     if _file[3] == "-":
#         return int(_file[0:3])
#     return int(_file[0:4])
#
#
# def get_receiver_id(_file):
#     if _file[0] == 'o':
#         return int(_file[5:8])
#     if _file[3] == "-":
#         return int(_file[4:7])
#     return int(_file[5:9])
#
#
#
#
# short_args = "p:s"
# long_args = ["path=", "show"]
# opts, args = getopt.getopt(sys.argv[1:], short_args, long_args)
# for opt, val in opts:
#     if opt in ("-p", "--path"):
#         path = val
#     if opt in ("-s", "--show"):
#         show = True
#
#
#
#
#
# read_logs(path)
# print("完成数据加载")
# base_path = ""
# version = 0
# # plt.figure(1)
# name = path[len(base_path):]
# name = name.replace("/", "-")
# name = name.replace("\\", "-")
# name = name + "--" + version
#
# plt.figure(figsize=(9, 9))
#
# plt.suptitle(name)
# ax = plt.subplot(111)
#
# y_scale = 1
#
# plot = []
#
#
#
#
#
#
# plt.show()
