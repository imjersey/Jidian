# -*- coding:utf-8 -*-
"""
===================================================
rssi graph with home made interfer detect algorithm
===================================================
"""

import getopt
import sys
import os
import math

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# import matplotlib.animation as animation


span = 32
# s_watch = 19
# i_watch = 54375
_WATCH_ = False
s_watch = 0
i_watch = 34023

human_index = 4
fft_rank = 7
baseline = 10
gate = 6.1
stdev_threshold = 0.001
version = "v4-f3"
show = False

# base_path = "test-data/"
#base_path = "d:\\backup-219\\外场数据汇总\\"
base_path = ""
# path = "97 临时数据\\20170814"

# path = "13 贵阳机场/2016年10月/"
# path += "10月27日 上线挂钩9cm+下线绑在网上"

# path = "09 南京下雨/南京下雨/"
# path += "20160308102138(有人)"
# path += "20160308102930"
# path += "20160308103706"

# path = "06 乌市机场测试数据/2016年10月/10月10日测试数据（砖墙）/距路4.5米砖墙测试PG数据/"

# path = "06 乌市机场测试数据/2017年5月/5月8日（NC数据）/510164520012/19/"
# path += "30"
# path += "20"
# path += "10"

# path = "06 乌市机场测试数据\\2017年5月\\5月8日（NC数据）\\143#-145#出警记录\\23\\58\\"
# path = "99 20170407顺义工厂_多向量双线铺地测试/"
path = "D:\\data\\"

# path += "7座商务车车距线4.5米从头到尾"
# path += "7座商务车车距线4.5米从尾到头20161010185452"
path += "exp14_双线原桂龙_两线之间走_20170407153710"
# path += "exp15_双线原桂龙_两线外侧离线0.5m_20170407154033"
# path += "exp16_双线原桂龙_两线外侧离线1.0m_20170407154314"
# path += "exp19_双线_多人在走动_20170407160251"
# path += "exp20_双线原桂龙_第3个点垂直入侵_原地晃动踱步_20170407160609"
# path += "exp21_双线原桂龙_和另一个人对向走_20170407160932"
# path += "exp22_双线人快速跑_20170407161440"
# path += "exp23_双线骆总开车沿线1m_速度20mile_20170407162450"
# path += "exp24_双线骆总开车沿线2m_速度20mile_20170407162706"

# path = "06 乌市机场测试数据/2017年7月24/1-左数据重组/"
# path += "1l25"

# path += "0804/1"


path = base_path + path


DETECT_NONE = 0
DETECT_ENVIRONMENT = 1
DETECT_SYSTEM = 2
DETECT_HUMAN = 3


def is_rssi_file(_file):
    _ext = _file[len(_file) - 4:len(_file)]
    return _ext == ".txt" or _ext == ".dat" or _ext == "data"


def get_sender_id(_file):
    if _file[0] == 'o':
        return int(_file[1:4])
    if _file[3] == "-":
        return int(_file[0:3])
    return int(_file[0:4])


def get_receiver_id(_file):
    if _file[0] == 'o':
        return int(_file[5:8])
    if _file[3] == "-":
        return int(_file[4:7])
    return int(_file[5:9])


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


def get_log_index(_path):
    _logs = []
    for parent, dirs, files in os.walk(_path):
        if parent != _path:
            continue
        for file in files:
            if is_rssi_file(file):
                _logs.append(get_sender_id(file) * 10000 + get_receiver_id(file))
    _logs.sort()
    i = 0
    _log_index = {}
    for _log in _logs:
        _log_index[_log] = i
        i += 1
    return _log_index


def read_log(_parent, _file):
    _sender = get_sender_id(_file)
    _receiver = get_receiver_id(_file)
    _index = log_map[_sender * 10000 + _receiver]
    file_object = open(_parent + '\\' + _file)
    i = 0
    try:
        for line in file_object:
            rssi_data[_index][i] = int(line)
            i = i + 1
    finally:
        file_object.close()


def read_logs(_path):
    for parent, dirs, files in os.walk(_path):
        if parent != _path:
            continue
        for file in files:
            if is_rssi_file(file):
                read_log(parent, file)


def padding_missing_data():
    for _sender in range(len(log_map)):
        _normal = -1000
        for i in range(max_lines):
            if rssi_data[_sender][i] < -1.1:
                _normal = rssi_data[_sender][i]
                break
        if _normal == -1000:
            rssi_data[_sender] = -61
            continue
        for i in range(max_lines):
            if rssi_data[_sender][i] > -1.1:
                rssi_data[_sender][i] = _normal
            else:
                _normal = rssi_data[_sender][i]


def get_scale(data):
    _average = np.mean(data)
    return min(4.0, max(0.6, (_average + 75.0) / 10.0))


def watch(_sender, _index, _i_width, _str):
    if _WATCH_ and _sender == s_watch and abs(_index - i_watch) < _i_width:
        print(_str)


def filter_small_wave():
    for _sender in range(len(log_map)):
        _scale = get_scale(rssi_data[_sender])
        _rssi_range = max(rssi_data[_sender]) - min(rssi_data[_sender])
        _rssi_range = max(gate, _rssi_range / 25.0)

        i = 0
        while i < max_lines - span:
            _maximum = max(rssi_data[_sender][i:i + span])
            _minimum = min(rssi_data[_sender][i:i + span])
            _average = np.mean(rssi_data[_sender][i:i + span])
            if (_maximum - _minimum) * _scale <= _rssi_range:
                if rssi_data[_sender][i] == rssi_data[_sender][i + span // 2]:
                    rssi_data[_sender][i: i + span // 2] = rssi_data[_sender][i]
                else:
                    rssi_data[_sender][i: i + span // 2] = np.arange(rssi_data[_sender][i],
                                                                     rssi_data[_sender][i + span // 2],
                                                                     (rssi_data[_sender][i + span // 2]
                                                                      - rssi_data[_sender][i]) / (span // 2))
                i += span // 2
            else:
                rssi_data[_sender][i] = _average + (rssi_data[_sender][i] - _average) * _scale
                i += 1


def max_fft(_data):
    _max_val = np.zeros(fft_rank + 1)
    _pos = np.zeros(fft_rank + 1)
    for i in range(1, int(span / 2 + 1.1)):
        for j in range(fft_rank - 1, -1, -1):
            if abs(_data[i]) > _max_val[j]:
                _max_val[j + 1] = _max_val[j]
                _pos[j + 1] = _pos[j]
                _max_val[j] = abs(_data[i])
                _pos[j] = i
    return _pos[:fft_rank]


"""
def filter_long_wave():
    for _sender in range(len(log_index)):
        for i in range(0, max_lines, int(span / 2)):
            _delta = min(max_lines - i, span)
            _fft = np.fft.fft(rssi_data[_sender][i:i + _delta])
            _fft[1:2] = 0
            _rssi_data = np.abs(np.fft.ifft(_fft))
            _delta = min(int(span/2), _delta)
            rssi_data[_sender][i:i + _delta] = _rssi_data[:_delta]
"""


def is_flat(_y):
    _l = len(_y)
    _x = np.arange(_l)
    _x_mean = (_l - 1.0) / 2.0
    _x1 = _x - _x_mean
    _a = np.dot(_x1, _y.T)  / np.dot(_x1, _x1.T)
    _b = np.mean(_y) - _a * _x_mean
    _e = _y - _a * _x - _b
    _p = np.dot(_e, _e.T)
    return _p < 1.1


def filter_fft(_sender, _index, _fft, _max_fft):
    watch(_sender, _index, 40, 1)

    # if _fft[peak_index] > 6000:
    #     return DETECT_ENVIRONMENT
    # if _sender == s_watch and abs(_index - i_watch) < 40:
    #     print(2)

    if _max_fft[0] > 9:
        return DETECT_SYSTEM
    watch(_sender, _index, 40, 2)

    _i0 = int(_max_fft[0] + 0.1)
    for i in range(10, 17):
        if _fft[i] > _fft[_i0] * 0.4 or _fft[i] > 500:
            return DETECT_SYSTEM
    watch(_sender, _index, 40, 2.5)

    if abs(_max_fft[0] - human_index) < 1.1 and _fft[int(_max_fft[0]+0.1)] / _fft[int(_max_fft[1]+0.1)] > 20:
        return DETECT_HUMAN
    watch(_sender, _index, 40, 3)

    if not (np.isin([human_index], _max_fft, True)[0]
            or np.isin([human_index - 1], _max_fft, True)[0]
            or np.isin([human_index + 1], _max_fft, True)[0]):
        if _fft[int(_max_fft[0] + 0.1)] < 90:
            return DETECT_NONE
        return DETECT_ENVIRONMENT
    watch(_sender, _index, 40, 4)

    if _max_fft[0] == 1:
        if _max_fft[1] != human_index and _fft[int(_max_fft[1]+0.1)] > 6 * _fft[human_index] \
                or _fft[1] * _fft[1] / _fft[2] > 20 * _fft[human_index] \
                or _fft[1] * _fft[1] / _fft[2] > 7 * _fft[human_index] + 6 * _fft[human_index - 1] \
                                                 + 6 * _fft[human_index + 1]:
            return DETECT_ENVIRONMENT
    else:
        if _max_fft[0] != human_index and _fft[int(_max_fft[0]+0.1)] > 4.2 * _fft[human_index]:
            return DETECT_ENVIRONMENT
    watch(_sender, _index, 40, 5)

    # if np.isin([6], _max_fft[:3], True)[0]:
    #     if not np.isin([2], _max_fft[:3], True)[0] or not np.isin([3], _max_fft[:3], True):
    #         return DETECT_ENVIRONMENT
    # else:
    #     if not np.isin([2], _max_fft[:2], True)[0] or not np.isin([3], _max_fft[:2], True):
    #         return DETECT_ENVIRONMENT
    if _fft[2] + _fft[3] > 5 * _fft[human_index]:
        return DETECT_ENVIRONMENT
    watch(_sender, _index, 40, 6)

    # if np.isin([15], _max_fft, True)[0]:
    #     return DETECT_ENVIRONMENT

    # for i in [1, 2]:
    #     if np.isin([i], _max_fft[:4], True)[0] and abs(_fft[i]) / abs(_fft[6]) > 10 / i:
    #         return True
    count = 0
    all = [1, 2, 3, 4, 5, 6]
    all.remove(human_index)
    for i in all:
        factor = _fft[i] / _fft[human_index]
        # if factor > 8 or factor > 4 and i != 1:
        #     return DETECT_ENVIRONMENT
        if 0.6 < factor < 1.7:
            count += 1
    if count == 5:
        return DETECT_ENVIRONMENT
    watch(_sender, _index, 40, 7)

    all = [4, 6]
    all.remove(human_index)
    if _fft[2] + _fft[3] + _fft[all[0]] > 20 * _fft[human_index]:
        return DETECT_ENVIRONMENT
    watch(_sender, _index, 40, 8)

    return DETECT_HUMAN


def normalize():
    for _sender in range(len(log_map)):
        i = 0
        while i < max_lines - span:
            _average = np.mean(rssi_data[_sender][i:i+span])
            _maximum = max(rssi_data[_sender][i:i + span])
            _minimum = min(rssi_data[_sender][i:i + span])

            if _maximum - _minimum <= gate:
                delta_i = min(5, max_lines - span - i)
                norm_rssi[_sender][i:i+delta_i] = 0
                i += delta_i
                continue

            is_flat_head = is_flat(rssi_data[_sender][i:i + span // 3])
            is_flat_middle = is_flat(rssi_data[_sender][i + span // 3:i + span * 2 // 3])
            is_flat_tail = is_flat(rssi_data[_sender][i + span * 2 // 3:i + span])
            if is_flat_head or is_flat_middle or is_flat_tail:
                if is_flat_head and is_flat_middle and is_flat_tail:
                    detect_type[_sender][i] = DETECT_NONE
                else:
                    detect_type[_sender][i] = DETECT_ENVIRONMENT
            else:
                # if _sender == s_watch and abs(i - i_watch) < 200:
                #     print(rssi_data[_sender][i])
                _fft = np.fft.fft(rssi_data[_sender][i:i + span])
                _fft = np.abs(_fft**2)
                _max_fft = max_fft(_fft)
                detect_type[_sender][i] = filter_fft(_sender, i, _fft, _max_fft)
                if _WATCH_ and _sender == s_watch and abs(i - i_watch) < 200:
                    print("%d %d %d %s %s" % (_sender, i, detect_type[_sender][i], _max_fft, _fft))
                    for j in range(32):
                        print("%.2lf, " % rssi_data[_sender][i + j], end = '')
                    print()

            if detect_type[_sender][i] == DETECT_NONE:
                norm_rssi[_sender][i] = 0
            else:
                norm_rssi[_sender][i] = rssi_data[_sender][i] - _average
                norm_rssi[_sender][i] /= (_maximum - _minimum)*0.0 + baseline
            i += 1


def get_stdev():
    ratio = np.arange(1.0, 0, -1.0 / span)
    for _sender in range(len(log_map)):
        for i in range(max_lines - 2 * span):
            stdev_rssi[_sender][i] = np.dot(norm_rssi[_sender][i:i+span], norm_rssi[_sender][i:i+span].T * ratio)
            stdev_rssi[_sender][i] = stdev_rssi[_sender][i] / (span + 1) * 2.0
            if _WATCH_ and _sender == s_watch and abs(i - i_watch) < 10:
                print(stdev_rssi[_sender][i])
            if stdev_rssi[_sender][i] > 0.27:
                stdev_rssi[_sender][i] = max(0.0, 0.54 - stdev_rssi[_sender][i])


def update_line(num, _data, line):
    line.set_data(_data[..., :num])
    return line,


short_args = "p:s"
long_args = ["path=", "show"]
opts, args = getopt.getopt(sys.argv[1:], short_args, long_args)
for opt, val in opts:
    if opt in ("-p", "--path"):
        path = val
    if opt in ("-s", "--show"):
        show = True

log_map = get_log_index(path)
max_lines = get_max_lines(path)
if max_lines == 0:
    exit(0)
rssi_data = np.zeros((len(log_map), max_lines))
rssi_data = rssi_data - 1
rssi_range = np.zeros((len(log_map)))
norm_rssi = np.zeros((len(log_map), max_lines))
detect_type = np.zeros((len(log_map), max_lines))
stdev_rssi = np.zeros((len(log_map), max_lines))

read_logs(path)
print("完成数据加载")

padding_missing_data()
print("完成漏报弥补")

# filter_small_wave()
# print("完成滤小杂波")

# filter_long_wave()
# print("完成滤长杂波")

normalize()#[][]
print("完成正规化")

get_stdev()#[][]
print("完成功率分析")

# plt.figure(1)
name = path[len(base_path):]
name = name.replace("/", "-")
name = name.replace("\\", "-")
name = name + "--" + version
#my_font = matplotlib.font_manager.FontProperties(fname='C:/Windows/Fonts/simsun.ttc')
plt.figure(figsize=(9, 9))
#plt.suptitle(name, FontProperties=my_font)
plt.suptitle(name)
ax = plt.subplot(111)

y_scale = 1

plot = []
for sender in range(len(log_map)):
    # print("绘制: ", sender)
    for ii in range(max_lines - span):
        if stdev_rssi[sender][ii] > stdev_threshold:
            l_i = max(0, int(ii - span * 0.2))
            r_i = min(int(ii + span * 0.2 + 1), max_lines)
            counter = np.zeros(DETECT_HUMAN + 1)
            for jj in range(l_i, r_i):
                counter[int(detect_type[sender][jj] + 0.1)] += 1
            counter[DETECT_NONE] = 0
            _type = np.argmax(counter)
            plot.append((ii / y_scale, sender,  stdev_threshold / stdev_rssi[sender][ii], _type))
print("完成")

plot.append((0, 0, 0.1, DETECT_SYSTEM))
plot.append(((max_lines - span - 1) / y_scale, len(log_map), 0.1, DETECT_SYSTEM))
plot2 = np.asarray(plot).T
d_max = np.zeros(DETECT_HUMAN + 1)
for ii in range(len(plot2[2])):
    t = int(plot2[3][ii] + 0.1)
    d_max[t] = max(d_max[t], plot2[2][ii])

size = np.zeros(len(plot2[0]))
color = np.zeros((len(plot2[0]), 3))
for ii in range(len(plot2[0])):
    t = int(plot2[3][ii] + 0.1)
    v = plot2[2][ii] / d_max[t]
    if t == DETECT_HUMAN:
        size[ii] = 86 * v + 14
        color[ii] = np.array([v / 2.0 + 0.5, 0.3, 0.2])
    if t == DETECT_SYSTEM:
        size[ii] = 40 * v
        color[ii] = np.array([v / 5.0 + 0.2, v / 3.0 + 0.6, 0.2])
    if t == DETECT_ENVIRONMENT:
        size[ii] = 14 * v
        color[ii] = np.array([0.2, v / 5.0 + 0.2, v / 3.0 + 0.5])

plt.scatter(plot2[0], plot2[1], s=size, c=color, alpha=0.5)
if show:
    plt.show()
else:
    subdir = "png/" + version
    if not os.path.exists(subdir):
        os.mkdir(subdir)
    plt.savefig(subdir + "/" + name + ".png", dpi=300)
