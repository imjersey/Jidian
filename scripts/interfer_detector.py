import time
import numpy as np

DIR_UP = 0
DIR_DOWN = 1
DIR_STABLE = 2

VALID_TIME = 2000                              # 有效时延2s
FAST_AVG_PROCESS_NUM = 25                     # n=25，在12ms的情况下，相当于300ms
SLOW_AVG_PROCESS_NUM = 2000                   # n=2000，在12ms的情况下，相当于24000ms
STABLE_AVG_NUM = 50
SCALE_DOWN = 0.5
BUFFER1_SIZE = 400
BUFFER2_SIZE = 800
RATIO = 1                                      # 缩减比例系数
THRESHOLD_01 = int(BUFFER1_SIZE / 200 * 3)
THRESHOLD_02 = int(BUFFER2_SIZE / 200 * 3)
BUFFER1_HALF_SIZE = int(BUFFER1_SIZE / 2)
BUFFER2_HALF_SIZE = int(BUFFER2_SIZE / 2)
WAITING_SIZE = BUFFER2_SIZE                    # max(SLOW_AVG_PROCESS_NUM,BUFFER2_SIZE)


class InterferDetector:
    s_energy_l1_buffer_size = 0        # 1级能量缓冲大小
    s_energy_l2_buffer_size = 0        # 2级能量缓冲大小
    s_interfer_detect_window_size = 0  # 滑动窗口大小
    s_frequency_monitor_period = 0     # 信号强度监测周期
    s_threshold_for_frequency = 0      # 拐点频率门限值
    s_threshold_for_integral = 0       # 信号强度积分门限值
    s_threshold_of_slope = 0           # 两次拐点间隔门限值
    s_frequency_weight = 0             # 频率权重
    s_centre_frequency = 0             # 对频率进行归一化处理
    s_scale = 0.0                      # 权重系数
    interfer_detector_version = "3E"

    def __init__(self):
        self.sample_counter = 0
        self.prev_inflexion_position_for_integral = 0
        self.inflexion_pos = 0
        self.dir_for_frequency = DIR_STABLE
        self.dir_for_integral = DIR_STABLE
        self.frequency_counter = []
        self.energy_l1_buffer = []
        self.energy_l2_buffer = []
        self.orig_rssi_buffer = []
        self.inflexion_rssi_for_energy = []
        self.prev_rssi_for_frequency = -1
        self.prev_rssi_for_integral = -1
        self.last_valid_rssi = 0
        self.interfer = 0
        self.arg_rssi_for_frequency = 0
        self.inflexion_pos_rssi = 0
        self.l2_last_sum_a = 0
        self.l2_last_sum_b = 0
        self.start_counter = 0
        self.delta_area = 0
        self.dir_for_preprocess = DIR_STABLE
        self.fast_avg_rssi = 0
        self.slow_avg_rssi = 0
        self.last_time = 0
        self.prev_rssi_for_preprocess = 0
        self.scaling_counter = []
        self.inflexion_counter = []
        self.buffer1_inflexion_num = 0
        self.buffer2_inflexion_num = 0
        self.buffer3_inflexion_num = 0
        self.buffer4_inflexion_num = 0

    def get_cur_freq(self):
        return len(self.frequency_counter)

    def get_cur_energy(self):
        if len(self.energy_l2_buffer) > 0:
            return self.energy_l2_buffer[len(self.energy_l2_buffer) - 1]
        return 0

    def get_interfer(self):
        return self.interfer

    # 判断是否为拐点
    @staticmethod
    def is_inflexion(current_value, last_inflexion_value, recent_direction, threshold):
        if current_value - last_inflexion_value > threshold:
            if recent_direction != DIR_UP:
                return DIR_UP
        if last_inflexion_value - current_value > threshold:
            if recent_direction != DIR_DOWN:
                return DIR_DOWN
        return DIR_STABLE

    def reset(self, rssi):
        self.prev_rssi_for_frequency = rssi
        self.prev_rssi_for_integral = rssi
        self.last_valid_rssi = rssi
        self.prev_inflexion_position_for_integral = self.sample_counter
        self.dir_for_frequency = DIR_STABLE
        self.dir_for_integral = DIR_STABLE
        self.arg_rssi_for_frequency = rssi
        self.inflexion_pos_rssi = rssi
        self.inflexion_rssi_for_energy.append(self.inflexion_pos_rssi)
        self.inflexion_pos = 1
        self.fast_avg_rssi = rssi
        self.slow_avg_rssi = rssi
        self.start_counter = 1
        self.delta_area = 0
        self.prev_rssi_for_preprocess = rssi
        self.dir_for_preprocess = DIR_STABLE
        self.last_time = time.time()

    def push_sample(self, org_rssi):
        _pre_rssi = self.prepare(org_rssi)
        scaling_ratio = self.calculate_scaling(_pre_rssi)
        rssi = self.preprocess(_pre_rssi, scaling_ratio)
        if (len(self.frequency_counter) > 0) \
                and (self.sample_counter - self.frequency_counter[0] > self.s_frequency_monitor_period):
            del self.frequency_counter[0]
        energy = 0
        # 判断拐点积分门限
        tmp_dir = self.is_inflexion(rssi, self.prev_rssi_for_integral, self.dir_for_integral,
                                    self.s_threshold_for_integral)
        if tmp_dir != DIR_STABLE:
            self.dir_for_integral = tmp_dir
            self.inflexion_rssi_for_energy.append(self.prev_rssi_for_integral)
            # sample_counter 当前出现拐点的位置
            if self.sample_counter - self.prev_inflexion_position_for_integral < self.s_threshold_of_slope:
                energy = pow(abs(self.inflexion_rssi_for_energy[0] - self.inflexion_rssi_for_energy[1]), 2)
            del self.inflexion_rssi_for_energy[0]
            self.prev_inflexion_position_for_integral = self.sample_counter
            if abs(self.inflexion_rssi_for_energy[0] - self.arg_rssi_for_frequency) > self.s_threshold_for_frequency:
                self.frequency_counter.append(self.inflexion_pos)
        if (rssi > self.prev_rssi_for_integral and self.dir_for_integral == DIR_UP) \
                or (rssi < self.prev_rssi_for_integral and self.dir_for_integral == DIR_DOWN):
            self.inflexion_pos_rssi = self.prev_rssi_for_integral
            self.prev_rssi_for_integral = rssi
            self.inflexion_pos = self.sample_counter
        self.orig_rssi_buffer.append(rssi)
        if len(self.orig_rssi_buffer) > self.s_energy_l1_buffer_size:
            del self.orig_rssi_buffer[0]
        self.energy_l1_buffer.append(energy)
        if len(self.energy_l1_buffer) >= self.s_energy_l1_buffer_size:
            if len(self.energy_l2_buffer) == 0:
                total_energy = np.sum(self.energy_l1_buffer[0:self.s_energy_l1_buffer_size])
            else:
                total_energy = self.energy_l2_buffer[len(self.energy_l2_buffer) - 1] + energy - self.energy_l1_buffer[0]
                del self.energy_l1_buffer[0]
            self.energy_l2_buffer.append(total_energy)
            self.l2_last_sum_a += total_energy
            if len(self.energy_l2_buffer) > self.s_energy_l2_buffer_size - self.s_interfer_detect_window_size:
                self.l2_last_sum_b += total_energy
            if len(self.energy_l2_buffer) == self.s_energy_l2_buffer_size:
                if self.l2_last_sum_a != 0 and self.l2_last_sum_b != 0:
                    self.interfer = self.l2_last_sum_b / self.l2_last_sum_a * self.l2_last_sum_b \
                               * pow(len(self.frequency_counter) / self.s_centre_frequency, self.s_frequency_weight) \
                               / self.s_scale
                else:
                    self.interfer = 0
                self.l2_last_sum_a -= self.energy_l2_buffer[0]
                self.l2_last_sum_b -= self.energy_l2_buffer[self.s_energy_l2_buffer_size
                                                            - self.s_interfer_detect_window_size]
                del self.energy_l2_buffer[0]
                if self.interfer > 200:
                    return True
        return False

    def prepare(self, org_rssi):
        if org_rssi < -1.1:
            if self.sample_counter == 0:
                self.reset(org_rssi)
            current_time = time.time()
            if current_time - self.last_time > 5:
                self.reset_preprocess()
            self.last_time = current_time
            self.last_valid_rssi = org_rssi
            self.sample_counter += 1
            self.fast_avg_rssi = ((FAST_AVG_PROCESS_NUM - 1.0) * self.fast_avg_rssi + org_rssi) / FAST_AVG_PROCESS_NUM
            self.slow_avg_rssi = ((SLOW_AVG_PROCESS_NUM - 1.0) * self.slow_avg_rssi + org_rssi) / SLOW_AVG_PROCESS_NUM
            self.arg_rssi_for_frequency = ((STABLE_AVG_NUM - 1.0) * self.arg_rssi_for_frequency + org_rssi)
            self.arg_rssi_for_frequency /= STABLE_AVG_NUM
        else:
            if self.sample_counter > 0:
                self.sample_counter += 1
                return self.last_valid_rssi
        return org_rssi

    def calculate_scaling(self, org_rssi):
        delta = abs(org_rssi - self.fast_avg_rssi)
        # 计算面积
        self.scaling_counter.append(delta)
        if len(self.scaling_counter) <= BUFFER1_HALF_SIZE:
            self.delta_area += delta
        else:
            self.delta_area += delta - self.scaling_counter[0]
            del self.scaling_counter[0]
        # 计算缩减比例
        if self.delta_area > BUFFER1_HALF_SIZE:
            return RATIO * (BUFFER1_HALF_SIZE / self.delta_area)
        else:
            return SCALE_DOWN

    def preprocess(self, pre_rssi, scaling_ratio):
        pre_rssi = self.fast_avg_rssi + (pre_rssi - self.fast_avg_rssi) * scaling_ratio
        # 超出跟踪缓冲区长度就删除无效的拐点
        if len(self.inflexion_counter) > 0 and self.sample_counter - self.inflexion_counter[0] >= WAITING_SIZE:
            del self.inflexion_counter[0]
        # 判断是否出现拐点
        tmp_dir = self.is_inflexion(pre_rssi, self.prev_rssi_for_preprocess, self.dir_for_preprocess,
                                    self.s_threshold_for_frequency)
        if tmp_dir != DIR_STABLE:
            self.dir_for_preprocess = tmp_dir
            self.set_inflexion_parm(self.sample_counter)
        if (pre_rssi > self.prev_rssi_for_preprocess and self.dir_for_preprocess == DIR_UP) \
                or (pre_rssi < self.prev_rssi_for_preprocess and self.dir_for_preprocess == DIR_DOWN):
            self.prev_rssi_for_preprocess = pre_rssi
        # 满足跟踪缓冲区长度，开始计算
        if len(self.inflexion_counter) > 0 and self.sample_counter - self.start_counter >= WAITING_SIZE - 1:
            self.buffer1_inflexion_num = 0
            self.buffer2_inflexion_num = 0
            self.buffer3_inflexion_num = 0
            self.buffer4_inflexion_num = 0
            for index in range(len(self.inflexion_counter)):
                pos = self.inflexion_counter[index]
                if pos > self.sample_counter - BUFFER1_HALF_SIZE:
                    self.buffer1_inflexion_num += 1
                    self.buffer3_inflexion_num += 1
                if self.sample_counter - BUFFER1_HALF_SIZE >= pos > self.sample_counter - BUFFER1_SIZE:
                    self.buffer2_inflexion_num += 1
                    self.buffer3_inflexion_num += 1
                if self.sample_counter - BUFFER1_SIZE > pos > self.sample_counter - BUFFER2_HALF_SIZE:
                    self.buffer3_inflexion_num += 1
                if self.sample_counter - BUFFER2_HALF_SIZE >= pos > self.sample_counter - BUFFER2_SIZE:
                    self.buffer4_inflexion_num += 1
            if self.buffer1_inflexion_num >= THRESHOLD_01 and self.buffer2_inflexion_num >= THRESHOLD_01 \
                    and self.buffer3_inflexion_num >= THRESHOLD_02 and self.buffer4_inflexion_num >= THRESHOLD_02:
                return self.fast_avg_rssi + (pre_rssi - self.fast_avg_rssi) * scaling_ratio
        return pre_rssi

    def set_inflexion_parm(self, inflexion_pos_for_preprocess):
        self.inflexion_counter.append(inflexion_pos_for_preprocess)

    def reset_preprocess(self):
        self.inflexion_counter = []
        self.start_counter = self.sample_counter

    def set_params(self, energy_l1_buffer_size, energy_l2_buffer_size, interfer_detect_window_size,
                   frequency_monitor_period, threshold_for_frequency, threshold_for_integral, threshold_of_slope,
                   frequency_weight, centre_frequency, scale):
        self.s_energy_l1_buffer_size = energy_l1_buffer_size
        self.s_energy_l2_buffer_size = energy_l2_buffer_size
        self.s_interfer_detect_window_size = interfer_detect_window_size
        self.s_frequency_monitor_period = frequency_monitor_period
        self.s_threshold_for_frequency = threshold_for_frequency
        self.s_threshold_for_integral = threshold_for_integral
        self.s_threshold_of_slope = threshold_of_slope
        self.s_frequency_weight = frequency_weight
        self.s_centre_frequency = centre_frequency
        self.s_scale = scale

    def get_version(self):
        return self.interfer_detector_version

    def get_orig_rssi(self):
        return self.orig_rssi_buffer
