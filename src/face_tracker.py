import numpy as np
import cv2


class FaceTracker:
    kf_count = 0

    def __init__(self, stateMat=None):
        self.kf = cv2.KalmanFilter(4, 2)  # 状态向量维度为4，测量向量维度为2
        self.measurement = np.zeros((2, 1), np.float32)  # 用以更新的观测矩阵
        self.state_history = []  # 存储历史状态信息

        self.id = FaceTracker.kf_count
        FaceTracker.kf_count += 1
        self.age = 0  # 跟踪器创建后的帧数

        self.currentPosition = None  # 保存当前的跟踪位置
        self.lastState = None  # 存储最后一次更新的矩形状态
        self.time_since_update = 0  # 自上次状态更新以来经过的时间
        self.num_hits = 0  # 跟踪器命中次数，成功匹配到目标的次数
        self.continual_hits = 0  # 连续成功匹配到目标的次数

        if stateMat is not None:
            self.initialize(stateMat)

    def initialize(self, stateMat):
        # 初始化卡尔曼滤波器状态
        self.kf.statePre = np.array([[stateMat[0]], [stateMat[1]], [0], [0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], np.float32)

    def predict(self):
        self.lastState = self.kf.predict()
        return self.lastState

    def update(self, new_bbox):
        self.measurement[0] = new_bbox[0] + new_bbox[2] / 2  # 计算中心点 x
        self.measurement[1] = new_bbox[1] + new_bbox[3] / 2  # 计算中心点 y
        self.kf.correct(self.measurement)

        self.currentPosition = new_bbox
        self.state_history.append(new_bbox)
        self.age += 1

        # 更新跟踪器命中次数
        self.num_hits += 1
        self.continual_hits += 1
        self.time_since_update = 0

    def xysr2rect(self, center_x, center_y, s, r):
        # 根据中心点、宽度和高度计算矩形
        return (center_x - s / 2, center_y - r / 2, s, r)

    def get_state(self):
        return self.lastState
