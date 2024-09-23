import cv2
import numpy as np


class FaceTracker:
    kf_count = 0

    def __init__(self, init_rect=None):
        self.last_state = None
        self.age = 0  # 跟踪器创建后的帧数
        self.time_since_update = 0
        self.num_hits = 0
        self.continual_hits = 0
        self.id = FaceTracker.kf_count
        FaceTracker.kf_count += 1

        self.state_history = []  # 存储状态历史记录以进行分析或可视化

        # 卡尔曼滤波器初始化
        self.kf = cv2.KalmanFilter(7, 4)
        self.measurement = np.zeros((4, 1), np.float32)  # 用以更新的观测矩阵

        self.initialize(init_rect)  # 使用初始矩形初始化卡尔曼滤波器

    def initialize(self, state_mat):
        state_num = 7
        measure_num = 4

        # Kalman filter setup
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ], np.float32)

        self.kf.measurementMatrix = np.eye(measure_num, state_num, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(state_num, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(measure_num, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(state_num, dtype=np.float32)

        # Initialize state vector with bounding box [cx, cy, s, r]
        cx = state_mat[0] + state_mat[2] / 2
        cy = state_mat[1] + state_mat[3] / 2
        s = state_mat[2] * state_mat[3]
        r = state_mat[2] / state_mat[3]

        self.kf.statePost = np.array([cx, cy, s, r, 0, 0, 0], np.float32)

    def xysr2rect(self, center_x, center_y, s, r):
        w = np.sqrt(s * r)
        h = s / w
        x = center_x - w / 2
        y = center_y - h / 2

        if x < 0 and center_x > 0:
            x = 0
        if y < 0 and center_y > 0:
            y = 0

        return np.array([x, y, w, h]).ravel()

    def predict(self) -> np.ndarray:
        """
        预测当前状态
        :return: 预测的矩形框（x, y, w, h）
        """
        prediction = self.kf.predict()
        self.age += 1

        if self.time_since_update > 0:
            self.continual_hits = 0
        self.time_since_update += 1

        predict_box = self.xysr2rect(prediction[0], prediction[1], prediction[2], prediction[3])
        self.state_history.append(predict_box)

        return self.state_history[-1]

    def update(self, new_bbox):
        self.last_state = new_bbox
        self.time_since_update = 0
        self.state_history.clear()
        self.num_hits += 1
        self.continual_hits += 1

        # Update measurement
        cx = new_bbox[0] + new_bbox[2] / 2
        cy = new_bbox[1] + new_bbox[3] / 2
        s = new_bbox[2] * new_bbox[3]
        r = new_bbox[2] / new_bbox[3]

        self.measurement[0, 0] = cx
        self.measurement[1, 0] = cy
        self.measurement[2, 0] = s
        self.measurement[3, 0] = r

        self.kf.correct(self.measurement)

    def get_state(self):
        state = self.kf.statePost
        return self.xysr2rect(state[0], state[1], state[2], state[3])
