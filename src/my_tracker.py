import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment


class StudentTracker:

    def __init__(self, bbox, kps, score, frame_num, id_):
        self.id = id_
        self.bbox = bbox
        self.kps = kps
        self.score = score
        self.last_update_frame = frame_num  # 最后一次更新的帧数
        self.inactive_count = 0  # 连续不更新的帧数
        self.age = 0  # 经历的帧数
        self.time_since_update = 0  # 距离上次更新的帧数
        self.num_hits = 0  # 检测到该学生的次数
        self.continual_hits = 0  # 持续检测到该学生的次数
        self.state_history = []

        # Kalman filter initialization
        self.kf = cv2.KalmanFilter(7, 4)  # 状态维数为7，测量维数为4
        self.kf.transitionMatrix = np.array([[1, 0, 0, 0, 0.5, 0, 0],
                                             [0, 1, 0, 0, 0, 0.5, 0],
                                             [0, 0, 1, 0, 0, 0, 0.5],
                                             [0, 0, 0, 1, 0, 0, 0],
                                             [0, 0, 0, 0, 1, 0, 0],
                                             [0, 0, 0, 0, 0, 1, 0],
                                             [0, 0, 0, 0, 0, 0, 1]], dtype=np.float32)
        self.kf.measurementMatrix = np.eye(4, 7, dtype=np.float32)  # 观测矩阵
        self.kf.processNoiseCov = np.eye(7, dtype=np.float32) * 1e-3  # 过程噪声协方差
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1  # 观测噪声协方差
        self.kf.errorCovPost = np.eye(7, dtype=np.float32)  # 后验误差协方差

        # 初始化状态
        self.kf.statePost = np.array([bbox[0] + bbox[2] / 2,
                                      bbox[1] + bbox[3] / 2,
                                      bbox[2] * bbox[3],
                                      bbox[2] / bbox[3], 0, 0, 0], dtype=np.float32)  # 状态向量

        self.measurement = np.zeros((4, 1), dtype=np.float32)  # 观测向量

    def get_smoothed_state(self, window_size=5):
        if len(self.state_history) == 0:
            return self.kf.statePost  # 如果没有历史状态，则返回当前状态
        if len(self.state_history) >= window_size:
            smoothed_state = np.mean(self.state_history[-window_size:], axis=0)
        else:
            smoothed_state = self.state_history[-1]

        return smoothed_state

    def update(self, bbox, kps, score, frame_num):
        self.bbox = bbox
        self.kps = kps
        self.score = score
        self.last_update_frame = frame_num

        # Measurement update
        self.measurement[0] = bbox[0] + bbox[2] / 2
        self.measurement[1] = bbox[1] + bbox[3] / 2
        self.measurement[2] = bbox[2] * bbox[3]
        self.measurement[3] = bbox[2] / bbox[3]
        self.kf.correct(self.measurement)

        self.time_since_update = 0
        self.num_hits += 1
        self.continual_hits += 1

    def predict(self):
        # 使用卡尔曼滤波器预测下一个位置
        self.continual_hits = 0
        prediction = self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        history = self.xysr2rect(prediction[0], prediction[1], prediction[2], prediction[3])
        if history is not None:
            self.state_history.append(history)

        return self.get_smoothed_state()

    def increment_inactive_count(self):
        self.inactive_count += 1

    def xysr2rect(self, center_x, center_y, s, r):
        if s <= 0 or r <= 0:
            return None  # 返回None表示无效框

        w = np.sqrt(s * r)
        h = s / w
        x = center_x - w / 2
        y = center_y - h / 2

        if x < 0 and center_x > 0:
            x = np.zeros(1)
        if y < 0 and center_y > 0:
            y = np.zeros(1)
        # print(f"x: {x}, y: {y}, w: {w}, h: {h}")
        # print(f"Types - x: {type(x)}, y: {type(y)}, w: {type(w)}, h: {type(h)}")

        return np.array([x, y, w, h], dtype=np.float32).ravel()


class Tracker:
    def __init__(self, max_inactive_frames=15):
        self.id_map = np.zeros((640, 360), dtype=np.uint8)
        self.id_width_height_map = np.zeros((640, 360), dtype=np.float32)
        self.student_trackers = []
        self.next_id = 1
        self.current_frame = 0  # 当前帧数
        self.used_ids = [False] * 60  # 最多 60 个学生
        self.max_inactive_frames = max_inactive_frames  # 如果在 max_inactive_frames 内没有更新，则认为学生离开了
        self.min_hits = 3  # 一个框被认为有效的最小帧数    还没有使用到

    def get_new_id(self):
        """
        返回没有被使用的 ID
        """
        for i in range(len(self.used_ids)):
            if not self.used_ids[i]:
                self.used_ids[i] = True
                return i + 1
        return -1

    def release_id(self, id_):
        """
        释放一个 ID
        """
        if 1 <= id_ <= 60:
            self.used_ids[id_ - 1] = False

    def update_id_map(self, x, y, w, h, id_):
        x, y, w, h = int(x), int(y), int(w), int(h)
        self.id_map[y:y + h, x:x + w] = id_

    def update_width_height_map(self, x, y, w, h, id_):
        x, y, w, h = int(x), int(y), int(w), int(h)
        self.id_width_height_map[y:y + h, x:x + w] = w * h

    def update(self, bboxes, kpss, scores):
        # 创建当前帧的检测字典列表
        current_detections = [{'bbox': bboxes[i], 'kps': kpss[i], 'score': scores[i]} for i in range(len(bboxes))]

        # 匹配当前帧的检测与现有的跟踪器
        matches = match_detections_to_trackers(self.student_trackers, current_detections)

        # 处理匹配的检测框
        matched_detections = set([m[1] for m in matches])
        for match in matches:
            tracker_index, detection_index = match
            tracker = self.student_trackers[tracker_index]
            detection = current_detections[detection_index]
            tracker.update(detection['bbox'], detection['kps'], detection['score'], self.current_frame)
            self.update_id_map(*detection['bbox'], tracker.id)
            self.update_width_height_map(*detection['bbox'], tracker.id)

        # 处理未匹配的检测框（新出现的学生）
        unmatched_detections = set(range(len(current_detections))) - matched_detections
        for detection_index in unmatched_detections:
            detection = current_detections[detection_index]
            new_id = self.get_new_id()
            student_tracker = StudentTracker(detection['bbox'], detection['kps'], detection['score'],
                                             self.current_frame, new_id)
            self.student_trackers.append(student_tracker)
            self.update_id_map(*detection['bbox'], new_id)
            self.update_width_height_map(*detection['bbox'], new_id)

        # 增加未匹配跟踪器的 inactive_count
        for tracker in self.student_trackers:
            if tracker.id not in [self.student_trackers[m[0]].id for m in matches]:
                tracker.increment_inactive_count()
                if tracker.inactive_count % 3 == 0:
                    tracker.predict()  # 使用卡尔曼滤波预测位置
                    # 更新 ID 和宽高地图，用预测的 bbox 代替原来的检测框
                    self.update_id_map(*tracker.bbox, tracker.id)  # 更新 ID 地图
                    self.update_width_height_map(*tracker.bbox, tracker.id)  # 更新宽高地图

        # 移除超过最大未活动帧数的跟踪器，并释放其 ID
        active_trackers = []
        for tracker in self.student_trackers:
            if tracker.inactive_count < self.max_inactive_frames:
                active_trackers.append(tracker)
            else:
                self.release_id(tracker.id)

        self.student_trackers = active_trackers

        # 更新当前帧号
        self.current_frame += 1


def compute_cost_matrix(trackers, detections):
    num_trackers = len(trackers)
    num_detections = len(detections)
    cost_matrix = np.zeros((num_trackers, num_detections))

    for t, tracker in enumerate(trackers):
        for d, detection in enumerate(detections):
            cost_matrix[t, d] = 1 - bbox_iou(tracker.bbox, detection['bbox'])  # 使用 IoU 计算匹配成本
    return cost_matrix


def match_detections_to_trackers(trackers, detections, threshold=50):
    cost_matrix = compute_cost_matrix(trackers, detections)
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=False)
    matches = []

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < threshold:  # 可以调整这个阈值
            matches.append((r, c))

    return matches


def bbox_iou(bbox1, bbox2):
    """
    计算两个边界框的交并比（IoU）
    :param bbox1: 第一个边界框 (x, y, w, h)
    :param bbox2: 第二个边界框 (x, y, w, h)
    :return: IoU 值
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)

    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2

    iou = inter_area / (bbox1_area + bbox2_area - inter_area)
    return iou
