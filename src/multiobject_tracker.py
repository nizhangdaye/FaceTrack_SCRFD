from typing import List, Dict

import cv2
import numpy as np
from src.hungarian import HungarianAlgorithm
from src.face_tracker import FaceTracker


class TrackingBox:
    def __init__(self, frame=0, box=None, id=0):
        self.frame = frame
        self.box = box
        self.id = id

    # def update(self, det_frame_box):


class MultiObjectTracker:
    """
    这个类实现了多目标跟踪器
    流程：
        1. 初始化跟踪器
        2. 预测位置
        3. 计算IoU矩阵
        4. 配对
        5. 更新跟踪器
        6. 创建新跟踪器
        7. 输出跟踪结果
    """

    def __init__(self, max_age=15, iou_threshold=0.6):
        self.total_frames = 0  # 总帧数
        self.total_time = 0.0  # 总时间
        self.frame_count = 0  # 当前帧数
        self.max_missing_frames = max_age  # 最大容忍跟踪框未更新的帧数，用以删除老旧的跟踪框
        self.min_hits = 3  # 一个框被认为有效的最小帧数
        self.iou_threshold = iou_threshold  # 匹配阈值
        self.trackers = []  # 跟踪器列表

        self.predicted_boxes = []  # 预测框列表
        self.iou_matrix = []  # 存储预测的边界框和检测边界框之间的IOU
        self.assignment = []  # 每个预测边界框和检测边界框的匹配关系,表示每个预测框和检测框的最佳匹配
        self.unmatched_detections = set()  # 未匹配的检测边界框集合
        self.unmatched_trajectories = set()  # 未匹配的跟踪轨迹集合
        self.all_items = set()  # 所有检测的集合
        self.matched_items = set()  # 匹配项
        self.matched_pairs = []  # 每个 cv::Point 表示一个匹配对，其中 x 是跟踪器的索引，y 是检测框的索引
        self.frame_tracking_result = []  # 跟踪结果列表
        self.trk_num = 0  # 当前帧跟踪器数量
        self.det_num = 0  # 当前帧的检测数量
        self.cycle_time = 0.0  # 周期时间

    def calculate_iou(self, rect1: np.ndarray, rect2: np.ndarray):
        """
        Calculate the intersection over union (IoU) of two bounding boxes.
        """
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        xA = max(x1, x2)
        yA = max(y1, y2)
        xB = min(x1 + w1, x2 + w2)
        yB = min(y1 + h1, y2 + h2)
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        box1Area = w1 * h1
        box2Area = w2 * h2
        iou = interArea / float(box1Area + box2Area - interArea)
        return iou

    def update(self, det_frame_data: List[Dict]) -> List[TrackingBox]:
        """
        更新跟踪器
        流程：
            1. 预测位置
            2. 计算IoU矩阵
            3. 配对
            4. 更新跟踪器
            5. 创建新跟踪器
            6. 输出跟踪结果
        :param det_frame_data: 包含检测框信息的列表，每个元素包含'box'和'id'字段
        :return: 跟踪结果列表
        """
        self.total_frames += 1
        self.frame_count += 1
        start_time = cv2.getTickCount()

        if len(self.trackers) == 0:  # 如果不存在跟踪链接，则初始化跟踪链接
            for det in det_frame_data:
                trk = FaceTracker(det['box'])  # 将检测框信息转换为跟踪器
                self.trackers.append(trk)
            return []

        # 从现有跟踪链接中获取预测位置
        self.predicted_boxes = []
        for tracker in self.trackers[:]:
            pBox = tracker.predict()
            if pBox[0] >= 0 and pBox[1] >= 0:
                self.predicted_boxes.append(pBox)  # 将预测框添加到列表中
            else:  # 如果预测框不在图像内，则删除该跟踪器
                self.trackers.remove(tracker)

        # 使用 IoU 矩阵将检测与跟踪链接相关联
        self.trk_num = len(self.predicted_boxes)
        self.det_num = len(det_frame_data)

        self.iou_matrix = np.zeros((self.trk_num, self.det_num), dtype=np.float32)
        for i in range(self.trk_num):
            for j in range(self.det_num):
                # print(
                #     f"predicted_boxes[{i}] = {type(self.predicted_boxes[i])}, det_frame_data[{j}] = {type(det_frame_data[j]['box'])}")
                self.iou_matrix[i][j] = 1 - self.calculate_iou(self.predicted_boxes[i], det_frame_data[j]['box'])

        # 使用 Hungarian 算法求解赋值问题
        # 生成的赋值为 [track（prediction） ： detection]，其中 len=preNum
        hungarian = HungarianAlgorithm()
        self.assignment, cost = hungarian.solve(self.iou_matrix)
        print(f"assignment = {self.assignment}")
        print(f"cost = {cost}")

        # 处理不匹配的轨迹和检测
        self.unmatched_trajectories.clear()
        self.unmatched_detections.clear()
        self.all_items = set(range(self.det_num))
        self.matched_items = set(self.assignment)

        if self.det_num > self.trk_num:
            self.unmatched_detections = self.all_items - self.matched_items
        else:
            for i in range(self.trk_num):
                if self.assignment[i] == -1:
                    self.unmatched_trajectories.add(i)

        # 筛选出 IoU 较低的匹配项
        self.matched_pairs = []
        for i in range(self.trk_num):
            if self.assignment[i] == -1:
                continue
            if 1 - self.iou_matrix[i][self.assignment[i]] < self.iou_threshold:
                self.unmatched_trajectories.add(i)
                self.unmatched_detections.add(self.assignment[i])
            else:
                self.matched_pairs.append((i, self.assignment[i]))

        # 更新匹配的跟踪链接
        for trk_idx, det_idx in self.matched_pairs:
            self.trackers[trk_idx].update(det_frame_data[det_idx]['box'])

        # 创建和初始化新的检测跟踪链接
        for det_idx in self.unmatched_detections:
            new_tracker = FaceTracker(det_frame_data[det_idx]['box'])
            self.trackers.append(new_tracker)

        # 准备跟踪链接的输出
        self.frame_tracking_result = []
        for tracker in self.trackers[:]:
            if tracker.time_since_update < 2 and (
                    tracker.continual_hits >= self.min_hits or self.frame_count <= self.min_hits):
                res = TrackingBox(box=tracker.last_state, id=tracker.id + 1, frame=self.frame_count)
                self.frame_tracking_result.append(res)

            # 删除过期的跟踪链接
            if tracker.time_since_update > self.max_missing_frames:
                self.trackers.remove(tracker)

        self.cycle_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        self.total_time += self.cycle_time

        return self.frame_tracking_result

    def remove_inactive_trackers(self):
        # 根据需要实施删除非活跃跟踪链接的逻辑
        pass
