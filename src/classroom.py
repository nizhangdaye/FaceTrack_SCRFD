from typing import List, Tuple, Any

import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
import time


# 检测到新学生时，创建学生对象，将 Student 与 ID 绑定
#


class ID:
    def __init__(self, id, is_active=False):
        self.id = id
        self.active = is_active  # 是否活跃
        self.last_active_frame = None  # 上一次活动帧数
        self.bbox = None  # id 的位置

    def release(self):
        """
        释放 ID，将 ID 状态设置为非活跃，并清空上一次活动帧数，但保留 ID 位置
        """
        self.active = False
        self.last_active_frame = None
        # self.bbox = None


class Student:
    def __init__(self, bbox, frame_num, new_id: ID):
        self.bbox = bbox
        self.frame_num = frame_num  # 第几帧检测到

        self.ID = new_id
        self.ID.bbox = bbox
        self.ID.last_active_frame = frame_num

    def update(self, bbox, frame_num):
        self.bbox = bbox
        self.frame_num = frame_num
        self.ID.last_active_frame = frame_num
        self.ID.bbox = bbox


class Classroom:
    def __init__(self, max_inactive_frames=15):
        self.id_map = np.zeros((640, 360), dtype=np.uint8)
        self.id_width_height_map = np.zeros((640, 360), dtype=np.float32)
        self.students = []
        self.current_frame = 0  # 当前帧数
        self.used_ids = [ID(i) for i in range(60)]  # 最多 60 个学生
        self.max_inactive_frames = max_inactive_frames  # 如果在 max_inactive_frames 内没有更新，则认为学生离开了

    def update(self, bboxes: np.ndarray) -> None:
        current_detections = bboxes

        #
        # 匹配当前帧的检测与记录的学生
        start_time = time.time()
        matches = match_detections_to_students(self.students, bboxes, threshold=50)
        print(f"匹配用时: {(time.time() - start_time) * 1000:.3f}ms")

        # 处理匹配的检测框
        start_time = time.time()
        for match in matches:
            student_index, detection_index = match
            student = self.students[student_index]
            detection = bboxes[detection_index]
            student.update(detection, self.current_frame)
            self.update_id_map_and_width_height_map(*detection, student.ID.id)
        print(f"处理匹配框用时: {(time.time() - start_time) * 1000:.3f}ms")

        # 处理未匹配的检测框  未匹配的检测框可视为新识别的学生
        start_time = time.time()
        matched_detections = set([m[1] for m in matches])  # 已经匹配的检测框
        unmatched_detections = set(range(len(bboxes))) - matched_detections  # 未匹配的检测框
        for detection_index in unmatched_detections:
            detection = bboxes[detection_index]
            new_ID = self.get_new_id()
            student = Student(detection, self.current_frame, new_ID)
            self.students.append(student)
            self.update_id_map_and_width_height_map(*detection, student.ID.id)
        print(f"处理未匹配框用时: {(time.time() - start_time) * 1000:.3f}ms")

        # 移除超过最大未活动帧数的学生，并释放其 ID
        start_time = time.time()
        active_trackers = []
        for student in self.students:
            if self.current_frame - student.frame_num < self.max_inactive_frames:
                active_trackers.append(student)
            else:
                student.ID.release()
        print(f"移除超过最大未活动帧数的学生用时: {(time.time() - start_time) * 1000:.3f}ms")

        self.students = active_trackers  # 更新学生列表

        self.current_frame += 1  # 更新帧数

    def get_new_id(self):
        """
        返回没有被使用的 ID
        """
        for i in range(len(self.used_ids)):
            if not self.used_ids[i].active:
                self.used_ids[i].active = True
                return self.used_ids[i]
        return -1

    def update_id_map_and_width_height_map(self, x, y, w, h, id_):
        x, y, w, h = int(x), int(y), int(w), int(h)
        self.id_width_height_map[y:y + h, x:x + w] = w * h
        self.id_map[y:y + h, x:x + w] = id_


def match_detections_to_students(tracks: list, detections: np.ndarray, threshold=10) -> list[tuple[Any, Any]]:
    """
    使用 Hungarian 算法匹配检测框与跟踪器
    """
    start_time = time.time()
    cost_matrix = compute_cost_matrix(tracks, detections)  # 计算匹配成本矩阵
    print(f"计算匹配成本矩阵用时: {(time.time() - start_time) * 1000:.3f}ms")
    start_time = time.time()
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=False)  # 使用 Hungarian 算法匹配
    print(f"使用 Hungarian 算法匹配用时: {(time.time() - start_time) * 1000:.3f}ms")
    matches = []

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < threshold:
            matches.append((r, c))

    return matches


def compute_cost_matrix(tracks: list, detections: np.ndarray, distance_weight=1, iou_weight=10) -> np.ndarray:
    """
    使用欧氏距离计算匹配成本矩阵
    :param tracks: 跟踪器的列表
    :param detections: 检测框的列表
    :return: 成本矩阵
    """
    cost_matrix = np.zeros((len(tracks), len(detections)))

    for i, track in enumerate(tracks):
        for j, detection in enumerate(detections):
            # 计算欧式距离
            distance = np.sqrt(
                (track.bbox[0] - detection[0]) ** 2 + (track.bbox[1] - detection[1]) ** 2
            )

            # 计算 IoU
            iou = bbox_iou(track.bbox, detection)

            # 将 IoU 转换为距离度量
            iou_cost = 1 - iou

            # 结合欧式距离和 IoU，按加权平均的方式
            cost_matrix[i, j] = distance_weight * distance + iou_weight * iou_cost

    return cost_matrix


def bbox_iou(bbox1, bbox2):
    # 计算两个边界框的交集区域
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
    y_bottom = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0  # 没有交集

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # 计算两个边界框的并集区域
    bbox1_area = bbox1[2] * bbox1[3]
    bbox2_area = bbox2[2] * bbox2[3]
    union_area = bbox1_area + bbox2_area - intersection_area

    # 计算 IoU
    iou = intersection_area / union_area
    return iou
