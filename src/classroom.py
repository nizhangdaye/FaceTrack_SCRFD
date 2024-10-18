from typing import List, Tuple, Any

import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
import time


class Student:
    def __init__(self, student_id):
        self.id = student_id  # 学生 ID
        self.bbox = None  # 学生的边界框
        self.frame_num = None  # 第几帧检测到
        self.is_used = False
        self.active = False  # 是否活跃
        self.last_active_frame = None  # 上一次活动的帧数
        self.is_occluded = False  # 是否被遮挡

    def release(self):
        """
        释放学生，将状态设置为非活跃，并清空上一次活动帧数
        """
        self.active = False
        self.last_active_frame = None

    def update(self, bbox, frame_num):
        """
        更新学生信息
        :param bbox: 学生的边界框
        :param frame_num: 当前帧数
        """
        self.bbox = bbox
        self.last_active_frame = frame_num
        self.active = True


class Classroom:
    def __init__(self, max_inactive_frames=15):
        self.id_map = np.zeros((640, 360), dtype=np.uint16)
        self.id_width_height_map = np.zeros((640, 360), dtype=np.float32)
        self.students = []
        self.current_frame = 0  # 当前帧数
        self.used_ids = [Student(i) for i in range(60)]  # 最多 60 个学生
        self.max_inactive_frames = max_inactive_frames  # 如果在 max_inactive_frames 内没有更新，则认为学生离开了
        self.interested_area = 50

    def update(self, bboxes: np.ndarray) -> None:
        """
        更新学生信息
        :param bboxes: 学生的边界框列表
        """
        self.current_frame += 1

        iou_threshold = 0.5  # 两个边界框的 IoU 阈值

        # 常规情况：逐个检测框与 ID Map 进行匹配
        unassigned_students = []
        assigned_detect_bboxes = []
        for i, student in enumerate(self.students):
            # 检测 student 周围的检测框
            xmin, ymin, xmax, ymax = (student.bbox[0], student.bbox[1],
                                      student.bbox[0] + student.bbox[2], student.bbox[1] + student.bbox[3])
            x1, y1, x2, y2 = (max(0, xmin - (xmax - xmin) // 2), max(0, ymin - (ymax - ymin) // 2),
                              min(640, xmax + (xmax - xmin) // 2), min(360, ymax + (ymax - ymin) // 2))
            # 找出与感兴趣区域有交集的检测框
            interested_bboxes = bboxes[np.where((bboxes[:, 0] >= x1) | (bboxes[:, 1] >= y1) |
                                                (bboxes[:, 2] <= x2) | (bboxes[:, 3] <= y2))]
            # 将有交集的检测框与感兴趣区域进行 IoU 计算，找出 IoU 最大的检测框
            max_iou = 0
            max_iou_bbox = None
            for bbox in interested_bboxes:
                iou = bbox_iou(bbox, student.bbox)
                if iou > max_iou:
                    max_iou = iou
                    max_iou_bbox = bbox
            if max_iou > iou_threshold:
                # 找到了 IoU 最大的检测框，将该检测框与学生绑定
                student.update(max_iou_bbox, self.current_frame)
                self.update_id_map_and_width_height_map(*max_iou_bbox, student.id)
                assigned_detect_bboxes.append(max_iou_bbox)
            else:
                # 没有找到 IoU 最大的检测框，将该学生标记为遮挡
                print(f"Student {student.id} is occluded")
                student.is_occluded = True
                unassigned_students.append(student)

        # 新目标检测框ID赋值(即没有匹配到ID的目标框)
        if not self.students:
            unassigned_bboxes = bboxes
            for bbox in unassigned_bboxes:
                # 遍历未分配的目标框，分配 ID
                # 没有找到空闲的 ID，则创建一个新的 ID
                new_id = self.get_new_id()
                if new_id == -1:
                    continue  # 没有空闲的 ID，跳过
                new_id.update(bbox, self.current_frame)
                self.students.append(new_id)
        else:
            print(f"未分配的 ID 数：{len(unassigned_students)}")
            assigned_detect_bboxes = np.array(assigned_detect_bboxes)
            mask = np.array([not np.any(np.all(bbox == assigned_detect_bboxes, axis=1)) for bbox in bboxes])
            unassigned_detect_bboxes = bboxes[mask]
            print(f"未分配的检测框数：{unassigned_detect_bboxes.shape[0]}")

            for i, student in enumerate(unassigned_students):
                if unassigned_detect_bboxes.size == 0:
                    print("未分配的检测框为空，无法计算距离！")
                    break
                # 找出与当前距离最近的未分配的检测框
                # 计算距离
                distances = np.sqrt(np.sum((unassigned_detect_bboxes - student.bbox) ** 2, axis=1))  # 计算距离
                min_distance_index = np.argmin(distances)  # 距离最小的索引
                # 计算面积
                area_ratio = student.bbox[2] * student.bbox[3] / (unassigned_detect_bboxes[min_distance_index][2] *
                                                                  unassigned_detect_bboxes[min_distance_index][3])
                # 若距离小于学生的高度且面积近似，认为是同一个目标
                if distances[min_distance_index] < np.sqrt(student.bbox[3]**2 + student.bbox[2]**2) and area_ratio > 0.8:
                    min_distance_bbox = unassigned_detect_bboxes[min_distance_index]  # 距离最近的检测框
                    # 更新学生信息
                    student.update(min_distance_bbox, self.current_frame)
                    self.update_id_map_and_width_height_map(*min_distance_bbox, student.id)
                    # 移除该检测框
                    unassigned_detect_bboxes = np.delete(unassigned_detect_bboxes, min_distance_index, axis=0)

    def get_new_id(self):
        """
        返回没有被使用的 ID
        """
        for i in range(len(self.used_ids)):
            if not self.used_ids[i].is_used:
                self.used_ids[i].is_used = True
                return self.used_ids[i]
        return -1

    def update_id_map_and_width_height_map(self, x, y, w, h, id_):
        # 没有考虑遮挡与被遮挡的情况
        x, y, w, h = int(x), int(y), int(w), int(h)
        self.id_width_height_map[y:y + h, x:x + w] = w * h
        self.id_map[y:y + h, x:x + w] = id_


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
