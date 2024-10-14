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

    def update(self, bbox, frame_num):
        """
        更新 ID 信息
        :param bbox: ID 的位置
        :param frame_num: 当前帧数
        """
        self.bbox = bbox
        self.last_active_frame = frame_num
        self.active = True


class Student:
    def __init__(self, bbox, frame_num, new_id: ID):
        self.bbox = bbox
        self.frame_num = frame_num  # 第几帧检测到

        self.ID = new_id
        self.ID.bbox = bbox
        self.ID.last_active_frame = frame_num
        self.is_occluded = False  # 是否被遮挡

    def update(self, bbox, frame_num):
        self.bbox = bbox
        self.frame_num = frame_num
        self.ID.last_active_frame = frame_num
        self.ID.bbox = bbox


class Classroom:
    def __init__(self, max_inactive_frames=15):
        self.id_map = np.zeros((640, 360), dtype=np.uint16)
        self.id_width_height_map = np.zeros((640, 360), dtype=np.float32)
        self.students = []
        self.current_frame = 0  # 当前帧数
        self.used_ids = [ID(i) for i in range(60)]  # 最多 60 个学生
        self.max_inactive_frames = max_inactive_frames  # 如果在 max_inactive_frames 内没有更新，则认为学生离开了
        self.interested_area = 20

    def update(self, bboxes: np.ndarray) -> None:
        """
        更新学生信息
        :param bboxes: 学生的边界框列表
        """
        self.current_frame += 1

        iou_threshold = 0.5  # 两个边界框的 IoU 阈值

        # 常规情况：逐个检测框与 ID Map 进行匹配
        unassigned_bboxes = []
        for i, student in enumerate(self.students):
            # 检测 student 周围的检测框
            xmin, ymin, xamx, ymax = student.bbox
            x1, y1, x2, y2 = (max(0, xmin - self.interested_area), max(0, ymin - self.interested_area),
                              min(640, xamx + self.interested_area), min(360, ymax + self.interested_area))
            # 找出与感兴趣区域有交集的检测框
            interested_bboxes = bboxes[
                np.where((bboxes[:, 0] >= x1) | (bboxes[:, 1] >= y1) | (bboxes[:, 2] <= x2) | (bboxes[:, 3] <= y2))]
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
                self.update_id_map_and_width_height_map(max_iou_bbox[0], max_iou_bbox[1], max_iou_bbox[2],
                                                        max_iou_bbox[3], student.ID.id)
            else:
                # 没有找到 IoU 最大的检测框，将该学生标记为遮挡
                student.is_occluded = True
                unassigned_bboxes.append(student.bbox)

        # # 处理遮挡学生
        # for student in self.students:
        #     if student.is_occluded:
        #         # 学生被遮挡，将其 ID 释放
        #         student.ID.release()
        #         student.is_occluded = False

        # # 处理失活学生
        # for i, student in enumerate(self.students):
        #     if self.current_frame - student.frame_num > self.max_inactive_frames:
        #         # 学生失活，将其 ID 释放
        #         student.ID.release()
        #         self.students.pop(i)

        # 新目标检测框ID赋值(即没有匹配到ID的目标框)
        if not self.students:
            unassigned_bboxes = bboxes
        print(f"未分配的目标框数：{len(unassigned_bboxes)}")
        for i, bbox in enumerate(unassigned_bboxes):
            # 如果 bbox 在 ID Map 范围内与 ID 有交集，若该 ID 失能，则将该 ID 与该 bbox 绑定
            xmin, ymin, xamx, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
            interested_area = self.id_map[max(0, ymin - self.interested_area):min(360, ymax + self.interested_area),
                              max(0, xmin - self.interested_area):min(640, xamx + self.interested_area)]
            interested_ids = np.unique(interested_area)  # 去重
            # ！！！！！！！！！！！！！！！！！！！！！可能会有多个失活 ID 的情况，所以需要遍历所有失活 ID，找出 IoU 最大的 ID 进行绑定
            max_iou = 0
            max_iou_id = -1
            for id_ in interested_ids:
                if id_ == 0:
                    continue  # 0 代表没有 ID
                if not self.used_ids[id_ - 1].active:
                    # 该 ID 失活，找出 IoU 最大的 ID 进行绑定
                    iou = bbox_iou(bbox, self.used_ids[id_ - 1].bbox)
                    if iou > max_iou:
                        max_iou = iou
                        max_iou_id = id_
            if max_iou_id != -1:
                # 找到了 IoU 最大的 ID，将该 ID 与该 bbox 绑定
                self.used_ids[max_iou_id - 1].active = True
                self.used_ids[max_iou_id - 1].bbox = bbox
                self.used_ids[max_iou_id - 1].last_active_frame = self.current_frame
                self.students.append(Student(bbox, self.current_frame, self.used_ids[max_iou_id - 1]))
            else:
                # 没有找到空闲的 ID，则创建一个新的 ID
                new_id = self.get_new_id()
                if new_id == -1:
                    print("No free ID available")
                    continue  # 没有空闲的 ID，跳过
                new_id.update(bbox, self.current_frame)
                self.students.append(Student(bbox, self.current_frame, new_id))

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
