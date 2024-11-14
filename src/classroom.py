from typing import List, Tuple, Any

import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
import time


class Student:
    def __init__(self, student_id):
        self.student_state_region = {}  # 学生的坐、站区域
        self.last_state = None
        self.state_flag = None
        self.id = student_id  # 学生 ID
        self.bbox = None  # 学生的边界框
        self.frame_num = None  # 第几帧检测到
        self.is_used = False
        self.active = False  # 是否活跃  该信息目前没有用到
        self.last_active_frame = None  # 上一次活动的帧数
        self.is_occluded = False  # 是否被遮挡  该信息目前没有用到
        self.state = -1  # -1 不稳定，0 未知，1 坐，2 站立
        self.student_count_map = np.zeros((360, 640), dtype=np.uint16)  # 表示该像素位置出现的目标（检测框中心）的次数

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
        # 更新学生起坐状态
        self.update_state(self.bbox[0] + self.bbox[2] // 2, self.bbox[1] + self.bbox[3] // 2, self.bbox[3])
        # 更新坐、站的区域
        self.update_state_region(self.student_count_map, self.bbox, self.state)

    def update_state_region(self, state_map, bbox, state):
        """
        更新学生的坐、站区域
        :param state_map: 学生的状态地图
        :param bbox: 学生的边界框
        :param state: 学生的状态
        """
        if state not in [2, 3]:
            return
        xmin, ymin, w, h = bbox
        xmax, ymax = xmin + w, ymin + h

        # 获取 bbox 附近的区域
        nearby_region = state_map[max(0, ymin - 10):min(ymax + 10, state_map.shape[0]),
                        max(0, xmin - 10):min(xmax + 10, state_map.shape[1])]

        # 找到数值大于 5 的位置
        sitting_area = np.argwhere(nearby_region > 5)

        # 更新区域的边界
        if sitting_area.size > 0:
            # 将 nearby_region 的局部坐标转换为全局坐标
            # 转换为标量
            y_min, x_min = np.min(sitting_area, axis=0).item(0) + max(0, ymin - 10), np.min(sitting_area, axis=1).item(
                0) + max(0, xmin - 10)
            y_max, x_max = np.max(sitting_area, axis=0).item(0) + max(0, ymin - 10), np.max(sitting_area, axis=1).item(
                0) + max(0, xmin - 10)

            new_square = [y_min, x_min, y_max, x_max]
            print(f"更新学生 {self.id} 的 {state} 区域：{new_square}")

            # 检查是否已有该状态的区域，若有则扩展范围
            if state in self.student_state_region:
                old_square = self.student_state_region[state]
                updated_square = [
                    min(old_square[0], new_square[0]), min(old_square[1], new_square[1]),
                    max(old_square[2], new_square[2]), max(old_square[3], new_square[3])
                ]
                self.student_state_region[state] = updated_square
            else:
                # 如果该状态还未记录，直接添加新区域
                self.student_state_region[state] = new_square

    def update_state(self, center_x, center_y, box_height):
        # 确保索引在有效范围内
        min_y, max_y = max(center_y - 10, 0), min(center_y + 10, self.student_count_map.shape[0])
        min_x, max_x = max(center_x - 10, 0), min(center_x + 10, self.student_count_map.shape[1])

        # 更新当前中心位置的计数
        self.student_count_map[center_y, center_x] += 1

        max_count = np.max(self.student_count_map[min_y:max_y, min_x:max_x])
        distance_moved = abs(center_y - self.last_center_y) if hasattr(self, 'last_center_y') else 0

        # 状态更新逻辑 (减少重复判断)
        if self.state == -1 and max_count >= 20:
            self.state = 0
        elif self.state == 0 and max_count < 5:
            self.state = 1
            self.last_center_y = center_y
        elif self.state == 1 and max_count >= 5:
            if distance_moved > 0.8 * box_height:
                self.state = 2 if center_y < self.last_center_y else 3
            else:
                self.state = 0 if not self.state_flag else self.last_state
            self.last_center_y = center_y
        elif self.state in [2, 3]:
            self.state_flag = True
            if max_count < 5:
                self.last_state = self.state
                self.state = 1

    def set_thresholds(self, stable_threshold=30, unstable_threshold=10):
        self.stable_threshold = stable_threshold
        self.unstable_threshold = unstable_threshold


class Classroom:
    def __init__(self, max_inactive_frames=15):
        self.id_map = np.zeros((640, 360), dtype=np.uint16)
        self.id_width_height_map = np.zeros((640, 360), dtype=np.float32)
        self.students = []
        self.current_frame = 0  # 当前帧数
        self.used_ids = [Student(i) for i in range(60)]  # 最多 60 个学生
        self.max_inactive_frames = max_inactive_frames  # 如果在 max_inactive_frames 内没有更新，则认为学生离开了
        self.interested_area = 50
        self.heat_map = None  # 学生的热力图

    def reset(self):
        """
        重置教室
        """
        self.id_map = np.zeros((640, 360), dtype=np.uint16)
        self.id_width_height_map = np.zeros((640, 360), dtype=np.float32)
        self.students = []
        self.current_frame = 0
        self.used_ids = [Student(i) for i in range(60)]

    def update(self, bboxes: np.ndarray) -> None:
        """
        更新学生信息
        :param bboxes: 学生的边界框列表
        """
        # 每隔 30 秒
        if self.current_frame % 900 == 0:
            # 重置教室
            self.reset()

        self.current_frame += 1

        iou_threshold = 0.5  # 两个边界框的 IoU 阈值

        # 常规情况：逐个检测框与 ID Map 进行匹配
        unassigned_students = []
        assigned_detect_bboxes = []
        start_time = time.time()
        for i, student in enumerate(self.students):
            # 检测 student 周围的检测框
            xmin, ymin, xmax, ymax = (student.bbox[0], student.bbox[1],
                                      student.bbox[0] + student.bbox[2], student.bbox[1] + student.bbox[3])
            x1, y1, x2, y2 = (max(0, xmin - (xmax - xmin) // 2), max(0, ymin - (ymax - ymin) // 2),
                              min(640, xmax + (xmax - xmin) // 2), min(360, ymax + (ymax - ymin) // 2))
            # 找出与感兴趣区域有交集的检测框
            interested_bboxes = bboxes[(bboxes[:, 0] < x2) & (bboxes[:, 1] < y2) &
                                       (bboxes[:, 0] + bboxes[:, 2] > x1) & (bboxes[:, 1] + bboxes[:, 3] > y1)]

            # 将有交集的检测框与感兴趣区域进行 IoU 计算，找出 IoU 最大的检测框
            max_iou = 0
            max_iou_bbox = None
            for bbox in interested_bboxes:
                iou = bbox_iou(bbox, student.bbox)
                if iou > max_iou:
                    max_iou = iou
                    max_iou_bbox = bbox
                    if max_iou >= iou_threshold:
                        break
            if max_iou > iou_threshold:
                # 找到了 IoU 最大的检测框，将该检测框与学生绑定
                student.update(max_iou_bbox, self.current_frame)
                self.update_id_map_and_width_height_map(*max_iou_bbox, student.id)
                # 热力图更新，检测框的中心位置加 1
                assigned_detect_bboxes.append(max_iou_bbox)
            else:
                # 没有找到 IoU 最大的检测框，将该学生标记为遮挡
                print(f"Student {student.id} is occluded")
                student.is_occluded = True
                unassigned_students.append(student)
        print(f"匹配时间：{(time.time() - start_time) * 1000:.3f} ms")

        # 新目标检测框ID赋值(即没有匹配到ID的目标框)
        start_time = time.time()
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

            # 转换为 NumPy 数组并提前计算
            assigned_detect_bboxes = np.array(assigned_detect_bboxes)

            # 创建未分配检测框的掩码，并计算未分配的检测框
            mask = np.array([not np.any(np.all(bbox == assigned_detect_bboxes, axis=1)) for bbox in bboxes])
            unassigned_detect_bboxes = bboxes[mask]
            print(f"未分配的检测框数：{unassigned_detect_bboxes.shape[0]}")

            # 若没有未分配的检测框，提前退出
            if unassigned_detect_bboxes.size == 0:
                print("未分配的检测框为空，无法继续分配！")
                return

            # 提前计算未分配检测框的面积
            unassigned_areas = unassigned_detect_bboxes[:, 2] * unassigned_detect_bboxes[:, 3]

            for i, student in enumerate(unassigned_students):
                # 计算学生的面积和对角线长度，提前计算减少重复
                student_area = student.bbox[2] * student.bbox[3]
                student_diag = np.sqrt(student.bbox[2] ** 2 + student.bbox[3] ** 2)

                # 计算所有未分配检测框与当前学生的距离
                distances = np.sqrt(np.sum((unassigned_detect_bboxes - student.bbox) ** 2, axis=1))
                min_distance_index = np.argmin(distances)

                # 计算面积比率
                area_ratio = student_area / unassigned_areas[min_distance_index]

                # 若距离小于对角线的 2/3 且面积比率接近 1，则匹配
                if distances[min_distance_index] < student_diag * 2 / 3 and area_ratio > 0.9:
                    min_distance_bbox = unassigned_detect_bboxes[min_distance_index]

                    # 更新学生信息
                    student.update(min_distance_bbox, self.current_frame)
                    self.update_id_map_and_width_height_map(*min_distance_bbox, student.id)

                    # 移除已分配的检测框
                    unassigned_detect_bboxes = np.delete(unassigned_detect_bboxes, min_distance_index, axis=0)
                    unassigned_areas = np.delete(unassigned_areas, min_distance_index)

                # 若未分配检测框为空，提前退出
                if unassigned_detect_bboxes.size == 0:
                    break

            print(f"未匹配 ID 再分配时间：{(time.time() - start_time) * 1000:.3f} ms")

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
