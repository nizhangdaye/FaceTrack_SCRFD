import numpy as np
import cv2


class StudentTracker:
    def __init__(self, bbox, kps, score, frame_num, id_):
        self.id = id_
        self.bbox = bbox
        self.kps = kps
        self.score = score
        self.last_update_frame = frame_num
        self.width_height_map = np.zeros((640, 360), dtype=np.float32)

    def update(self, bbox, kps, score, frame_num):
        self.bbox = bbox
        self.kps = kps
        self.score = score
        self.last_update_frame = frame_num


class Tracker:
    def __init__(self):
        self.id_map = np.zeros((640, 360), dtype=np.uint8)
        self.id_width_height_map = np.zeros((640, 360), dtype=np.float32)
        self.student_trackers = []
        self.next_id = 1
        self.current_frame = 0
        self.used_ids = [False] * 60  # 最多 60 个学生

    def get_new_id(self):
        """
        返回没有被使用的 ID
        """
        for i in range(len(self.used_ids)):
            if not self.used_ids[i]:
                self.used_ids[i] = True
                return i + 1
        return -1

    def update_id_map(self, x, y, w, h, id_):
        x, y, w, h = int(x), int(y), int(w), int(h)
        self.id_map[y:y + h, x:x + w] = id_

    def update_width_height_map(self, x, y, w, h, id_):
        x, y, w, h = int(x), int(y), int(w), int(h)
        self.id_width_height_map[y:y + h, x:x + w] = w * h

    def update(self, bboxes, kpss, scores):
        # 首先对检测框进行 ID 赋值
        self.student_trackers = []  # 清空之前的跟踪结果
        self.used_ids = [False] * 60  # 重置已使用的 ID 列表
        for i in range(len(bboxes)):
            new_id = self.get_new_id()
            student_tracker = StudentTracker(bboxes[i], kpss[i], scores[i], self.current_frame, new_id)
            self.student_trackers.append(student_tracker)
            self.update_id_map(bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3], new_id)  # 更新 ID 映射
            self.update_width_height_map(bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3], new_id)  # 更新宽高映射

    def is_close(self, bbox1, bbox2, threshold=50):
        # 判断两个检测框是否接近
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        center1 = (x1 + w1 // 2, y1 + h1 // 2)
        center2 = (x2 + w2 // 2, y2 + h2 // 2)
        distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
        return distance < threshold
