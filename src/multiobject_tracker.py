import numpy as np
import cv2
from src.face_tracker import FaceTracker


class MultiObjectTracker:
    def __init__(self, max_age, iouThreshold):
        self.max_missing_frames = max_age
        self.iouThreshold = iouThreshold
        self.trackers = []
        self.predicted_boxes = []
        self.total_frames = 0
        self.frame_count = 0
        self.frame_tracking_result = []
        self.min_hits = 3  # 根据需要设置最小命中次数
        self.cycle_time = 0
        self.total_time = 0

    def calculate_iou(self, rect1, rect2):
        inter_area = (rect1 & rect2).area()
        union_area = rect2.area() + rect1.area() - inter_area
        if union_area < np.finfo(float).eps:
            return 0
        return inter_area / union_area

    def update(self, det_frame_data):
        self.total_frames += 1
        self.frame_count += 1
        start_time = cv2.getTickCount()

        if len(self.trackers) == 0:  # initialize kalman trackers using first detections.
            for detection in det_frame_data:
                trk = FaceTracker(detection.box)
                self.trackers.append(trk)
            return []

        # get predicted locations from existing trackers.
        self.predicted_boxes.clear()

        for it in self.trackers[:]:
            p_box = it.predict()
            if p_box.x >= 0 and p_box.y >= 0:
                self.predicted_boxes.append(p_box)
            else:
                self.trackers.remove(it)

        # associate detections to tracked object (both represented as bounding boxes)
        trk_num = len(self.predicted_boxes)
        det_num = len(det_frame_data)

        iou_matrix = np.zeros((trk_num, det_num))

        for i in range(trk_num):  # compute iou matrix as a distance matrix
            for j in range(det_num):
                # use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
                iou_matrix[i][j] = 1 - self.calculate_iou(self.predicted_boxes[i], det_frame_data[j].box)

        # solve the assignment problem using hungarian algorithm
        hung_algo = HungarianAlgorithm()
        assignment = hung_algo.Solve(iou_matrix)

        unmatched_trajectories = set()
        unmatched_detections = set()
        matched_pairs = []

        if det_num > trk_num:
            all_items = set(range(det_num))
            matched_items = set(assignment)

            unmatched_detections = all_items - matched_items
        else:
            for i in range(trk_num):
                if assignment[i] == -1:  # unassigned label will be set as -1
                    unmatched_trajectories.add(i)

        # filter out matched with low IOU
        for i in range(trk_num):
            if assignment[i] == -1:
                continue

            if 1 - iou_matrix[i][assignment[i]] < self.iouThreshold:
                unmatched_trajectories.add(i)
                unmatched_detections.add(assignment[i])
            else:
                matched_pairs.append((i, assignment[i]))

        # updating trackers
        for trk_idx, det_idx in matched_pairs:
            self.trackers[trk_idx].update(det_frame_data[det_idx].box)

        # create and initialise new trackers for unmatched detections
        for umd in unmatched_detections:
            tracker = FaceTracker(det_frame_data[umd].box)
            self.trackers.append(tracker)

        # get trackers' output
        self.frame_tracking_result.clear()
        for it in self.trackers:
            if it.time_since_update < 2 and (it.continual_hits >= self.min_hits or self.frame_count <= self.min_hits):
                res = TrackingBox()
                res.box = it.lastState
                res.id = it.id + 1
                res.frame = self.frame_count
                self.frame_tracking_result.append(res)

        # remove dead tracklet
        self.trackers = [it for it in self.trackers if it.time_since_update <= self.max_missing_frames]

        self.cycle_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        self.total_time += self.cycle_time

        return self.frame_tracking_result

    def remove_inactive_trackers(self):
        pass  # 根据需要实现相应逻辑
