import os
import cv2
import numpy as np
from pathlib import Path
import argparse
import time

import src.scrfd as scrfd
from src.scrfd import SCRFD
from src.utils import is_image_file, is_video_file, print_progress_bar
from src.classroom import Classroom

clustering_done = False
initial_face_boxes = []  # 存储初始检测框
clustered_rects = []  # 存储聚类后的检测框
face_trackers = {}  # 存储每个ID对应的追踪器
face_tracker_ids = {}  # 存储每个追踪器对应的ID


def process_file(file_path: Path, detector: SCRFD, result_dir: Path, save=False) -> None:
    classroom = Classroom()  # 追踪器初始化

    if is_image_file(str(file_path)):
        # -------------------------------------检测人脸-------------------------------------------------#
        start_time = time.time()
        img = cv2.imread(str(file_path))
        # 缩放到 640x360
        img = cv2.resize(img, (640, 360))
        print(f"[INFO] Processing image: {file_path}")
        if img is None:
            print(f"[ERROR] cv2.imread {file_path} failed")
            return

        bboxes, kpss, scores = detector.detect(img)

        print(f"检测用时：{(time.time() - start_time) * 1000:.3f}ms")
        # ----------------------------------------------对人脸进行排序--------------------------------#
        start_time = time.time()
        bboxes = scrfd.sort_faces_by_row(bboxes)
        print(f"排序用时：{(time.time() - start_time) * 1000:.3f}ms")
        # --------------------------------------------更新教室信息------------------------------------#
        start_time = time.time()
        classroom.update(bboxes)  # 更新教室信息
        print(f"更新教室信息用时：{(time.time() - start_time) * 1000:.3f}ms")
        # -----------------------------------------------绘制人脸框---------------------------------------#
        # 获取每个框的信息
        start_time = time.time()
        boxes, ids = zip(*[(student.bbox, student.id) for student in classroom.students])
        detector.draw(img, np.array(boxes), np.array(ids))  # 绘制人脸框
        print(f"绘制用时：{(time.time() - start_time) * 1000:.3f}ms")
        # ----------------------------------------------------------------------------------------------#

        output_path = result_dir / (file_path.stem + ".png")
        cv2.imwrite(str(output_path), img)

        print(f"[INFO] Image saved to: {output_path}")

    elif is_video_file(str(file_path)):
        cap = cv2.VideoCapture(str(file_path))  # 打开视频文件
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        interval = 1
        standard = 60  # 设定的标准帧数量
        processed_frames = 0
        update_interval = 10  # 更新进度条

        if save:
            output_path = result_dir / (file_path.stem + ".mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 四字符编码
            video = cv2.VideoWriter(str(output_path), fourcc, cap.get(cv2.CAP_PROP_FPS), (640, 360))

        while cap.isOpened():
            ret, srcimg = cap.read()  # 每次读取一帧

            if not ret:
                break  # 如果没有读取到帧，退出

            # 缩放到 640x360
            srcimg = cv2.resize(srcimg, (640, 360))

            # -----------------------------------------------检测人脸------------------------------------#
            start_time = time.time()
            bboxes, kpss, scores = detector.detect(srcimg)
            print("-----------------------------------------------")
            print(f"检测用时：{(time.time() - start_time) * 1000:.3f}ms")
            # ----------------------------------------------对人脸进行排序--------------------------------#
            start_time = time.time()
            bboxes = scrfd.sort_faces_by_row(bboxes)
            print(f"排序用时：{(time.time() - start_time) * 1000:.3f}ms")
            # --------------------------------------------更新教室信息------------------------------------#
            start_time = time.time()
            classroom.update(bboxes)
            print(f"更新教室用时：{(time.time() - start_time) * 1000:.3f}ms")
            # -----------------------------------------------绘制人脸框------------------------------------#
            start_time = time.time()
            boxes, ids = zip(*[(student.bbox, student.id) for student in classroom.students])
            detector.draw(srcimg, np.array(boxes), np.array(ids))  # 绘制人脸框
            cv2.imshow('Deep learning object detection in OpenCV', srcimg)  # 显示检测结果
            print(f"绘制并显示用时：{(time.time() - start_time) * 1000:.3f}ms")
            # -----------------------------------------------保存视频--------------------------------------#
            if save:
                video.write(srcimg)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 键退出
                break

        cap.release()  # 释放视频捕获对象
        if save:
            video.release()  # 释放视频写入对象
            print(f"[INFO] Video saved to: {output_path}")


def main(input_path, onnxmodel_path, result_dir=Path("result"), prob_threshold=0.5, nms_threshold=0.4, save=False):
    if not os.path.exists(onnxmodel_path):
        print("[ERROR] Model File or Param File does not exist")
        return
    detector = SCRFD(onnxmodel_path, prob_threshold, nms_threshold)

    if os.path.isfile(input_path):  # 如果输入的是文件
        process_file(Path(input_path), detector, Path(result_dir), save=save)
    else:  # 如果输入的是目录
        for entry in Path(input_path).iterdir():  # 遍历目录
            if entry.is_file():
                process_file(Path(entry), detector, Path(result_dir), save=save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/理想场景/正面显示场景.mp4", help="input image or video path")
    parser.add_argument("--onnxmodel", type=str, default="weights/scrfd_10g_kps.onnx", help="onnx model path")
    parser.add_argument("--prob_threshold", type=float, default=0.5, help="face detection probability threshold")
    parser.add_argument("--nms_threshold", type=float, default=0.4, help="non-maximum suppression threshold")
    parser.add_argument("--result_dir", type=str, default="result", help="result directory")
    parser.add_argument('--save', type=bool, default=True, help='Save detection results (image or video)')
    parser.add_argument('--confThreshold', default=0.5, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=0.4, type=float, help='nms iou thresh')
    args = parser.parse_args()
    print(args)

    main(args.input, args.onnxmodel, args.result_dir, args.confThreshold, args.nmsThreshold, save=args.save)
