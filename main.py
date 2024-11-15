import os
import cv2
import numpy as np
from pathlib import Path
import argparse
import time
from datetime import datetime

from ultralytics import YOLO
import src.scrfd as scrfd
from src.scrfd import SCRFD, draw
from src.utils import is_image_file, is_video_file, print_progress_bar
from src.classroom import Classroom

clustering_done = False
initial_face_boxes = []  # 存储初始检测框
clustered_rects = []  # 存储聚类后的检测框
face_trackers = {}  # 存储每个ID对应的追踪器
face_tracker_ids = {}  # 存储每个追踪器对应的ID
import cv2
import numpy as np


def detect_seat_regions(heat_map, threshold=10):
    heat_map = cv2.GaussianBlur(heat_map, (3, 3), 0)  # 滤波
    heat_map = cv2.convertScaleAbs(heat_map)
    # 二值化处理，找到热力图中活跃区域
    _, binary_map = cv2.threshold(heat_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 查找轮廓
    contours, _ = cv2.findContours(binary_map.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    seat_regions = []

    for contour in contours:
        # 忽略过小的区域，以避免噪声
        if cv2.contourArea(contour) < 1:  # 根据需要调整面积阈值
            continue
        # 计算坐标
        x, y, w, h = cv2.boundingRect(contour)
        # 左上角坐标 (x, y) 和右下角坐标 (x + w, y + h)
        seat_regions.append(((x, y), (x + w, y + h)))
    print(f"检测到{len(seat_regions)}个座位区域")

    return seat_regions


def process_file(file_path: Path, detector: YOLO, result_dir: Path, save=False) -> None:
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
        current_date = datetime.now().strftime("%Y-%m-%d")
        output_path = output_path.with_suffix(f".{current_date}.png")  # 加上时间戳
        cv2.imwrite(str(output_path), img)

        print(f"[INFO] Image saved to: {output_path}")

    elif is_video_file(str(file_path)):
        cap = cv2.VideoCapture(str(file_path))  # 打开视频文件
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        interval = 1
        standard = 60  # 设定的标准帧数量
        processed_frames = 0  # 已处理的帧数
        update_interval = 10  # 更新进度条

        if save:
            output_path = result_dir / (file_path.stem + ".mp4")
            current_date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            output_path = output_path.with_suffix(f".{current_date}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 四字符编码
            video = cv2.VideoWriter(str(output_path), fourcc, cap.get(cv2.CAP_PROP_FPS), (640, 360))

        heat_map = np.zeros((360, 640), dtype=np.float32)
        seat_regions = None

        while cap.isOpened():
            processed_frames += 1
            ret, srcimg = cap.read()  # 每次读取一帧

            if not ret:
                break  # 如果没有读取到帧，退出

            # 缩放到 640x360
            srcimg = cv2.resize(srcimg, (640, 360))

            # -----------------------------------------------检测人脸------------------------------------#
            start_time = time.time()
            results = detector.predict(srcimg, show=False, save=False, save_txt=False, classes=[0], visualize=False,
                                       device='0')
            bboxes = []
            for result in results:
                for box in result.boxes:
                    # 提取边界框的坐标并构建为 [x_min, y_min, x_max, y_max] 的格式
                    x_min = int(box.xyxy[0][0])
                    y_min = int(box.xyxy[0][1])
                    w = int(box.xyxy[0][2] - box.xyxy[0][0])
                    h = int(box.xyxy[0][3] - box.xyxy[0][1])

                    if heat_map[y_min + h // 2, x_min + w // 2,] < 60:
                        heat_map[y_min + h // 2, x_min + w // 2,] += 1  # 热力图更新  不能一直累加

                    # 添加到 bboxes 列表中
                    bboxes.append([x_min, y_min, w, h])
            bboxes = np.array(bboxes)
            print("-----------------------------------------------")
            print(f"检测用时：{(time.time() - start_time) * 1000:.3f}ms")

            # ----------------------------------------------座位确定-------#
            start_time = time.time()
            # heat_map 不能一直累加
            if processed_frames % 900 == 0:
                heat_map = np.maximum(heat_map - 2, 0)
                # # 将图像保存到本地heat_map 文件夹下，用帧数命名，格式为npy
                # np.save(f"heat_map/{processed_frames}.npy", heat_map)

            if processed_frames % 60 == 0:
                classroom.heat_map = heat_map
                seat_regions = detect_seat_regions(heat_map)

            print(f"座位确定用时：{(time.time() - start_time) * 1000:.3f}ms")
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
            draw(srcimg, classroom.students)  # 绘制人脸框
            if seat_regions:
                for region in seat_regions:
                    cv2.rectangle(srcimg, region[0], region[1], (0, 255, 0), 2)  # 绘制座位区域
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
            # 保存热力图
            min_val = np.min(heat_map)
            max_val = np.max(heat_map)
            print(f"min_val: {min_val}, max_val: {max_val}")

            normalized_heat_map = (heat_map - min_val) * 255.0 / (max_val - min_val)
            heat_map = cv2.normalize(normalized_heat_map, None, 0, 255, cv2.NORM_MINMAX)  # 归一化
            heat_map = cv2.applyColorMap(heat_map.astype(np.uint8), cv2.COLORMAP_HOT)  # 颜色映射

            cv2.imwrite(str(result_dir / "heat_map.png"), heat_map)

            print(f"座位个数：{len(seat_regions)}")


def main(input_path, onnxmodel_path, result_dir=Path("result"), prob_threshold=0.5, nms_threshold=0.4, save=False):
    if not os.path.exists(onnxmodel_path):
        print("[ERROR] Model File or Param File does not exist")
        return

    detector = YOLO("weights/yolov10n.pt")

    if os.path.isfile(input_path):  # 如果输入的是文件
        process_file(Path(input_path), detector, Path(result_dir), save=save)
    else:  # 如果输入的是目录
        for entry in Path(input_path).iterdir():  # 遍历目录
            if entry.is_file():
                process_file(Path(entry), detector, Path(result_dir), save=save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/正面教室.mp4", help="input image or video path")
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
