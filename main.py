import os
import cv2
import numpy as np
from pathlib import Path
import argparse

from src.scrfd import SCRFD
from src.utils import is_image_file, is_video_file, print_progress_bar
from src.cluster import cluster_faces_for_id
from src.my_tracker import Tracker

clustering_done = False
initial_face_boxes = []  # 存储初始检测框
clustered_rects = []  # 存储聚类后的检测框
face_trackers = {}  # 存储每个ID对应的追踪器
face_tracker_ids = {}  # 存储每个追踪器对应的ID


def process_file(file_path: Path, detector: SCRFD, result_dir: Path, save=False) -> None:
    tracker = Tracker()  # 追踪器初始化

    if is_image_file(str(file_path)):
        img = cv2.imread(str(file_path))
        print(f"[INFO] Processing image: {file_path}")
        if img is None:
            print(f"[ERROR] cv2.imread {file_path} failed")
            return

        bboxes, kpss, scores = detector.detect(img)
        tracker.update(bboxes, kpss, scores)  # 更新追踪器

        # cv2.imshow("ID Map", id_map)
        # cv2.imshow("Width/Height Map", width_height_map)
        # cv2.waitKey(0)

        # 获取每个框的信息
        boxes = [student.bbox for student in tracker.student_trackers]
        kpss = [student.kps for student in tracker.student_trackers]
        ids = [student.id for student in tracker.student_trackers]

        detector.draw(img, np.array(boxes), np.array(kpss), np.array(ids))  # 绘制人脸框

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

            bboxes, kpss, scores = detector.detect(srcimg)  # 检测人脸
            tracker.update(bboxes, kpss, scores)  # 更新追踪器

            boxes = [student.bbox for student in tracker.student_trackers]
            kpss = [student.kps for student in tracker.student_trackers]
            ids = [student.id for student in tracker.student_trackers]

            outimg = detector.draw(srcimg, np.array(boxes), np.array(kpss), np.array(ids))  # 绘制人脸框

            cv2.imshow('Deep learning object detection in OpenCV', outimg)  # 显示检测结果

            if save:
                video.write(outimg)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 键退出
                break

        cap.release()  # 释放视频捕获对象
        if save:
            video.release()  # 释放视频写入对象
            print(f"[INFO] Video saved to: {output_path}")

        # output_path = result_dir / (Path(file_path).stem + ".mp4")
        # video = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'MP4V'),
        #                         cap.get(cv2.CAP_PROP_FPS), (frame_width, frame_height))
        #
        # num_clusters = 0
        # # prob_threshold = 0.3  # 人脸置信度阈值
        # # nms_threshold = 0.3  # 非极大值抑制阈值
        #
        # tracker = MultiObjectTracker()  # 追踪器初始化
        # id_to_boxes = {}  # 存储每个ID对应的检测框
        # id_to_standard_box = {}  # 存储每个ID对应的标准框
        #
        # while cap.isOpened():
        #     ret, frame = cap.read()
        #     if not ret:
        #         break
        #
        #     det_frame_data = []  # 存储当前帧的检测框信息
        #
        #     # 如果当前帧数小于标准帧数，则进行检测，否则进行追踪，即每隔 standard 帧进行一次追踪
        #     if processed_frames <= standard:  # 如果在标准帧范围之内，进行检测并存下检测信息用作后续聚类
        #         boxes, kpss, scores = detector.detect(frame)  # 检测人脸
        #         detector.draw(frame, boxes, kpss, scores)  # 在当前帧绘制检测框
        #
        #         detected_boxes = boxes
        #
        #         for i, box in enumerate(detected_boxes):
        #             cur_box = {
        #                 'box': box,
        #                 'id': i,
        #                 'frame': processed_frames
        #             }
        #             det_frame_data.append(cur_box)  # 存下当前帧的检测框信息
        #
        #         tracking_results = tracker.update(det_frame_data)  # 进行追踪
        #
        #         for it in tracking_results:
        #             cv2.rectangle(frame, it.box, (255, 0, 255), 2)
        #             label = f"ID: {it.id}"
        #             cv2.putText(frame, label, (it.box[0], it.box[1]),
        #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255))
        #             id_to_boxes.setdefault(it.id, []).append(it.box)
        #
        #         initial_face_boxes.append([boxes, kpss, scores])
        #
        #         if processed_frames == standard:
        #             for id, boxes in id_to_boxes.items():
        #                 num_clusters = 1  # 根据ID的检测框进行聚类
        #                 standard_box = cluster_faces_for_id(boxes, num_clusters)
        #                 id_to_standard_box[id] = standard_box
        #             global clustering_done
        #             clustering_done = True
        #     else:
        #         if (processed_frames - standard) % interval == 0:
        #             detector.detect(frame)
        #             detector.draw(frame)
        #
        #         if clustering_done:
        #             for id, standard_box in id_to_standard_box.items():
        #                 cv2.rectangle(frame, standard_box, (0, 255, 0), 2)
        #                 label = f"Standard ID: {id}"
        #                 cv2.putText(frame, label, (standard_box.x, standard_box.y),
        #                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        #
        #     video.write(frame)
        #
        #     # 更新进度条
        #     processed_frames += 1
        #     if processed_frames % update_interval == 0:
        #         progress = processed_frames / total_frames
        #         print_progress_bar(progress)
        #
        # cap.release()
        # video.release()
        #
        # print(f"[INFO] Video saved to: {output_path}")


def main(input_path, onnxmodel_path, result_dir=Path("result"), prob_threshold=0.5, nms_threshold=0.4, save=False):
    # if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
    #     print(f"[ERROR] File does not exist: {dir_path}")
    #     return

    if not os.path.exists(onnxmodel_path):
        print("[ERROR] Model File or Param File does not exist")
        return

    # result_dir = Path(dir_path) / "result"
    # if not result_dir.exists():
    #     result_dir.mkdir()

    detector = SCRFD(onnxmodel_path, prob_threshold, nms_threshold)

    if os.path.isfile(input_path):  # 如果输入的是文件
        process_file(Path(input_path), detector, Path(result_dir), save=save)
    else:  # 如果输入的是目录
        for entry in Path(input_path).iterdir():  # 遍历目录
            if entry.is_file():
                process_file(Path(entry), detector, Path(result_dir), save=save)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/classroom_20s.mp4", help="input image or video path")
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


