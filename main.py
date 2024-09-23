import os
import cv2
import numpy as np
from pathlib import Path
import argparse

from src.scrfd import SCRFD
from src.multiobject_tracker import MultiObjectTracker
from src.utils import is_image_file, is_video_file, print_progress_bar
from src.cluster import cluster_faces_for_id

clustering_done = False
initial_face_boxes = []  # 存储初始检测框
clustered_rects = []  # 存储聚类后的检测框
face_trackers = {}  # 存储每个ID对应的追踪器
face_tracker_ids = {}  # 存储每个追踪器对应的ID


def process_file(file_path: Path, detector: SCRFD, result_dir: Path) -> None:
    if is_image_file(str(file_path)):
        img = cv2.imread(str(file_path))
        print(f"[INFO] Processing image: {file_path}")
        if img is None:
            print(f"[ERROR] cv2.imread {file_path} failed")
            return

        face_objects = detector.detect(img)
        detector.draw(img, *face_objects)  # 绘制人脸框

        output_path = result_dir / (Path(file_path).stem + ".png")
        cv2.imwrite(str(output_path), img)

        print(f"[INFO] Image saved to: {output_path}")

    elif is_video_file(file_path):
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            return

        interval = 1
        standard = 60  # 设定的标准帧数量
        processed_frames = 0
        update_interval = 10  # 更新进度条

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_path = result_dir / (Path(file_path).stem + ".mp4")
        video = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'MP4V'),
                                cap.get(cv2.CAP_PROP_FPS), (frame_width, frame_height))

        num_clusters = 0
        prob_threshold = 0.3  # 人脸置信度阈值
        nms_threshold = 0.3  # 非极大值抑制阈值

        tracker = MultiObjectTracker()
        id_to_boxes = {}
        id_to_standard_box = {}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            det_frame_data = []

            if processed_frames <= standard:
                face_objects = detector.detect(frame, [], prob_threshold, nms_threshold)
                detector.draw(frame, face_objects)

                detected_boxes = [face.rect for face in face_objects]

                for i, box in enumerate(detected_boxes):
                    cur_box = {
                        'box': box,
                        'id': i,
                        'frame': processed_frames
                    }
                    det_frame_data.append(cur_box)

                tracking_results = tracker.update(det_frame_data)

                for it in tracking_results:
                    cv2.rectangle(frame, it['box'], (255, 0, 255), 2)
                    label = f"ID: {it['id']}"
                    cv2.putText(frame, label, (it['box'].x, it['box'].y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255))
                    id_to_boxes.setdefault(it['id'], []).append(it['box'])

                initial_face_boxes.append(face_objects)

                if processed_frames == standard:
                    for id, boxes in id_to_boxes.items():
                        num_clusters = 1  # 根据ID的检测框进行聚类
                        standard_box = cluster_faces_for_id(boxes, num_clusters)
                        id_to_standard_box[id] = standard_box
                    global clustering_done
                    clustering_done = True
            else:
                if (processed_frames - standard) % interval == 0:
                    detector.detect(frame, face_objects, prob_threshold, nms_threshold)
                    detector.draw(frame, face_objects)

                if clustering_done:
                    for id, standard_box in id_to_standard_box.items():
                        cv2.rectangle(frame, standard_box, (0, 255, 0), 2)
                        label = f"Standard ID: {id}"
                        cv2.putText(frame, label, (standard_box.x, standard_box.y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            video.write(frame)

            # 更新进度条
            processed_frames += 1
            if processed_frames % update_interval == 0:
                progress = processed_frames / total_frames
                print_progress_bar(progress)

        cap.release()
        video.release()

        print(f"[INFO] Video saved to: {output_path}")


def main(input_path, onnxmodel_path, result_dir=Path("result"), prob_threshold=0.5, nms_threshold=0.4):
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

    # for entry in Path(dir_path).iterdir():  # 遍历目录
    #     if entry.is_file():
    #         process_file(entry, detector, result_dir)

    process_file(Path(input_path), detector, Path(result_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/selfie.jpg", help="input image or video path")
    parser.add_argument("--onnxmodel", type=str, default="weights/scrfd_10g_kps.onnx", help="onnx model path")
    parser.add_argument("--prob_threshold", type=float, default=0.5, help="face detection probability threshold")
    parser.add_argument("--nms_threshold", type=float, default=0.4, help="non-maximum suppression threshold")
    parser.add_argument("--result_dir", type=str, default="result", help="result directory")
    parser.add_argument('--save', type=bool, default=True, help='Save detection results (image or video)')
    parser.add_argument('--confThreshold', default=0.5, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=0.4, type=float, help='nms iou thresh')
    args = parser.parse_args()
    print(args)

    main(args.input, args.onnxmodel, args.result_dir, args.confThreshold, args.nmsThreshold)
