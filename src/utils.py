import cv2
import numpy as np
import random
import os


def get_random_color():
    """生成随机颜色"""
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def is_image_file(filename):
    """判断文件是否为图像文件"""
    ext = filename.split('.')[-1].lower()
    return ext in ["jpg", "jpeg", "png", "bmp"]


def is_video_file(filename):
    """判断文件是否为视频文件"""
    ext = filename.split('.')[-1].lower()
    return ext in ["avi", "mp4", "mov", "mkv"]


def print_progress_bar(progress):
    """打印进度条"""
    bar_width = 70
    pos = int(bar_width * progress)
    bar = "[" + "=" * pos + ">" + " " * (bar_width - pos - 1) + "]"
    print(f"{bar} {int(progress * 100)} %", end='\r')
    if progress == 1:
        print()  # 完成时换行


def cluster_faces(face_boxes, num_clusters):
    """聚类人脸框"""
    clustered_rects = [None] * num_clusters
    if not face_boxes or num_clusters == 0:
        return clustered_rects

    samples = np.zeros((len(face_boxes), 2), dtype=np.float32)
    for i, face_group in enumerate(face_boxes):
        for face in face_group:
            center_x = (face['rect'][0] + face['rect'][2]) / 2.0
            center_y = (face['rect'][1] + face['rect'][3]) / 2.0
            samples[i] = [center_x, center_y]

    if samples.ndim == 2 and samples.dtype == np.float32 and num_clusters > 0:
        # 使用K-Means进行聚类
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)
        _, labels, centers = cv2.kmeans(samples, num_clusters, None, criteria, 5, cv2.KMEANS_PP_CENTERS)

        # 根据聚类结果计算每个聚类的平均位置
        cluster_counts = [0] * num_clusters
        for i in range(len(labels)):
            cluster_idx = labels[i][0]
            center = (centers[cluster_idx][0], centers[cluster_idx][1])
            w = face_boxes[i][0]['rect'][2] - face_boxes[i][0]['rect'][0]  # 宽度
            h = face_boxes[i][0]['rect'][3] - face_boxes[i][0]['rect'][1]  # 高度

            clustered_rects[cluster_idx] = (int(center[0] - w / 2), int(center[1] - h / 2), w, h)
            cluster_counts[cluster_idx] += 1
    else:
        print("K-means 输入数据无效.")

    return clustered_rects
