import cv2
import numpy as np


def cluster_faces_for_id(boxes, num_clusters):
    samples = np.zeros((len(boxes), 1, 2), dtype=np.float32)

    total_width = 0
    total_height = 0

    for i, box in enumerate(boxes):
        center_x = box[0] + box[2] / 2.0  # box[0] is x, box[2] is width
        center_y = box[1] + box[3] / 2.0  # box[1] is y, box[3] is height
        total_width += box[2]  # width
        total_height += box[3]  # height
        samples[i, 0, 0] = center_x
        samples[i, 0, 1] = center_y

    # K-means聚类
    _, labels, centers = cv2.kmeans(samples, num_clusters, None,
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 230, 0.1),
                                    5, cv2.KMEANS_PP_CENTERS)

    # 计算聚类中心的位置
    result_x = 0
    result_y = 0

    for i in range(centers.shape[0]):
        center = centers[i]
        result_x = center[0]
        result_y = center[1]

    avg_width = total_width // len(boxes) if len(boxes) > 0 else 0
    avg_height = total_height // len(boxes) if len(boxes) > 0 else 0

    cluster_center = centers[0]
    return (int(cluster_center[0] - avg_width / 2), int(cluster_center[1] - avg_height / 2), avg_width, avg_height)
