import numpy as np
import cv2
import onnxruntime as ort


class FaceObject:
    def __init__(self, rect, prob, landmark=None):
        self.rect = rect  # Rect means (x, y, width, height)
        self.prob = prob
        self.landmark = landmark if landmark is not None else []


def intersection_area(a, b):
    inter_x = max(a[0], b[0])
    inter_y = max(a[1], b[1])
    inter_w = min(a[0] + a[2], b[0] + b[2]) - inter_x
    inter_h = min(a[1] + a[3], b[1] + b[3]) - inter_y

    if inter_w <= 0 or inter_h <= 0:
        return 0
    return inter_w * inter_h


def qsort_descent_inplace(faceobjects):
    if not faceobjects:
        return []
    return sorted(faceobjects, key=lambda x: x.prob, reverse=True)


def nms_sorted_bboxes(faceobjects, nms_threshold):
    picked = []
    n = len(faceobjects)

    areas = [obj.rect[2] * obj.rect[3] for obj in faceobjects]

    for i in range(n):
        a = faceobjects[i]
        keep = True
        for j in picked:
            b = faceobjects[j]
            inter_area = intersection_area(a.rect, b.rect)
            union_area = areas[i] + areas[j] - inter_area
            if inter_area / union_area > nms_threshold:
                keep = False
                break
        if keep:
            picked.append(i)

    return picked


def generate_proposals(anchors, score_blob, bbox_blob, prob_threshold):
    faceobjects = []

    # 确定 anchors 的数量
    num_anchors = anchors.shape[0]

    # 检查 score_blob 维数
    if score_blob.ndim == 2:  # 假设为 (1, 类别数) 的形状
        for q in range(num_anchors):
            anchor = anchors[q]
            prob = score_blob[0, q]  # 取出当前锚框的得分

            if prob >= prob_threshold:
                # 处理 bbox_blob（需要根据实际形状调整）
                bbox = bbox_blob[q]  # 假设 bbox_blob 的形状可以直接索引
                dx, dy, dw, dh = bbox  # 提取相关的偏移量

                # 计算中心
                cx = anchor[0] + anchor[2] / 2
                cy = anchor[1] + anchor[3] / 2

                # 计算边界框
                x0 = cx - dx
                y0 = cy - dy
                x1 = cx + dw
                y1 = cy + dh

                faceobjects.append(FaceObject(rect=[x0, y0, x1 - x0, y1 - y0], prob=prob))

    else:
        # 如果 score_blob 具有其他形状，则需适应这种情况
        (h, w) = score_blob.shape[2:4]  # 可能需要根据实际形状调整获取高度和宽度逻辑
        for q in range(num_anchors):
            anchor = anchors[q]
            score = score_blob[q, 0, :, :].flatten()  # 或其他方式访问 score_blob
            bbox = bbox_blob[q].reshape(4, -1)  # 假设 bbox_blob 的形状支持这一访问

            for i in range(h):
                for j in range(w):
                    index = i * w + j
                    prob = score[index]
                    if prob >= prob_threshold:
                        dx, dy, dw, dh = bbox[:, index]

                        cx = anchor[0] + anchor[2] / 2
                        cy = anchor[1] + anchor[3] / 2

                        x0 = cx - dx
                        y0 = cy - dy
                        x1 = cx + dw
                        y1 = cy + dh

                        faceobjects.append(FaceObject(rect=[x0, y0, x1 - x0, y1 - y0], prob=prob))

    return faceobjects


def generate_anchors(base_size, ratios, scales):
    anchors = []

    for ratio in ratios:
        # 计算宽和高
        w = base_size * np.sqrt(ratio)
        h = base_size / np.sqrt(ratio)

        for scale in scales:
            # 根据缩放因子计算真实的锚框尺寸
            anchor_w = w * scale
            anchor_h = h * scale

            # 生成锚框的坐标（x, y, w, h）
            anchors.append([-anchor_w / 2, -anchor_h / 2, anchor_w / 2, anchor_h / 2])

    return np.array(anchors, dtype=np.float32)


class SCRFD:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)

    def detect(self, image, prob_threshold=0.5, nms_threshold=0.5):
        height, width = image.shape[:2]
        target_size = 640

        # Resize and pad image
        scale = min(target_size / width, target_size / height)
        new_w = int(width * scale)
        new_h = int(height * scale)
        padded_image = cv2.resize(image, (new_w, new_h))
        delta_w = (target_size - new_w) // 2
        delta_h = (target_size - new_h) // 2
        padded_image = cv2.copyMakeBorder(padded_image, delta_h, target_size - new_h - delta_h,
                                          delta_w, target_size - new_w - delta_w,
                                          cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # Normalize
        input_image = (padded_image - 127.5) / 128.0
        input_image = input_image.astype(np.float32)
        input_image = np.transpose(input_image, (2, 0, 1))  # Change to (C, H, W)
        input_image = np.expand_dims(input_image, axis=0)

        # Forward pass
        outputs = self.session.run(None, {'images': input_image})
        print(len(outputs))
        score_blob = outputs[0]
        bbox_blob = outputs[1]
        kps_blob = outputs[2]

        # Generate anchors (example parameters, you might need to change)
        anchors = generate_anchors(base_size=16, ratios=[1.0], scales=[1.0, 2.0])

        # Generate proposals
        faceobjects = generate_proposals(anchors, score_blob, bbox_blob, prob_threshold)

        # Sort and nms
        faceobjects = qsort_descent_inplace(faceobjects)
        picked_indices = nms_sorted_bboxes(faceobjects, nms_threshold)

        return [faceobjects[i] for i in picked_indices]


# Example usage
if __name__ == "__main__":
    model_path = "../assets/scrfd_10g_kps.onnx"  # 请替换为你的ONNX模型路径
    scrfd = SCRFD(model_path)

    image = cv2.imread("../test/1.png")  # 请替换为你的图像路径
    detections = scrfd.detect(image)

    for detection in detections:
        x, y, w, h = detection.rect
        cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
        cv2.putText(image, f"{detection.prob:.2f}", (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                    2)

    cv2.imshow("Detections", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
