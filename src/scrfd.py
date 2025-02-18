import cv2
import argparse
import numpy as np
import os


class SCRFD():
    def __init__(self, onnxmodel_path: str, confThreshold=0.5, nmsThreshold=0.5):
        self.inpWidth = 640
        self.inpHeight = 640
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.net = cv2.dnn.readNet(onnxmodel_path)
        # self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)  # 使用CUDA作为后端
        # self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)  # 设置目标为CUDA
        self.keep_ratio = True
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2

    def resize_image(self, srcimg: np.ndarray) -> np.ndarray:
        """
        将图片缩放到指定大小
        :param srcimg:
        :return:
        """
        padh, padw, newh, neww = 0, 0, self.inpHeight, self.inpWidth
        if self.keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.inpHeight, int(self.inpWidth / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                padw = int((self.inpWidth - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, padw, self.inpWidth - neww - padw, cv2.BORDER_CONSTANT,
                                         value=0)  # add border
            else:
                newh, neww = int(self.inpHeight * hw_scale) + 1, self.inpWidth
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                padh = int((self.inpHeight - newh) * 0.5)
                img = cv2.copyMakeBorder(img, padh, self.inpHeight - newh - padh, 0, 0, cv2.BORDER_CONSTANT, value=0)
        else:
            img = cv2.resize(srcimg, (self.inpWidth, self.inpHeight), interpolation=cv2.INTER_AREA)
        return img, newh, neww, padh, padw

    def distance2bbox(self, points: np.ndarray, distance: np.ndarray, max_shape=None) -> np.ndarray:
        """
        将距离值转换为bbox
        :param points:
        :param distance:
        :param max_shape:
        :return:
        """
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = x1.clamp(min=0, max=max_shape[1])
            y1 = y1.clamp(min=0, max=max_shape[0])
            x2 = x2.clamp(min=0, max=max_shape[1])
            y2 = y2.clamp(min=0, max=max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)

    def distance2kps(self, points: np.ndarray, distance: np.ndarray, max_shape=None) -> np.ndarray:
        """
        将距离值转换为关键点坐标
        :param points:
        :param distance:
        :param max_shape:
        :return:
        """
        preds = []
        for i in range(0, distance.shape[1], 2):
            px = points[:, i % 2] + distance[:, i]
            py = points[:, i % 2 + 1] + distance[:, i + 1]
            if max_shape is not None:
                px = px.clamp(min=0, max=max_shape[1])
                py = py.clamp(min=0, max=max_shape[0])
            preds.append(px)
            preds.append(py)
        return np.stack(preds, axis=-1)

    def detect(self, srcimg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        检测图片中的人脸并返回检测结果
        流程：
            1. 缩放图片到指定大小
            2. 将图片转换为dnn输入格式
            3. 运行网络，得到输出
            4. 解析输出，得到检测结果
        :param srcimg: 输入的原始图片
        :return: 返回检测结果 (bboxes, kpss, scores)
        """
        img, newh, neww, padh, padw = self.resize_image(srcimg)
        blob = cv2.dnn.blobFromImage(img, 1.0 / 128, (self.inpWidth, self.inpHeight), (127.5, 127.5, 127.5),
                                     swapRB=True)
        self.net.setInput(blob)
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())

        scores_list, bboxes_list, kpss_list = [], [], []
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = outs[idx][0]
            bbox_preds = outs[idx + self.fmc * 1][0] * stride
            kps_preds = outs[idx + self.fmc * 2][0] * stride

            height = blob.shape[2] // stride
            width = blob.shape[3] // stride
            anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))

            if self._num_anchors > 1:
                anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))

            pos_inds = np.where(scores >= self.confThreshold)[0]
            bboxes = self.distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)

            kpss = self.distance2kps(anchor_centers, kps_preds)
            kpss = kpss.reshape((kpss.shape[0], -1, 2))
            pos_kpss = kpss[pos_inds]
            kpss_list.append(pos_kpss)

        scores = np.vstack(scores_list).ravel()
        bboxes = np.vstack(bboxes_list)
        kpss = np.vstack(kpss_list)

        # Adjust the bounding boxes and keypoints to the original image size
        bboxes[:, 2:4] = bboxes[:, 2:4] - bboxes[:, 0:2]
        ratioh, ratiow = srcimg.shape[0] / newh, srcimg.shape[1] / neww
        bboxes[:, 0] = (bboxes[:, 0] - padw) * ratiow
        bboxes[:, 1] = (bboxes[:, 1] - padh) * ratioh
        bboxes[:, 2] = bboxes[:, 2] * ratiow
        bboxes[:, 3] = bboxes[:, 3] * ratioh
        kpss[:, :, 0] = (kpss[:, :, 0] - padw) * ratiow
        kpss[:, :, 1] = (kpss[:, :, 1] - padh) * ratioh
        # 非极大值抑制
        indices = cv2.dnn.NMSBoxes(bboxes.tolist(), scores.tolist(), self.confThreshold, self.nmsThreshold)

        return bboxes[indices], kpss[indices], scores[indices]

    def draw(self, srcimg: np.ndarray, bboxes: np.ndarray, kpss: np.ndarray, scores: np.ndarray):
        """
        根据检测结果在原始图片上绘制人脸检测的框和关键点
        :param srcimg: 原始图片
        :param bboxes: 检测到的边界框
        :param kpss: 检测到的关键点
        :param scores: 检测的分数
        :return: 返回绘制了检测结果的图片
        """
        for i in range(bboxes.shape[0]):
            xmin, ymin, xamx, ymax = int(bboxes[i, 0]), int(bboxes[i, 1]), int(bboxes[i, 0] + bboxes[i, 2]), int(
                bboxes[i, 1] + bboxes[i, 3])
            # 绘制边界框
            cv2.rectangle(srcimg, (xmin, ymin), (xamx, ymax), (0, 0, 255), thickness=2)
            # 绘制关键点
            for j in range(5):
                cv2.circle(srcimg, (int(kpss[i, j, 0]), int(kpss[i, j, 1])), 1, (0, 255, 0), thickness=-1)
            # 绘制分数
            cv2.putText(srcimg, str(round(scores[i], 3)), (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                        thickness=1)
        return srcimg


if __name__ == '__main__':
    # example usage: python scrfd.py --input ../data/selfie.jpg --save True --output detection --output_dir ../result
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='../data/selfie.jpg', help='image or video path')
    parser.add_argument('--onnxmodel', default='../weights/scrfd_10g_kps.onnx', type=str,
                        choices=['../weights/scrfd_500m_kps.onnx', '../weights/scrfd_2.5g_kps.onnx',
                                 '../weights/scrfd_10g_kps.onnx'], help='onnx model')
    parser.add_argument('--confThreshold', default=0.5, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=0.4, type=float, help='nms iou thresh')
    parser.add_argument('--save', type=bool, default=True, help='Save detection results (image or video)')
    parser.add_argument('--output', type=str, default='detection', help='Output file name for saving results')
    parser.add_argument('--output_dir', type=str, default='../result',
                        help='Output directory for saving detection results')
    args = parser.parse_args()

    # 创建 result 目录（如果不存在）
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    mynet = SCRFD(args.onnxmodel, confThreshold=args.confThreshold, nmsThreshold=args.nmsThreshold)

    if args.input.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        cap = cv2.VideoCapture(args.input)  # 打开视频文件
        # 获取视频帧的宽度和高度
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if args.save:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 四字符编码
            out = cv2.VideoWriter(os.path.join(args.output_dir, args.output + '.mp4'), fourcc, 20.0, (width, height))

        while True:
            ret, srcimg = cap.read()  # 每次读取一帧
            if not ret:
                break  # 如果没有读取到帧，退出

            bboxes, kpss, scores = mynet.detect(srcimg)  # 检测人脸
            outimg = mynet.draw(srcimg, bboxes, kpss, scores)  # 绘制检测结果

            cv2.imshow('Deep learning object detection in OpenCV', outimg)  # 显示检测结果

            if args.save:
                out.write(outimg)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 键退出
                break

        cap.release()  # 释放视频捕获对象
        if args.save:
            out.release()  # 释放视频写入对象
    else:
        srcimg = cv2.imread(args.input)
        bboxes, kpss, scores = mynet.detect(srcimg)
        outimg = mynet.draw(srcimg, bboxes, kpss, scores)

        winName = 'Deep learning object detection in OpenCV'
        cv2.namedWindow(winName, 0)
        cv2.imshow(winName, outimg)

        if args.save:
            cv2.imwrite(os.path.join(args.output_dir, args.output + '.jpg'), outimg)

        cv2.waitKey(0)

    cv2.destroyAllWindows()  # 销毁所有窗口
