#!/usr/bin/env python3
import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from drone_eye_msgs.msg import BoundingBox, BoundingBoxes
from std_msgs.msg import Header
import logging

from drone_eye_detector.detector import Detector

_LOG = logging.getLogger("detector_node")
logging.basicConfig(level=logging.INFO)


class DetectorNode(Node):
    def __init__(self):
        super().__init__("detector_node")
        self.bridge = CvBridge()

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        model_path = "/home/ros2/ws/models/yolov8n.pt"  # nano
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.detector = Detector(
            model_path=model_path,
            device=device,
            half=True,  # will be ignored on CPU
            confidence_threshold=0.25,  # tune 0.25..0.5
            iou=0.45,
            input_size=416,  # 320/416 for speed
            allowed_labels=["car", "truck", "bus", "person"],
            max_det=200,
            agnostic_nms=False,
        )

        self.create_subscription(Image, "/camera/image_raw", self.image_callback, qos)
        self.boxes_pub = self.create_publisher(
            BoundingBoxes, "/drone_eye/detections", qos
        )
        self.image_pub = self.create_publisher(Image, "/drone_eye/video_results", qos)

        _LOG.info(f"DetectorNode started with model: {model_path} on {device}")

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            detections = self.detector.detect(frame)
            annotated = self.detector.detect_and_draw(frame.copy())

            out_img = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            out_img.header = Header()
            out_img.header.stamp = msg.header.stamp
            out_img.header.frame_id = msg.header.frame_id
            self.image_pub.publish(out_img)

            boxes_msg = BoundingBoxes()
            boxes_msg.header = Header()
            boxes_msg.header.stamp = msg.header.stamp
            boxes_msg.header.frame_id = msg.header.frame_id

            h, w, _ = annotated.shape
            for i, det in enumerate(detections):
                box = BoundingBox()
                box.id = i
                box.label = det["label"]
                box.probability = det["confidence"]
                x, y, bw, bh = det["bbox"]
                x1, y1, x2, y2 = x, y, x + bw, y + bh
                box.xmin = x1 / w
                box.ymin = y1 / h
                box.xmax = x2 / w
                box.ymax = y2 / h
                boxes_msg.boxes.append(box)

            self.boxes_pub.publish(boxes_msg)
            _LOG.info(f"Published {len(boxes_msg.boxes)} detections")

        except Exception as e:
            _LOG.error(f"Error processing frame: {e}")


def main():
    rclpy.init()
    node = DetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
