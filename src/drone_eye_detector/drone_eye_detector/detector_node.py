import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from drone_eye_msgs.msg import BoundingBoxes, BoundingBox
from .detector import Detector


class DetectorNode(Node):
    def __init__(self):
        super().__init__("drone_eye_detector")

        self.bridge = CvBridge()
        model_path = "/home/ros2/ws/models/yolov8n.pt"
        self.detector = Detector(model_path)

        self.subscription = self.create_subscription(
            Image, "/camera/image_raw", self.image_callback, 10
        )
        self.publisher = self.create_publisher(
            BoundingBoxes, "/drone_eye/detections", 10
        )

        self.get_logger().info(f"DetectorNode started with model: {model_path}")

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        detections = self.detector.detect(frame)

        bb_msg = BoundingBoxes()
        bb_msg.header = msg.header
        for det in detections:
            box_msg = BoundingBox()
            box_msg.id = 0
            box_msg.label = det["label"]
            box_msg.probability = det["confidence"]
            x, y, w, h = det["bbox"]
            box_msg.xmin = x
            box_msg.ymin = y
            box_msg.xmax = x + w
            box_msg.ymax = y + h
            bb_msg.boxes.append(box_msg)

        self.publisher.publish(bb_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
