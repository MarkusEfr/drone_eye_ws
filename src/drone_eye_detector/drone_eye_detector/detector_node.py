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

        # Subscribe to raw camera feed
        self.subscription = self.create_subscription(
            Image, "/camera/image_raw", self.image_callback, 10
        )

        # Publisher for detection metadata
        self.publisher = self.create_publisher(
            BoundingBoxes, "/drone_eye/detections", 10
        )

        # Publisher for annotated video stream
        self.result_publisher = self.create_publisher(
            Image, "/drone_eye/video_results", 10
        )

        self.get_logger().info(f"DetectorNode started with model: {model_path}")

    def image_callback(self, msg):
        # Convert ROS Image → OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # Detect objects (metadata)
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

        # Draw detections on frame
        annotated_frame = self.detector.detect_and_draw(frame.copy())

        # Publish annotated frame
        annotated_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
        self.result_publisher.publish(annotated_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
