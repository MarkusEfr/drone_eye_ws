#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import logging

from drone_eye_msgs.msg import BoundingBoxes, TrackingArray
from drone_eye_tracker.tracker import Tracker  # DeepSORT wrapper

_LOG = logging.getLogger("tracker_node")
logging.basicConfig(level=logging.INFO)


class TrackerNode(Node):
    def __init__(self):
        super().__init__("tracker_node")
        self.bridge = CvBridge()

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        # Tracker instance
        self.tracker = Tracker(
            max_age=30,
            n_init=3,
            max_cosine_distance=0.4,
        )

        # Subscriptions
        self.create_subscription(Image, "/camera/image_raw", self.image_callback, qos)
        self.create_subscription(
            BoundingBoxes, "/drone_eye/detections", self.detections_callback, qos
        )

        # Publisher for tracks
        self.tracks_pub = self.create_publisher(TrackingArray, "/drone_eye/tracks", qos)

        self.last_frame = None
        _LOG.info("TrackerNode started")

    def image_callback(self, msg: Image):
        """Store latest frame for tracker use"""
        try:
            self.last_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            _LOG.error(f"Error converting frame: {e}")

    def detections_callback(self, msg: BoundingBoxes):
        """Run tracking given new detections"""
        if self.last_frame is None:
            return

        # Pass raw BoundingBoxes to Tracker (it handles pixel conversion inside)
        tracks = self.tracker.update(msg.boxes, self.last_frame)

        # Wrap into TrackingArray
        tracks_msg = TrackingArray()
        tracks_msg.header = msg.header
        tracks_msg.tracks.extend(tracks)

        self.tracks_pub.publish(tracks_msg)
        _LOG.info(f"Published {len(tracks)} tracks")


def main():
    rclpy.init()
    node = TrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
