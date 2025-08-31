#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Header

from drone_eye_msgs.msg import BoundingBoxes, TrackingArray
from drone_eye_visualizer.painter import Painter

import time


class VisualizerNode(Node):
    def __init__(self):
        super().__init__("visualizer_node")
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        self.bridge = CvBridge()
        self.painter = Painter()

        # Subscribers
        self.create_subscription(Image, "/camera/image_raw", self.image_callback, qos)
        self.create_subscription(
            BoundingBoxes, "/drone_eye/detections", self.dets_callback, qos
        )
        self.create_subscription(
            TrackingArray, "/drone_eye/tracks", self.tracks_callback, qos
        )

        # Publisher
        self.image_pub = self.create_publisher(Image, "/drone_eye/visualization", qos)

        # Cache latest data
        self.latest_frame = None
        self.latest_dets = []
        self.latest_tracks = []

        self.last_time = time.time()
        self.fps = 0.0

        self.get_logger().info("VisualizerNode started")

    def dets_callback(self, msg: BoundingBoxes):
        self.latest_dets = msg.boxes

    def tracks_callback(self, msg: TrackingArray):
        self.latest_tracks = msg.tracks

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_frame = frame.copy()

            h, w, _ = frame.shape

            # --- Build list of track bboxes to avoid duplicate drawing ---
            track_boxes = []
            for trk in self.latest_tracks:
                x1 = int(trk.xmin)
                y1 = int(trk.ymin)
                x2 = int(trk.xmax)
                y2 = int(trk.ymax)
                track_boxes.append((x1, y1, x2, y2))

                self.painter.draw_bbox(
                    frame,
                    (x1, y1, x2, y2),
                    label=trk.label,
                    track_id=trk.track_id,
                    confidence=trk.probability,
                    mode="track",
                )

            # --- Draw remaining detections not covered by tracks ---
            for det in self.latest_dets:
                x1 = int(det.xmin * w)
                y1 = int(det.ymin * h)
                x2 = int(det.xmax * w)
                y2 = int(det.ymax * h)

                # Check if detection overlaps any track bbox
                overlap = False
                for tb in track_boxes:
                    tx1, ty1, tx2, ty2 = tb
                    if not (x2 < tx1 or x1 > tx2 or y2 < ty1 or y1 > ty2):
                        overlap = True
                        break

                if not overlap:
                    self.painter.draw_bbox(
                        frame,
                        (x1, y1, x2, y2),
                        label=det.label,
                        confidence=det.probability,
                        mode="det",
                    )

            # --- FPS display ---
            now = time.time()
            dt = now - self.last_time
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
            self.last_time = now
            self.painter.draw_fps(frame, self.fps)

            # --- Publish annotated frame ---
            out_img = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            out_img.header = Header()
            out_img.header.stamp = msg.header.stamp
            out_img.header.frame_id = msg.header.frame_id
            self.image_pub.publish(out_img)

        except Exception as e:
            self.get_logger().error(f"Error in visualization: {e}")


def main():
    rclpy.init()
    node = VisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
