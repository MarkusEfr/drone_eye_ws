# drone_eye_tracker/tracker.py
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from drone_eye_msgs.msg import TrackingInfo


class Tracker:
    def __init__(self, max_age=30, n_init=3, nn_budget=100, max_cosine_distance=0.2):
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            nn_budget=nn_budget,
            max_cosine_distance=max_cosine_distance,
        )

    def update(self, detections, frame=None):
        """
        detections: list of dicts {label, confidence, bbox[x,y,w,h]}
        returns: list of TrackingInfo messages
        """
        xywhs, confs, clss = [], [], []

        for det in detections:
            x, y, w, h = det["bbox"]
            xywhs.append([x, y, w, h])
            confs.append(det["confidence"])
            clss.append(det["label"])

        outputs = self.tracker.update_tracks(
            np.array(xywhs), np.array(confs), np.array(clss), frame
        )

        tracks = []
        for track in outputs:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue

            x1, y1, x2, y2 = track.to_ltrb()

            t = TrackingInfo()
            t.track_id = track.track_id
            t.label = track.get_det_class()
            t.probability = float(track.get_det_conf())

            # Map bbox to center point
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            t.x = float(cx)
            t.y = float(cy)
            t.z = 0.0  # no depth yet

            # Velocity (not exposed directly by DeepSort; set to 0.0 for now)
            t.velocity_x = 0.0
            t.velocity_y = 0.0
            t.velocity_z = 0.0

            tracks.append(t)

        return tracks
