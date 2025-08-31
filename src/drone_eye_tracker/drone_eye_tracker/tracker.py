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
        detections: list of drone_eye_msgs/BoundingBox
        frame: np.ndarray (image), needed for DeepSORT embedding
        """
        if frame is None:
            return []

        h, w, _ = frame.shape

        # --- Convert ROS detections -> DeepSORT format ---
        dets = []
        for det in detections:
            x1 = int(det.xmin * w)
            y1 = int(det.ymin * h)
            x2 = int(det.xmax * w)
            y2 = int(det.ymax * h)
            dets.append(([x1, y1, x2, y2], float(det.probability), det.label))

        if not dets:
            return []

        # --- Run DeepSORT ---
        outputs = self.tracker.update_tracks(raw_detections=dets, frame=frame)

        # --- Convert DeepSORT tracks -> TrackingInfo msgs ---
        tracks = []
        for track in outputs:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue

            x1, y1, x2, y2 = track.to_ltrb()
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            t = TrackingInfo()
            t.track_id = int(track.track_id)
            t.label = str(track.get_det_class())
            t.probability = float(track.get_det_conf())

            # Bounding box
            t.xmin = float(x1)
            t.ymin = float(y1)
            t.xmax = float(x2)
            t.ymax = float(y2)

            # Center point
            t.x = float(cx)
            t.y = float(cy)
            t.z = 0.0

            # Velocity (placeholder)
            t.velocity_x = 0.0
            t.velocity_y = 0.0
            t.velocity_z = 0.0

            tracks.append(t)

        return tracks
