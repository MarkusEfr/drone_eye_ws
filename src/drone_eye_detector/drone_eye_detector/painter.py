import cv2


class Painter:
    def __init__(self):
        self.label_colors = {
            "car": (0, 255, 180),  # neon aqua
            "truck": (255, 200, 0),  # cyber orange
            "person": (0, 255, 0),  # drone green
            "default": (255, 255, 255),  # fallback white
        }

    def draw_bbox(self, frame, bbox, label=None, track_id=None, confidence=None):
        x1, y1, x2, y2 = map(int, bbox)
        color = self.label_colors.get(str(label), self.label_colors["default"])

        # Draw tech-style corner box
        self._draw_corner_box(frame, (x1, y1, x2, y2), color)

        # Compose futuristic label text
        text = ""
        if label is not None:
            text += f"[{label.upper()}]"
        if track_id is not None:
            text += f" ID:{track_id}"
        if confidence is not None:
            text += f" CONF:{confidence:.2f}"

        if text:
            cv2.putText(
                frame,
                text,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_PLAIN,
                1.3,
                (200, 255, 200),
                2,
            )

    def draw_fps(self, frame, fps):
        text = f"SYSTEM FPS: {fps:.2f}"
        cv2.putText(
            frame,
            text,
            (10, 25),
            cv2.FONT_HERSHEY_PLAIN,
            1.2,
            (0, 255, 100),
            2,
        )

    def draw_trail(self, frame, trail_points, color=(0, 128, 255)):
        for point in trail_points:
            cv2.circle(frame, point, 2, color, -1)

    def _draw_corner_box(self, frame, bbox, color, thickness=2, length=10):
        x1, y1, x2, y2 = bbox

        # Top-left corner
        cv2.line(frame, (x1, y1), (x1 + length, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + length), color, thickness)

        # Top-right corner
        cv2.line(frame, (x2, y1), (x2 - length, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + length), color, thickness)

        # Bottom-left corner
        cv2.line(frame, (x1, y2), (x1 + length, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - length), color, thickness)

        # Bottom-right corner
        cv2.line(frame, (x2, y2), (x2 - length, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - length), color, thickness)
