import cv2
import numpy as np


class Painter:
    def __init__(self):
        # Цвета по метке (tech-cyber palette)
        self.label_colors = {
            "car": (0, 255, 180),  # neon aqua
            "truck": (255, 200, 0),  # cyber orange
            "person": (0, 255, 0),  # drone green
            "default": (255, 255, 255),  # fallback white
        }

    def draw_bbox(
        self, frame, bbox, label=None, track_id=None, confidence=None, mode="track"
    ):
        """
        Draw tech-style bounding box with optional HUD info.
        mode="det"   → thin gray box
        mode="track" → neon box with ID and confidence
        """
        x1, y1, x2, y2 = map(int, bbox)

        # Цвет и толщина линии
        if mode == "det":
            color = (180, 180, 180)
            thickness = 1
        else:
            color = self.label_colors.get(str(label), self.label_colors["default"])
            thickness = 2

        # Рисуем угловой бокс
        self._draw_corner_box(frame, (x1, y1, x2, y2), color, thickness)

        # Формируем текст
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
                (200, 255, 200) if mode == "track" else (180, 180, 180),
                2,
            )

    def draw_fps(self, frame, fps):
        text = f"SYSTEM FPS: {fps:.2f}"
        cv2.putText(
            frame, text, (10, 25), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 255, 100), 2
        )

    def draw_trail(self, frame, trail_points, color=(0, 128, 255)):
        for point in trail_points:
            cv2.circle(frame, point, 2, color, -1)

    def draw_hud_grid(self, frame, step=80):
        """Лёгкая сетка HUD для ориентации"""
        h, w, _ = frame.shape
        color = (0, 80, 0)
        for x in range(0, w, step):
            cv2.line(frame, (x, 0), (x, h), color, 1, cv2.LINE_AA)
        for y in range(0, h, step):
            cv2.line(frame, (0, y), (w, y), color, 1, cv2.LINE_AA)

    def _draw_corner_box(self, frame, bbox, color, thickness=2, length=10):
        x1, y1, x2, y2 = bbox

        # Верхний левый
        cv2.line(frame, (x1, y1), (x1 + length, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + length), color, thickness)

        # Верхний правый
        cv2.line(frame, (x2, y1), (x2 - length, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + length), color, thickness)

        # Нижний левый
        cv2.line(frame, (x1, y2), (x1 + length, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - length), color, thickness)

        # Нижний правый
        cv2.line(frame, (x2, y2), (x2 - length, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - length), color, thickness)
