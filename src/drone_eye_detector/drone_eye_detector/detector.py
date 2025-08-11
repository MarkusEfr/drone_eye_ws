from ultralytics import YOLO
from drone_eye_detector.painter import (
    Painter,
)


class Detector:
    def __init__(
        self, model_path: str, confidence_threshold: float = 0.5, allowed_labels=None
    ):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.allowed_labels = set(allowed_labels or ["car", "truck", "bus", "person"])
        self.painter = Painter()

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            confidence = float(box.conf[0])
            if confidence < self.confidence_threshold:
                continue
            class_id = int(box.cls[0])
            label = self.model.names[class_id]
            if label not in self.allowed_labels:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            bbox = [round(float(x), 2) for x in [x1, y1, x2 - x1, y2 - y1]]
            detections.append(
                {"label": label, "confidence": round(confidence, 2), "bbox": bbox}
            )
        return detections

    def detect_and_draw(self, frame):
        results = self.model(frame, verbose=False)[0]
        for box in results.boxes:
            confidence = float(box.conf[0])
            if confidence < self.confidence_threshold:
                continue
            class_id = int(box.cls[0])
            label = self.model.names[class_id]
            if label not in self.allowed_labels:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            bbox = [x1, y1, x2, y2]
            self.painter.draw_bbox(frame, bbox=bbox, label=label, confidence=confidence)
        return frame
