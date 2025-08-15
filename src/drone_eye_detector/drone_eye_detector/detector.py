from ultralytics import YOLO
from drone_eye_detector.painter import Painter


class Detector:
    def __init__(
        self, model_path: str, confidence_threshold: float = 0.5, allowed_labels=None
    ):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.allowed_labels = set(allowed_labels or ["car", "truck", "bus", "person"])
        self.painter = Painter()

    def _process_results(self, results):
        """Extract valid detections from YOLO results."""
        detections = []
        for box in results.boxes:
            confidence = float(box.conf[0])
            if confidence < self.confidence_threshold:
                continue

            class_id = int(box.cls[0])
            label = self.model.names[class_id]
            if label not in self.allowed_labels:
                continue

            x1, y1, x2, y2 = map(float, box.xyxy[0].cpu().numpy())
            bbox = [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]

            detections.append(
                {"label": label, "confidence": round(confidence, 2), "bbox": bbox}
            )
        return detections

    def detect(self, frame):
        """Run object detection and return structured results."""
        results = self.model(frame, verbose=False)[0]
        return self._process_results(results)

    def detect_and_draw(self, frame):
        """Run detection, draw bounding boxes, and return the frame."""
        results = self.model(frame, verbose=False)[0]
        detections = self._process_results(results)

        for det in detections:
            x, y, w, h = det["bbox"]
            x2, y2 = x + w, y + h
            self.painter.draw_bbox(
                frame,
                bbox=[int(x), int(y), int(x2), int(y2)],
                label=det["label"],
                confidence=det["confidence"],
            )
        return frame
