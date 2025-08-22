# drone_eye_detector/detector.py
import torch
from ultralytics import YOLO


class Detector:
    def __init__(
        self,
        model_path: str = "/home/ros2/ws/models/yolov8n.pt",
        confidence_threshold: float = 0.5,
        iou: float = 0.45,
        max_det: int = 300,
        allowed_labels=None,
        input_size: int = 416,
        device: str = "auto",
        half: bool = True,
        agnostic_nms: bool = False,
    ):
        if device in (None, "auto"):
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = YOLO(model_path)
        try:
            self.model.fuse()
        except Exception:
            pass

        self.confidence_threshold = float(confidence_threshold)
        self.iou = float(iou)
        self.max_det = int(max_det)
        self.agnostic_nms = bool(agnostic_nms)
        self.allowed_labels = set(allowed_labels or ["car", "truck", "bus", "person"])
        self.input_size = int(input_size)
        self.half = bool(half and torch.cuda.is_available())

        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True

    def _process_results(self, results):
        detections = []
        names = getattr(results, "names", None) or getattr(self.model, "names", {})

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < self.confidence_threshold:
                continue

            class_id = int(box.cls[0])
            label = names.get(class_id, str(class_id))
            if label not in self.allowed_labels:
                continue

            x1, y1, x2, y2 = map(float, box.xyxy[0].cpu().numpy())
            detections.append(
                {
                    "label": label,
                    "confidence": round(conf, 2),
                    "bbox": [x1, y1, x2, y2],
                }  # store absolute xyxy
            )
        return detections

    def _predict_one(self, frame):
        return self.model.predict(
            frame,
            imgsz=self.input_size,
            conf=self.confidence_threshold,
            iou=self.iou,
            max_det=self.max_det,
            agnostic_nms=self.agnostic_nms,
            device=self.device,
            half=self.half,
            verbose=False,
        )[0]

    def detect(self, frame):
        results = self._predict_one(frame)
        return self._process_results(results)
