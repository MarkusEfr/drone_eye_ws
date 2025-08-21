import torch
from ultralytics import YOLO
from drone_eye_detector.painter import Painter


class Detector:
    def __init__(
        self,
        model_path: str = "/home/ros2/ws/models/yolov8n.pt",  # nano model for speed
        confidence_threshold: float = 0.5,
        iou: float = 0.45,  # NMS IoU
        max_det: int = 300,  # max detections per image
        allowed_labels=None,
        input_size: int = 416,  # 320/416 are good for realtime
        device: str = "auto",
        half: bool = True,  # will be used only on CUDA
        agnostic_nms: bool = False,
    ):
        # Device selection
        if device in (None, "auto"):
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Model
        self.model = YOLO(model_path)
        # .fuse() exists on most Ultralytics models, but guard just in case
        try:
            self.model.fuse()
        except Exception:
            pass

        # Settings
        self.confidence_threshold = float(confidence_threshold)
        self.iou = float(iou)
        self.max_det = int(max_det)
        self.agnostic_nms = bool(agnostic_nms)
        self.allowed_labels = set(allowed_labels or ["car", "truck", "bus", "person"])
        self.input_size = int(input_size)
        # Use half only if CUDA is available
        self.half = bool(half and torch.cuda.is_available())

        # Optional: let cuDNN pick best algos for constant input size
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True

        self.painter = Painter()

    def _process_results(self, results):
        """Filter and format YOLO results."""
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
            bbox = [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]
            detections.append(
                {"label": label, "confidence": round(conf, 2), "bbox": bbox}
            )
        return detections

    def _predict_one(self, frame):
        """Unified predict call with all knobs."""
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

    def detect_and_draw(self, frame):
        results = self._predict_one(frame)
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
