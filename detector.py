import numpy as np
from ultralytics import YOLO
import config


class PersonDetector:
    def __init__(self):
        self.model = YOLO(config.YOLO_MODEL)

    def detect(self, frame: np.ndarray) -> list[dict]:
        results = self.model(frame, conf=config.DETECTION_CONFIDENCE, classes=[config.PERSON_CLASS_ID], verbose=False)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append({"bbox": [x1, y1, x2, y2], "conf": float(box.conf[0])})
        return detections
