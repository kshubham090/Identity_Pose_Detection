# object_detector.py — Detects all non-person objects using YOLOv8

import numpy as np
from ultralytics import YOLO
import config

# COCO class names (subset we care about showing)
COCO_NAMES = {
    1:"bicycle", 2:"car", 3:"motorcycle", 4:"airplane", 5:"bus",
    6:"train", 7:"truck", 8:"boat", 9:"traffic light", 10:"fire hydrant",
    11:"stop sign", 13:"bench", 14:"bird", 15:"cat", 16:"dog",
    17:"horse", 18:"sheep", 19:"cow", 20:"elephant", 21:"bear",
    22:"zebra", 23:"giraffe", 24:"backpack", 25:"umbrella", 26:"handbag",
    27:"tie", 28:"suitcase", 29:"frisbee", 30:"skis", 31:"snowboard",
    32:"sports ball", 33:"kite", 39:"bottle", 40:"wine glass", 41:"cup",
    42:"fork", 43:"knife", 44:"spoon", 45:"bowl", 46:"banana",
    47:"apple", 48:"sandwich", 49:"orange", 56:"chair", 57:"couch",
    58:"potted plant", 59:"bed", 60:"dining table", 62:"tv",
    63:"laptop", 64:"mouse", 65:"remote", 66:"keyboard", 67:"cell phone",
    68:"microwave", 69:"oven", 70:"toaster", 71:"sink", 72:"refrigerator",
    73:"book", 74:"clock", 75:"vase", 76:"scissors", 77:"teddy bear",
    78:"hair drier", 79:"toothbrush",
}

# Color per object category (BGR)
OBJ_COLOR = (180, 180, 50)


class ObjectDetector:
    def __init__(self):
        self.model = YOLO(config.YOLO_MODEL)

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Returns list of {bbox, label, conf} for non-person objects."""
        if not config.SHOW_OBJECTS:
            return []

        results = self.model(
            frame,
            conf=config.OBJECT_CONFIDENCE,
            verbose=False,
        )[0]

        objects = []
        for box in results.boxes:
            cls = int(box.cls[0])
            if cls == config.PERSON_CLASS_ID:
                continue  # persons handled by PersonDetector
            label = COCO_NAMES.get(cls, f"obj_{cls}")
            if label in config.OBJECT_LABELS_SKIP:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            objects.append({
                "bbox":  [x1, y1, x2, y2],
                "label": label,
                "conf":  round(float(box.conf[0]), 2),
            })
        return objects