import time
import config


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-6)


class Identity:
    def __init__(self, uid, bbox):
        self.uid = uid
        self.name = None
        self.bbox = bbox
        self.last_seen = time.time()
        self.pose_data = None
        self.frame_count = 0

    @property
    def display_name(self):
        return self.name if self.name else f"{config.UNKNOWN_LABEL_PREFIX} {self.uid}"

    def update(self, bbox, pose_data):
        self.bbox = bbox
        self.last_seen = time.time()
        self.frame_count += 1
        if pose_data:
            self.pose_data = pose_data

    def to_dict(self):
        p = self.pose_data or {}
        return {
            "uid": self.uid,
            "label": self.display_name,
            "bbox": self.bbox,
            "posture_label": p.get("posture_label", "—"),
            "posture_score": p.get("posture_score"),
            "posture_issues": p.get("posture_issues", []),
            "shoulder_tilt": p.get("shoulder_tilt"),
            "spine_angle": p.get("spine_angle"),
            "fwd_head": p.get("fwd_head"),
            "frames_tracked": self.frame_count,
        }


class IdentityManager:
    def __init__(self):
        self._identities: dict[int, Identity] = {}
        self._next_uid = 1

    def update(self, detections, pose_results):
        self._expire()
        matched = set()
        active = []
        for det, pose in zip(detections, pose_results):
            bbox = det["bbox"]
            identity = self._match(bbox, matched)
            if identity is None:
                if len(self._identities) < config.MAX_IDENTITIES:
                    identity = Identity(self._next_uid, bbox)
                    self._identities[self._next_uid] = identity
                    self._next_uid += 1
                else:
                    continue
            identity.update(bbox, pose)
            matched.add(identity.uid)
            active.append(identity)
        return active

    def assign_name(self, uid, name):
        if uid in self._identities:
            self._identities[uid].name = name

    def get_summary(self):
        return [i.to_dict() for i in self._identities.values()]

    def _match(self, bbox, matched):
        best, best_uid = config.IDENTITY_IOU_THRESHOLD, None
        for uid, identity in self._identities.items():
            if uid in matched:
                continue
            iou = _iou(bbox, identity.bbox)
            if iou > best:
                best, best_uid = iou, uid
        return self._identities[best_uid] if best_uid else None

    def _expire(self):
        now = time.time()
        stale = [uid for uid, i in self._identities.items() if now - i.last_seen > config.IDENTITY_TIMEOUT_SECONDS]
        for uid in stale:
            del self._identities[uid]
