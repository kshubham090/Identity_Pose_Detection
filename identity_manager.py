import time
import config


def _iou(a, b):
    ix1,iy1 = max(a[0],b[0]), max(a[1],b[1])
    ix2,iy2 = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    if inter == 0: return 0.0
    return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter+1e-6)


def _center(bbox):
    return ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)


def _dist(a, b):
    return ((a[0]-b[0])**2+(a[1]-b[1])**2)**0.5


class Identity:
    def __init__(self, uid, bbox):
        self.uid         = uid
        self.name        = None
        self.bbox        = bbox
        self.last_seen   = time.time()
        self.pose_data   = None
        self.emotion     = None
        self.gestures    = []
        self.frame_count = 0

    @property
    def display_name(self):
        return self.name if self.name else f"{config.UNKNOWN_LABEL_PREFIX} {self.uid}"

    def update(self, bbox, pose_data):
        self.bbox       = bbox
        self.last_seen  = time.time()
        self.frame_count += 1
        if pose_data:
            self.pose_data = pose_data

    def to_dict(self):
        p = self.pose_data or {}
        return {
            "uid":           self.uid,
            "label":         self.display_name,
            "bbox":          self.bbox,
            "posture_label": p.get("posture_label","—"),
            "body_position": p.get("body_position","—"),
            "posture_score": p.get("posture_score"),
            "posture_issues":p.get("posture_issues",[]),
            "shoulder_tilt": p.get("shoulder_tilt"),
            "spine_angle":   p.get("spine_angle"),
            "fwd_head":      p.get("fwd_head"),
            "emotion":       self.emotion,
            "gestures":      self.gestures,
            "frames_tracked":self.frame_count,
        }

    def get_all(self):
        return list(self._identities.values())


class IdentityManager:
    def __init__(self):
        self._identities: dict[int, Identity] = {}
        self._next_uid = 1

    def get_all(self):
        return list(self._identities.values())

    def update(self, detections, pose_results):
        self._expire()
        matched = set()
        active  = []

        for det, pose in zip(detections, pose_results):
            bbox     = det["bbox"]
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
        # Try IoU first, then fall back to centroid distance
        best_score = config.IDENTITY_IOU_THRESHOLD
        best_uid   = None
        cx, cy     = _center(bbox)
        bh         = bbox[3] - bbox[1]

        for uid, identity in self._identities.items():
            if uid in matched:
                continue
            iou = _iou(bbox, identity.bbox)
            if iou > best_score:
                best_score = iou
                best_uid   = uid

        # centroid fallback — helps when person turns head or moves fast
        if best_uid is None:
            best_d = bh * 1.2  # 120% of box height — generous
            for uid, identity in self._identities.items():
                if uid in matched:
                    continue
                d = _dist(_center(identity.bbox), (cx, cy))
                if d < best_d:
                    best_d   = d
                    best_uid = uid

        return self._identities[best_uid] if best_uid else None

    def _expire(self):
        now   = time.time()
        stale = [uid for uid, i in self._identities.items()
                 if now - i.last_seen > config.IDENTITY_TIMEOUT_SECONDS]
        for uid in stale:
            del self._identities[uid]