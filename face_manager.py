import os
import pickle
import numpy as np
import config

try:
    import face_recognition
    HAS_FACE_REC = True
except ImportError:
    HAS_FACE_REC = False
    print("[FaceManager] face_recognition not installed.")


class FaceManager:
    def __init__(self):
        self._db: dict[str, list] = {}
        self._load()

    def _load(self):
        if os.path.exists(config.FACE_DB_PATH):
            try:
                with open(config.FACE_DB_PATH, "rb") as f:
                    self._db = pickle.load(f)
                print(f"[FaceManager] Loaded {len(self._db)} faces: {list(self._db.keys())}")
            except Exception as e:
                print(f"[FaceManager] Load error: {e}")
                self._db = {}
        else:
            print("[FaceManager] No face DB yet.")

    def _save(self):
        try:
            with open(config.FACE_DB_PATH, "wb") as f:
                pickle.dump(self._db, f)
            print(f"[FaceManager] Saved DB: {list(self._db.keys())}")
        except Exception as e:
            print(f"[FaceManager] Save error: {e}")

    def identify(self, frame_rgb: np.ndarray, bbox: list) -> str | None:
        if not HAS_FACE_REC or not self._db:
            return None
        x1,y1,x2,y2 = [int(v) for v in bbox]
        H,W = frame_rgb.shape[:2]
        # face_recognition wants (top, right, bottom, left)
        loc = [(max(0,y1), min(W,x2), min(H,y2), max(0,x1))]
        try:
            encs = face_recognition.face_encodings(frame_rgb, known_face_locations=loc, num_jitters=1)
        except Exception as e:
            return None
        if not encs:
            return None
        enc = encs[0]
        best_name = None
        best_dist = config.FACE_MATCH_TOLERANCE
        for name, stored in self._db.items():
            if not stored:
                continue
            dists = face_recognition.face_distance(stored, enc)
            if len(dists) and dists.min() < best_dist:
                best_dist = dists.min()
                best_name = name
        return best_name

    def register(self, frame_rgb: np.ndarray, bbox: list, name: str) -> bool:
        if not HAS_FACE_REC:
            return False
        x1,y1,x2,y2 = [int(v) for v in bbox]
        H,W = frame_rgb.shape[:2]
        loc = [(max(0,y1), min(W,x2), min(H,y2), max(0,x1))]
        try:
            encs = face_recognition.face_encodings(frame_rgb, known_face_locations=loc, num_jitters=2)
        except Exception as e:
            print(f"[FaceManager] Encoding error: {e}")
            return False
        if not encs:
            print(f"[FaceManager] No face found for '{name}'")
            return False
        if name not in self._db:
            self._db[name] = []
        if len(self._db[name]) < 8:
            self._db[name].append(encs[0])
        else:
            idx = int(np.random.randint(8))
            self._db[name][idx] = encs[0]
        self._save()
        print(f"[FaceManager] '{name}' now has {len(self._db[name])} encodings")
        return True

    def delete(self, name: str):
        if name in self._db:
            del self._db[name]
            self._save()

    def known_names(self):
        return list(self._db.keys())

    @property
    def available(self):
        return HAS_FACE_REC