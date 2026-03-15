# face_manager.py — Persistent face recognition using face_recognition library
# Saves encodings to disk — remembers faces across ALL runs forever.

import os
import pickle
import numpy as np
import config

try:
    import face_recognition
    HAS_FACE_REC = True
except ImportError:
    HAS_FACE_REC = False
    print("[FaceManager] face_recognition not installed. Run: pip install face-recognition")


class FaceManager:
    def __init__(self):
        self._db: dict[str, list] = {}   # name -> list of encodings
        self._load()

    # ── Persistence ───────────────────────────────────────────────────

    def _load(self):
        if os.path.exists(config.FACE_DB_PATH):
            try:
                with open(config.FACE_DB_PATH, "rb") as f:
                    self._db = pickle.load(f)
                print(f"[FaceManager] Loaded {len(self._db)} known identities: {list(self._db.keys())}")
            except Exception as e:
                print(f"[FaceManager] Could not load DB: {e}")
                self._db = {}
        else:
            print("[FaceManager] No face DB found, starting fresh.")

    def _save(self):
        try:
            with open(config.FACE_DB_PATH, "wb") as f:
                pickle.dump(self._db, f)
        except Exception as e:
            print(f"[FaceManager] Save error: {e}")

    # ── Core API ──────────────────────────────────────────────────────

    def identify(self, frame_rgb: np.ndarray, bbox: list) -> str | None:
        """
        Try to identify the person in bbox.
        Returns known name, or None if unknown / face_recognition not available.
        """
        if not HAS_FACE_REC or not self._db:
            return None

        x1, y1, x2, y2 = bbox
        h, w = frame_rgb.shape[:2]
        # face_recognition uses (top, right, bottom, left)
        face_loc = [(max(0,y1), min(w,x2), min(h,y2), max(0,x1))]

        try:
            encs = face_recognition.face_encodings(frame_rgb, known_face_locations=face_loc)
        except Exception:
            return None

        if not encs:
            return None

        enc = encs[0]
        best_name  = None
        best_dist  = config.FACE_MATCH_TOLERANCE

        for name, stored_encs in self._db.items():
            dists = face_recognition.face_distance(stored_encs, enc)
            if len(dists) and dists.min() < best_dist:
                best_dist  = dists.min()
                best_name  = name

        return best_name

    def register(self, frame_rgb: np.ndarray, bbox: list, name: str) -> bool:
        """
        Register the face in bbox under the given name.
        Saves to disk immediately.
        """
        if not HAS_FACE_REC:
            return False

        x1, y1, x2, y2 = bbox
        h, w = frame_rgb.shape[:2]
        face_loc = [(max(0,y1), min(w,x2), min(h,y2), max(0,x1))]

        try:
            encs = face_recognition.face_encodings(frame_rgb, known_face_locations=face_loc)
        except Exception as e:
            print(f"[FaceManager] Encoding error: {e}")
            return False

        if not encs:
            print(f"[FaceManager] No face found in crop for '{name}'")
            return False

        if name not in self._db:
            self._db[name] = []
        # Store up to 5 encodings per person for robustness
        if len(self._db[name]) < 5:
            self._db[name].append(encs[0])
        else:
            self._db[name][np.random.randint(5)] = encs[0]  # replace random slot

        self._save()
        print(f"[FaceManager] Registered '{name}' ({len(self._db[name])} encodings saved)")
        return True

    def delete(self, name: str):
        if name in self._db:
            del self._db[name]
            self._save()

    def known_names(self) -> list[str]:
        return list(self._db.keys())

    @property
    def available(self):
        return HAS_FACE_REC