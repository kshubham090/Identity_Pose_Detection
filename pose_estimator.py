import numpy as np
import math
import config

try:
    import mediapipe as mp
    _ = mp.solutions.pose
    OLD_API = True
except AttributeError:
    OLD_API = False


def _angle(a, b, c):
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return math.degrees(math.acos(np.clip(cos, -1.0, 1.0)))


def _body_position(lm_list):
    """
    Classify body position: Standing / Sitting / Lying Down / Slouched / Leaning
    Uses relative Y positions of shoulders, hips, knees, ankles.
    Y increases downward in normalised coords.
    """
    ls, rs   = lm_list[11], lm_list[12]   # shoulders
    lh, rh   = lm_list[23], lm_list[24]   # hips
    lk, rk   = lm_list[25], lm_list[26]   # knees
    la, ra   = lm_list[27], lm_list[28]   # ankles
    le, re   = lm_list[7],  lm_list[8]    # ears
    nose     = lm_list[0]

    sh_mid_y  = (ls[1] + rs[1]) / 2
    sh_mid_x  = (ls[0] + rs[0]) / 2
    hip_mid_y = (lh[1] + rh[1]) / 2
    hip_mid_x = (lh[0] + rh[0]) / 2
    knee_mid_y= (lk[1] + rk[1]) / 2
    ankle_mid_y=(la[1] + ra[1]) / 2
    ear_mid_y = (le[1] + re[1]) / 2
    ear_mid_x = (le[0] + re[0]) / 2

    # Vertical span of body
    body_height = abs(ankle_mid_y - sh_mid_y) + 1e-6

    # How horizontal is the body? (lying = shoulders and hips at similar Y)
    spine_horizontal = abs(sh_mid_y - hip_mid_y) / body_height

    # Knee bend: are knees significantly above/below hips?
    knee_above_hip = knee_mid_y < hip_mid_y - 0.05   # sitting/crossed legs

    # Is the person mostly horizontal?
    if spine_horizontal < 0.15:
        position = "Lying Down"

    # Standing: hips high relative to knees and ankles
    elif hip_mid_y < knee_mid_y - 0.05 and body_height > 0.35:
        # check for slouch/lean
        spine_angle = _angle(
            (hip_mid_x, hip_mid_y),
            (sh_mid_x,  sh_mid_y),
            (ear_mid_x, ear_mid_y),
        )
        if spine_angle < 150:
            position = "Standing (Slouched)"
        else:
            # lateral lean
            lateral_offset = abs(sh_mid_x - hip_mid_x)
            if lateral_offset > 0.08:
                position = "Standing (Leaning)"
            else:
                position = "Standing"

    # Sitting: hips and knees at similar height
    elif abs(hip_mid_y - knee_mid_y) < 0.12 or knee_above_hip:
        spine_angle = _angle(
            (hip_mid_x, hip_mid_y),
            (sh_mid_x,  sh_mid_y),
            (ear_mid_x, ear_mid_y),
        )
        if spine_angle < 140:
            position = "Sitting (Slouched)"
        else:
            position = "Sitting"

    else:
        position = "Unknown"

    return position


def _posture_issues(lm_list):
    ls, rs = lm_list[11], lm_list[12]
    lh, rh = lm_list[23], lm_list[24]
    le, re = lm_list[7],  lm_list[8]
    nose   = lm_list[0]

    shoulder_tilt = abs(math.degrees(math.atan2(rs[1]-ls[1], rs[0]-ls[0])))
    sh_mid  = ((ls[0]+rs[0])/2, (ls[1]+rs[1])/2)
    hip_mid = ((lh[0]+rh[0])/2, (lh[1]+rh[1])/2)
    ear_mid = ((le[0]+re[0])/2, (le[1]+re[1])/2)
    spine_angle = _angle(hip_mid, sh_mid, ear_mid)
    fwd_head = abs(nose[0] - sh_mid[0])

    issues = []
    if shoulder_tilt > config.SHOULDER_TILT_THRESHOLD:
        issues.append("Uneven Shoulders")
    if spine_angle < config.SPINE_ANGLE_THRESHOLD:
        issues.append("Slouching")
    if fwd_head > config.FORWARD_HEAD_RATIO:
        issues.append("Forward Head")

    return issues, round(shoulder_tilt, 1), round(spine_angle, 1), round(fwd_head, 3)


def _metrics(lm_list):
    position          = _body_position(lm_list)
    issues, tilt, spine, fwd = _posture_issues(lm_list)

    score = max(0, 100 - len(issues) * 20)
    label = position
    if issues:
        label += " · " + " | ".join(issues)

    return {
        "body_position":  position,
        "posture_label":  label,
        "posture_score":  score,
        "posture_issues": issues,
        "shoulder_tilt":  tilt,
        "spine_angle":    spine,
        "fwd_head":       fwd,
    }


# ── Old API ───────────────────────────────────────────────────────────────────

class _OldPose:
    def __init__(self):
        mp_pose = mp.solutions.pose
        self._pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,          # 0 = fastest
            smooth_landmarks=True,
            min_detection_confidence=config.POSE_CONFIDENCE,
            min_tracking_confidence=config.POSE_CONFIDENCE,
        )

    def estimate(self, frame_rgb):
        res = self._pose.process(frame_rgb)
        if not res.pose_landmarks:
            return None
        lm_list = [(l.x, l.y, l.z) for l in res.pose_landmarks.landmark]
        result = _metrics(lm_list)
        result["landmarks"] = res.pose_landmarks
        return result

    def close(self):
        self._pose.close()


# ── New API ───────────────────────────────────────────────────────────────────

class _NewPose:
    def __init__(self):
        import urllib.request, os, tempfile
        model_path = os.path.join(tempfile.gettempdir(), "pose_landmarker_lite.task")
        if not os.path.exists(model_path):
            print("[PoseEstimator] Downloading pose landmarker model (~30MB)...")
            url = ("https://storage.googleapis.com/mediapipe-models/"
                   "pose_landmarker/pose_landmarker_lite/float16/latest/"
                   "pose_landmarker_lite.task")   # lite = fastest
            urllib.request.urlretrieve(url, model_path)
            print("[PoseEstimator] Done.")

        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision

        opts = mp_vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=config.POSE_CONFIDENCE,
            min_tracking_confidence=config.POSE_CONFIDENCE,
        )
        self._lm = mp_vision.PoseLandmarker.create_from_options(opts)

    def estimate(self, frame_rgb):
        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        res = self._lm.detect(mp_image)
        if not res.pose_landmarks:
            return None
        lm_list = [(l.x, l.y, l.z) for l in res.pose_landmarks[0]]
        result = _metrics(lm_list)
        result["landmarks"] = None
        result["raw_landmarks"] = res.pose_landmarks[0]
        return result

    def close(self):
        self._lm.close()


# ── Public ────────────────────────────────────────────────────────────────────

class PoseEstimator:
    def __init__(self):
        if OLD_API:
            self._impl = _OldPose()
            print("[PoseEstimator] old API")
        else:
            self._impl = _NewPose()
            print("[PoseEstimator] new API")

    def estimate(self, frame_rgb: np.ndarray) -> dict | None:
        if frame_rgb is None or frame_rgb.size == 0:
            return None
        try:
            return self._impl.estimate(frame_rgb)
        except Exception as e:
            print(f"[PoseEstimator] {e}")
            return None

    def close(self):
        self._impl.close()