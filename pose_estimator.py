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
    ba = np.array([a[0]-b[0], a[1]-b[1]])
    bc = np.array([c[0]-b[0], c[1]-b[1]])
    cos = np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
    return math.degrees(math.acos(np.clip(cos,-1.0,1.0)))


def _vis(lm, idx, threshold=0.4):
    """Check if landmark is visible enough."""
    try:
        return lm[idx].visibility > threshold
    except Exception:
        return True  # new API doesn't have visibility per-landmark


def _body_position(lm_list, vis_fn=None):
    ls,rs   = lm_list[11], lm_list[12]
    lh,rh   = lm_list[23], lm_list[24]
    lk,rk   = lm_list[25], lm_list[26]
    la,ra   = lm_list[27], lm_list[28]
    le,re   = lm_list[7],  lm_list[8]
    nose    = lm_list[0]

    sh_y    = (ls[1]+rs[1])/2
    sh_x    = (ls[0]+rs[0])/2
    hip_y   = (lh[1]+rh[1])/2
    hip_x   = (lh[0]+rh[0])/2
    knee_y  = (lk[1]+rk[1])/2
    ankle_y = (la[1]+ra[1])/2
    ear_y   = (le[1]+re[1])/2
    ear_x   = (le[0]+re[0])/2

    body_h  = abs(ankle_y - sh_y) + 1e-6
    spine_h = abs(sh_y - hip_y)

    # Lying down: shoulders and hips at nearly same Y
    if spine_h / body_h < 0.18:
        return "Lying Down"

    # Standing vs sitting: hips above knees = standing
    knee_hip_diff = knee_y - hip_y  # positive = knees below hips

    if knee_hip_diff > 0.08:
        # Standing
        spine_angle = _angle(
            (hip_x, hip_y), (sh_x, sh_y), (ear_x, ear_y)
        )
        lateral = abs(sh_x - hip_x)
        if spine_angle < 145:
            return "Standing (Slouched)"
        elif lateral > 0.1:
            return "Standing (Leaning)"
        else:
            return "Standing"
    else:
        # Sitting
        spine_angle = _angle(
            (hip_x, hip_y), (sh_x, sh_y), (ear_x, ear_y)
        )
        if spine_angle < 130:
            return "Sitting (Slouched)"
        return "Sitting"


def _metrics(lm_list):
    ls,rs = lm_list[11], lm_list[12]
    lh,rh = lm_list[23], lm_list[24]
    le,re = lm_list[7],  lm_list[8]
    nose  = lm_list[0]

    shoulder_tilt = abs(math.degrees(math.atan2(rs[1]-ls[1], rs[0]-ls[0])))
    sh_mid  = ((ls[0]+rs[0])/2, (ls[1]+rs[1])/2)
    hip_mid = ((lh[0]+rh[0])/2, (lh[1]+rh[1])/2)
    ear_mid = ((le[0]+re[0])/2, (le[1]+re[1])/2)
    spine_angle = _angle(hip_mid, sh_mid, ear_mid)
    fwd_head    = abs(nose[0] - sh_mid[0])

    position = _body_position(lm_list)

    issues = []
    if shoulder_tilt > config.SHOULDER_TILT_THRESHOLD:
        issues.append("Uneven Shoulders")
    if spine_angle < config.SPINE_ANGLE_THRESHOLD:
        issues.append("Slouching")
    if fwd_head > config.FORWARD_HEAD_RATIO:
        issues.append("Forward Head")

    score = max(0, 100 - len(issues)*20)
    label = position + (" · " + " | ".join(issues) if issues else "")

    return {
        "body_position":  position,
        "posture_label":  label,
        "posture_score":  score,
        "posture_issues": issues,
        "shoulder_tilt":  round(shoulder_tilt,1),
        "spine_angle":    round(spine_angle,1),
        "fwd_head":       round(fwd_head,3),
    }


class _OldPose:
    def __init__(self):
        mp_pose = mp.solutions.pose
        self._pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=True,
            min_detection_confidence=config.POSE_CONFIDENCE,
            min_tracking_confidence=config.POSE_CONFIDENCE,
        )

    def estimate(self, frame_rgb):
        res = self._pose.process(frame_rgb)
        if not res.pose_landmarks:
            return None
        lm_list = [(l.x,l.y,l.z) for l in res.pose_landmarks.landmark]
        result  = _metrics(lm_list)
        result["landmarks"] = res.pose_landmarks
        return result

    def close(self):
        self._pose.close()


class _NewPose:
    def __init__(self):
        import urllib.request, os, tempfile
        model_path = os.path.join(tempfile.gettempdir(), "pose_landmarker_lite.task")
        if not os.path.exists(model_path):
            print("[PoseEstimator] Downloading model...")
            url = ("https://storage.googleapis.com/mediapipe-models/"
                   "pose_landmarker/pose_landmarker_lite/float16/latest/"
                   "pose_landmarker_lite.task")
            urllib.request.urlretrieve(url, model_path)
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
        lm_list = [(l.x,l.y,l.z) for l in res.pose_landmarks[0]]
        result  = _metrics(lm_list)
        result["landmarks"]     = None
        result["raw_landmarks"] = res.pose_landmarks[0]
        return result

    def close(self):
        self._lm.close()


class PoseEstimator:
    def __init__(self):
        if OLD_API:
            self._impl = _OldPose()
            print("[PoseEstimator] old API")
        else:
            self._impl = _NewPose()
            print("[PoseEstimator] new API")

    def estimate(self, frame_rgb):
        if frame_rgb is None or frame_rgb.size == 0:
            return None
        try:
            return self._impl.estimate(frame_rgb)
        except Exception as e:
            return None

    def close(self):
        self._impl.close()