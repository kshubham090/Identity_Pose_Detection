import numpy as np
import mediapipe as mp
import math
import config

mp_pose = mp.solutions.pose
LM = mp_pose.PoseLandmark


def _angle(a, b, c):
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return math.degrees(math.acos(np.clip(cos, -1.0, 1.0)))


class PoseEstimator:
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=config.POSE_CONFIDENCE,
            min_tracking_confidence=config.POSE_CONFIDENCE,
        )

    def estimate(self, frame_rgb: np.ndarray) -> dict | None:
        res = self.pose.process(frame_rgb)
        if not res.pose_landmarks:
            return None

        lm = res.pose_landmarks.landmark

        ls = lm[LM.LEFT_SHOULDER]
        rs = lm[LM.RIGHT_SHOULDER]
        lh = lm[LM.LEFT_HIP]
        rh = lm[LM.RIGHT_HIP]
        le = lm[LM.LEFT_EAR]
        re = lm[LM.RIGHT_EAR]
        nose = lm[LM.NOSE]

        shoulder_tilt = abs(math.degrees(math.atan2(rs.y - ls.y, rs.x - ls.x)))

        hip_mid  = ((lh.x + rh.x) / 2, (lh.y + rh.y) / 2)
        sh_mid   = ((ls.x + rs.x) / 2, (ls.y + rs.y) / 2)
        ear_mid  = ((le.x + re.x) / 2, (le.y + re.y) / 2)
        spine_angle = _angle(hip_mid, sh_mid, ear_mid)
        fwd_head = abs(nose.x - sh_mid[0])

        issues = []
        if shoulder_tilt > config.SHOULDER_TILT_THRESHOLD:
            issues.append("Uneven Shoulders")
        if spine_angle < config.SPINE_ANGLE_THRESHOLD:
            issues.append("Slouching")
        if fwd_head > config.FORWARD_HEAD_RATIO:
            issues.append("Forward Head")

        return {
            "landmarks": res.pose_landmarks,
            "posture_label": "Good Posture" if not issues else " | ".join(issues),
            "posture_score": max(0, 100 - len(issues) * 25),
            "posture_issues": issues,
            "shoulder_tilt": round(shoulder_tilt, 1),
            "spine_angle": round(spine_angle, 1),
            "fwd_head": round(fwd_head, 3),
        }

    def close(self):
        self.pose.close()
