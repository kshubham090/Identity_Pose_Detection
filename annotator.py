import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

PALETTE = [
    (0, 230, 118), (0, 176, 255), (255, 171, 0), (213, 0, 249),
    (255, 61, 0),  (0, 229, 255), (255, 214, 0), (100, 255, 218),
]


def _color(uid):
    return PALETTE[(uid - 1) % len(PALETTE)]


def draw_frame(frame, active_identities, pose_results, show_skeleton=True):
    out = frame.copy()
    h, w = out.shape[:2]

    for identity, pose in zip(active_identities, pose_results):
        color = _color(identity.uid)
        x1, y1, x2, y2 = identity.bbox

        # bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # corner accents
        L = 16
        for cx, cy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(out, (cx, cy), (cx+dx*L, cy), color, 3)
            cv2.line(out, (cx, cy), (cx, cy+dy*L), color, 3)

        # label banner
        label = identity.display_name
        score = pose["posture_score"] if pose else None
        posture = pose["posture_label"] if pose else "No Pose"
        sc = (0,230,118) if (score or 0) >= 75 else (255,171,0) if (score or 0) >= 50 else (0,80,255)

        by = max(0, y1 - 48)
        ov = out.copy()
        cv2.rectangle(ov, (x1, by), (x2, y1), (15, 18, 28), -1)
        cv2.addWeighted(ov, 0.75, out, 0.25, 0, out)
        cv2.putText(out, label, (x1+8, by+18), cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 1, cv2.LINE_AA)
        cv2.putText(out, posture, (x1+8, by+38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, sc, 1, cv2.LINE_AA)
        if score is not None:
            txt = f"{score}%"
            tw, _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0]
            cv2.putText(out, txt, (x2-tw-8, by+18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, sc, 1, cv2.LINE_AA)

        # skeleton on crop
        if show_skeleton and pose and pose.get("landmarks"):
            cy1, cy2, cx1, cx2 = max(0,y1), min(h,y2), max(0,x1), min(w,x2)
            if cy2 > cy1 and cx2 > cx1:
                crop = out[cy1:cy2, cx1:cx2].copy()
                mp_drawing.draw_landmarks(
                    crop, pose["landmarks"], mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(200,200,200), thickness=1),
                )
                out[cy1:cy2, cx1:cx2] = crop

    n = len(active_identities)
    cv2.putText(out, f"Tracking: {n} person{'s' if n != 1 else ''}", (12, 28),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
    return out
