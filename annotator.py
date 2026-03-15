import cv2
import numpy as np

try:
    import mediapipe as mp
    mp_drawing = mp.solutions.drawing_utils
    mp_pose    = mp.solutions.pose
    HAS_DRAWING = True
except AttributeError:
    HAS_DRAWING = False

PALETTE = [
    (0, 230, 118), (0, 176, 255), (255, 171, 0), (213, 0, 249),
    (255, 61, 0),  (0, 229, 255), (255, 214, 0), (100, 255, 218),
]

CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(24,26),(26,28),
    (0,11),(0,12),
]

# Position emoji map
POS_ICON = {
    "Standing":           "🧍",
    "Standing (Slouched)":"🧍",
    "Standing (Leaning)": "🧍",
    "Sitting":            "🪑",
    "Sitting (Slouched)": "🪑",
    "Lying Down":         "🛌",
    "Unknown":            "?",
}


def _color(uid):
    return PALETTE[(uid - 1) % len(PALETTE)]


def _draw_skeleton_manual(out, raw_lm, x1, y1, x2, y2, color):
    h_c, w_c = y2 - y1, x2 - x1
    pts = {i: (int(x1 + l.x * w_c), int(y1 + l.y * h_c)) for i, l in enumerate(raw_lm)}
    for a, b in CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(out, pts[a], pts[b], (220, 220, 220), 2, cv2.LINE_AA)
    for pt in pts.values():
        cv2.circle(out, pt, 5, color, -1, cv2.LINE_AA)


def draw_frame(frame, active_identities, pose_results, show_skeleton=True):
    out = frame.copy()
    h, w = out.shape[:2]

    for identity, pose in zip(active_identities, pose_results):
        color = _color(identity.uid)
        x1, y1, x2, y2 = identity.bbox

        # ── Bounding box ──────────────────────────────────────────────
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)

        # Corner accents — thicker, longer
        L, T = 24, 4
        for cx, cy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(out, (cx, cy), (cx+dx*L, cy), color, T)
            cv2.line(out, (cx, cy), (cx, cy+dy*L), color, T)

        # ── Labels ────────────────────────────────────────────────────
        label    = identity.display_name
        score    = pose["posture_score"]   if pose else None
        position = pose.get("body_position", "") if pose else ""
        posture  = pose["posture_label"]   if pose else "No Pose"

        sc = (0,230,118) if (score or 0) >= 75 else (255,171,0) if (score or 0) >= 50 else (0,80,255)

        # Banner height — bigger
        BANNER = 68
        by = max(0, y1 - BANNER)

        ov = out.copy()
        cv2.rectangle(ov, (x1, by), (x2, y1), (10, 14, 24), -1)
        cv2.addWeighted(ov, 0.78, out, 0.22, 0, out)

        # Name — big
        cv2.putText(out, label,
                    (x1+10, by+26),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2, cv2.LINE_AA)

        # Posture label — medium
        cv2.putText(out, posture,
                    (x1+10, by+54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, sc, 1, cv2.LINE_AA)

        # Score top-right of banner
        if score is not None:
            stxt = f"{score}%"
            tw   = cv2.getTextSize(stxt, cv2.FONT_HERSHEY_DUPLEX, 0.85, 2)[0][0]
            cv2.putText(out, stxt,
                        (x2 - tw - 10, by + 30),
                        cv2.FONT_HERSHEY_DUPLEX, 0.85, sc, 2, cv2.LINE_AA)

        # Position badge inside bbox top-left
        if position:
            badge_txt = position
            cv2.putText(out, badge_txt,
                        (x1+10, y1+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

        # ── Skeleton ──────────────────────────────────────────────────
        if show_skeleton and pose:
            cy1, cy2 = max(0, y1), min(h, y2)
            cx1, cx2 = max(0, x1), min(w, x2)
            if cy2 > cy1 and cx2 > cx1:
                if HAS_DRAWING and pose.get("landmarks"):
                    crop = out[cy1:cy2, cx1:cx2].copy()
                    mp_drawing.draw_landmarks(
                        crop, pose["landmarks"], mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=color, thickness=3, circle_radius=5),
                        mp_drawing.DrawingSpec(color=(220,220,220), thickness=2),
                    )
                    out[cy1:cy2, cx1:cx2] = crop
                elif pose.get("raw_landmarks"):
                    _draw_skeleton_manual(out, pose["raw_landmarks"], x1, y1, x2, y2, color)

    # ── HUD ───────────────────────────────────────────────────────────
    n = len(active_identities)
    hud = f"TRACKING: {n} PERSON{'S' if n != 1 else ''}"
    cv2.putText(out, hud, (14, 36),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)

    return out