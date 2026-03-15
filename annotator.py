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
OBJ_COLOR = (60, 210, 210)


def _color(uid):
    return PALETTE[(uid - 1) % len(PALETTE)]


def _draw_skeleton_manual(out, raw_lm, x1, y1, x2, y2, color):
    h_c, w_c = y2-y1, x2-x1
    pts = {i: (int(x1+l.x*w_c), int(y1+l.y*h_c)) for i,l in enumerate(raw_lm)}
    for a,b in CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(out, pts[a], pts[b], (220,220,220), 2, cv2.LINE_AA)
    for pt in pts.values():
        cv2.circle(out, pt, 5, color, -1, cv2.LINE_AA)


def _put_label_bg(out, text, x, y, font, scale, color, thickness=2):
    """Draw text with dark background pill."""
    (tw, th), bl = cv2.getTextSize(text, font, scale, thickness)
    pad = 5
    cv2.rectangle(out, (x-pad, y-th-pad), (x+tw+pad, y+bl+pad), (10,14,24), -1)
    cv2.putText(out, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_objects(out, objects: list[dict]):
    """Draw non-person object boxes."""
    for obj in objects:
        x1,y1,x2,y2 = obj["bbox"]
        label = f"{obj['label']} {obj['conf']:.0%}"
        cv2.rectangle(out, (x1,y1), (x2,y2), OBJ_COLOR, 2)
        # dashed top-left corner accent
        L = 14
        for cx,cy,dx,dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(out,(cx,cy),(cx+dx*L,cy),OBJ_COLOR,2)
            cv2.line(out,(cx,cy),(cx,cy+dy*L),OBJ_COLOR,2)
        _put_label_bg(out, label, x1+4, y1-6,
                      cv2.FONT_HERSHEY_SIMPLEX, 0.58, OBJ_COLOR, 1)


def draw_frame(frame, active_identities, pose_results, objects=None, show_skeleton=True):
    out = frame.copy()
    h, w = out.shape[:2]

    # Draw objects first (behind persons)
    if objects:
        draw_objects(out, objects)

    for identity, pose in zip(active_identities, pose_results):
        color = _color(identity.uid)
        x1,y1,x2,y2 = identity.bbox

        # Bbox
        cv2.rectangle(out, (x1,y1), (x2,y2), color, 3)
        L, T = 24, 4
        for cx,cy,dx,dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(out,(cx,cy),(cx+dx*L,cy),color,T)
            cv2.line(out,(cx,cy),(cx,cy+dy*L),color,T)

        # Labels
        label    = identity.display_name
        score    = pose["posture_score"]    if pose else None
        position = pose.get("body_position","") if pose else ""
        posture  = pose["posture_label"]    if pose else "No Pose"
        sc = (0,230,118) if (score or 0)>=75 else (255,171,0) if (score or 0)>=50 else (0,80,255)

        BANNER = 72
        by = max(0, y1-BANNER)
        ov = out.copy()
        cv2.rectangle(ov, (x1,by), (x2,y1), (10,14,24), -1)
        cv2.addWeighted(ov, 0.78, out, 0.22, 0, out)

        # Name — big bold
        cv2.putText(out, label,   (x1+10, by+28), cv2.FONT_HERSHEY_DUPLEX,  1.0, color, 2, cv2.LINE_AA)
        # Posture
        cv2.putText(out, posture, (x1+10, by+58), cv2.FONT_HERSHEY_SIMPLEX, 0.62, sc,   1, cv2.LINE_AA)
        # Score top-right
        if score is not None:
            stxt = f"{score}%"
            tw = cv2.getTextSize(stxt, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)[0][0]
            cv2.putText(out, stxt, (x2-tw-10, by+30), cv2.FONT_HERSHEY_DUPLEX, 0.9, sc, 2, cv2.LINE_AA)
        # Position inside box
        if position:
            cv2.putText(out, position, (x1+10, y1+32), cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2, cv2.LINE_AA)

        # Skeleton
        if show_skeleton and pose:
            cy1,cy2 = max(0,y1), min(h,y2)
            cx1,cx2 = max(0,x1), min(w,x2)
            if cy2>cy1 and cx2>cx1:
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

    # HUD
    n = len(active_identities)
    no = len(objects) if objects else 0
    hud = f"PEOPLE: {n}   OBJECTS: {no}"
    cv2.putText(out, hud, (14, 38), cv2.FONT_HERSHEY_DUPLEX, 1.1, (255,255,255), 2, cv2.LINE_AA)
    return out