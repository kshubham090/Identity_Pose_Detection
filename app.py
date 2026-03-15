"""
Real-time Identity & Posture System
- OpenCV window: live feed, hand drawing (pinch to draw), skeleton, objects
- Gradio at :7860 — big font panel with identity, pose, emotion, gestures
- Press Q to quit, C to clear drawing
"""

import cv2
import numpy as np
import threading
import time
import math
import gradio as gr
import mediapipe as mp

from detector          import PersonDetector
from pose_estimator    import PoseEstimator
from identity_manager  import IdentityManager
from face_manager      import FaceManager
from object_detector   import ObjectDetector
from annotator         import draw_frame
from qa_engine         import QAEngine, _empty
import config

# ── Models ────────────────────────────────────────────────────────────────────
detector     = PersonDetector()
obj_detector = ObjectDetector()
pose_est     = PoseEstimator()
id_manager   = IdentityManager()
face_mgr     = FaceManager()
qa_engine    = QAEngine()

# MediaPipe hands + face mesh for emotion
# mediapipe new API — use tasks
from mediapipe.tasks import python as _mp_tasks
from mediapipe.tasks.python import vision as _mp_vision
import urllib.request, os, tempfile

def _get_model(name, url):
    p = os.path.join(tempfile.gettempdir(), name)
    if not os.path.exists(p):
        print(f"[Setup] Downloading {name}...")
        urllib.request.urlretrieve(url, p)
        print(f"[Setup] Done.")
    return p

_hands_model = _get_model(
    "hand_landmarker.task",
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
_face_model = _get_model(
    "face_landmarker.task",
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)

hands_detector = _mp_vision.HandLandmarker.create_from_options(
    _mp_vision.HandLandmarkerOptions(
        base_options=_mp_tasks.BaseOptions(model_asset_path=_hands_model),
        running_mode=_mp_vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
)
face_mesh_detector = _mp_vision.FaceLandmarker.create_from_options(
    _mp_vision.FaceLandmarkerOptions(
        base_options=_mp_tasks.BaseOptions(model_asset_path=_face_model),
        running_mode=_mp_vision.RunningMode.IMAGE,
        num_faces=4,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
)

# Skeleton connections for hands (new API index pairs)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# ── State ─────────────────────────────────────────────────────────────────────
_lock           = threading.Lock()
_frame_count    = 0
_cached_pose    = []
_last_objects   = []
_last_panel     = _empty()
_last_frame_rgb = None

# Drawing canvas
_canvas         = None
_drawing        = False
_prev_draw_pt   = None
_draw_color     = (0, 255, 128)
_draw_thickness = 6

# Per-frame state shared with Gradio
_shared = {
    "identities": [],
    "gestures":   [],
    "emotions":   [],
    "fps":        0,
}


def _iou(a, b):
    ix1,iy1 = max(a[0],b[0]), max(a[1],b[1])
    ix2,iy2 = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    if inter == 0: return 0.0
    return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter+1e-6)


# ── Gesture detection ─────────────────────────────────────────────────────────

def _finger_up(lm, tip, pip):
    return lm[tip].y < lm[pip].y

def _dist_pts(a, b):
    return math.hypot(a.x-b.x, a.y-b.y)

def detect_gesture(hand_landmarks, handedness):
    lm = hand_landmarks.landmark
    label = handedness.classification[0].label  # Left/Right

    # Finger states
    thumb_up  = lm[4].x < lm[3].x if label=="Right" else lm[4].x > lm[3].x
    index_up  = _finger_up(lm, 8,  6)
    middle_up = _finger_up(lm, 12, 10)
    ring_up   = _finger_up(lm, 16, 14)
    pinky_up  = _finger_up(lm, 20, 18)
    fingers   = [thumb_up, index_up, middle_up, ring_up, pinky_up]
    count     = sum(fingers)

    # Pinch: thumb tip to index tip distance
    pinch_dist = _dist_pts(lm[4], lm[8])
    is_pinch   = pinch_dist < 0.05

    if is_pinch:
        return "Pinch ✏️", lm[8], True   # drawing active
    if count == 0:
        return "Fist ✊", None, False
    if count == 5:
        return "Open Hand 🖐", None, False
    if index_up and not middle_up and not ring_up and not pinky_up:
        return "Pointing ☝️", None, False
    if index_up and middle_up and not ring_up and not pinky_up:
        return "Peace ✌️", None, False
    if thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
        return "Thumbs Up 👍", None, False
    if not thumb_up and not index_up and not middle_up and not ring_up and pinky_up:
        return "Pinky 🤙", None, False
    if index_up and pinky_up and not middle_up and not ring_up:
        return "Rock On 🤘", None, False
    return f"{count} Fingers", None, False


# ── Emotion detection via face mesh landmarks ─────────────────────────────────

def detect_emotion_mesh(face_landmarks, frame_shape):
    """
    Simple rule-based emotion from facial landmark ratios.
    Uses mouth open ratio and eyebrow raise for basic emotions.
    """
    lm = face_landmarks.landmark
    h, w = frame_shape[:2]

    # Mouth open ratio: vertical / horizontal
    mouth_top    = lm[13]   # upper lip
    mouth_bottom = lm[14]   # lower lip
    mouth_left   = lm[61]
    mouth_right  = lm[291]
    mouth_open   = abs(mouth_top.y - mouth_bottom.y)
    mouth_width  = abs(mouth_left.x - mouth_right.x) + 1e-6
    mouth_ratio  = mouth_open / mouth_width

    # Eyebrow raise: distance from eyebrow to eye
    l_brow = lm[70]
    l_eye  = lm[159]
    r_brow = lm[300]
    r_eye  = lm[386]
    brow_raise = ((abs(l_brow.y - l_eye.y) + abs(r_brow.y - r_eye.y)) / 2)

    # Smile: mouth corners up relative to center
    left_corner  = lm[61]
    right_corner = lm[291]
    mouth_center = lm[0]
    smile_score  = ((mouth_center.y - left_corner.y) + (mouth_center.y - right_corner.y)) / 2

    if mouth_ratio > 0.25 and brow_raise > 0.06:
        return "Surprised 😲"
    elif smile_score > 0.01 and mouth_ratio < 0.15:
        return "Happy 😊"
    elif mouth_ratio > 0.2:
        return "Talking 🗣"
    elif brow_raise < 0.03:
        return "Angry 😠"
    elif brow_raise > 0.07:
        return "Worried 😟"
    else:
        return "Neutral 😐"


# ── New API gesture + emotion (landmark list, not object with .landmark) ────────

def detect_gesture_new(lm_list, handedness):
    label = handedness[0].category_name  # Left/Right

    def up(tip, pip): return lm_list[tip].y < lm_list[pip].y
    def dist(a, b): return math.hypot(lm_list[a].x-lm_list[b].x, lm_list[a].y-lm_list[b].y)

    thumb_up  = lm_list[4].x < lm_list[3].x if label=="Right" else lm_list[4].x > lm_list[3].x
    index_up  = up(8,  6)
    middle_up = up(12, 10)
    ring_up   = up(16, 14)
    pinky_up  = up(20, 18)
    count     = sum([thumb_up, index_up, middle_up, ring_up, pinky_up])

    pinch_dist = dist(4, 8)
    is_pinch   = pinch_dist < 0.05

    if is_pinch:
        return "Pinch Draw", lm_list[8], True
    if count == 0:
        return "Fist", None, False
    if count == 5:
        return "Open Hand", None, False
    if index_up and not middle_up and not ring_up and not pinky_up:
        return "Pointing", None, False
    if index_up and middle_up and not ring_up and not pinky_up:
        return "Peace", None, False
    if thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
        return "Thumbs Up", None, False
    if index_up and pinky_up and not middle_up and not ring_up:
        return "Rock On", None, False
    return f"{count} Fingers", None, False


def detect_emotion_new(lm_list, frame_shape):
    mouth_open  = abs(lm_list[13].y - lm_list[14].y)
    mouth_width = abs(lm_list[61].x - lm_list[291].x) + 1e-6
    mouth_ratio = mouth_open / mouth_width
    brow_raise  = (abs(lm_list[70].y - lm_list[159].y) + abs(lm_list[300].y - lm_list[386].y)) / 2
    smile_score = ((lm_list[0].y - lm_list[61].y) + (lm_list[0].y - lm_list[291].y)) / 2
    if mouth_ratio > 0.25 and brow_raise > 0.06:
        return "Surprised"
    elif smile_score > 0.01 and mouth_ratio < 0.15:
        return "Happy"
    elif mouth_ratio > 0.2:
        return "Talking"
    elif brow_raise < 0.03:
        return "Angry"
    elif brow_raise > 0.07:
        return "Worried"
    return "Neutral"


# ── Main process function ─────────────────────────────────────────────────────

def process_frame(frame_bgr):
    global _frame_count, _cached_pose, _last_objects
    global _last_panel, _last_frame_rgb, _canvas
    global _drawing, _prev_draw_pt, _shared

    if _canvas is None or _canvas.shape[:2] != frame_bgr.shape[:2]:
        _canvas = np.zeros_like(frame_bgr)

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    _last_frame_rgb = frame_rgb
    h, w = frame_bgr.shape[:2]
    _frame_count += 1

    # ── Person detection ──────────────────────────────────────────────
    scale = min(1.0, 640/max(h,w,1))
    small = cv2.resize(frame_bgr,(int(w*scale),int(h*scale))) if scale<1.0 else frame_bgr
    dets_small = detector.detect(small)
    detections = []
    for d in dets_small:
        x1,y1,x2,y2 = d["bbox"]
        if scale < 1.0:
            x1,y1,x2,y2 = int(x1/scale),int(y1/scale),int(x2/scale),int(y2/scale)
        detections.append({"bbox":[x1,y1,x2,y2],"conf":d["conf"]})

    # ── Objects every 8th frame ───────────────────────────────────────
    if _frame_count % 8 == 0:
        objs = obj_detector.detect(small)
        if scale < 1.0:
            for o in objs:
                ox1,oy1,ox2,oy2 = o["bbox"]
                o["bbox"] = [int(ox1/scale),int(oy1/scale),int(ox2/scale),int(oy2/scale)]
        _last_objects = objs

    # ── Pose every 3rd frame ──────────────────────────────────────────
    if _frame_count % 3 == 0 or not _cached_pose:
        pose_results = []
        for det in detections:
            x1,y1,x2,y2 = det["bbox"]
            crop = frame_rgb[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
            if crop.size:
                ch,cw = crop.shape[:2]
                ps = min(1.0,256/max(ch,cw,1))
                if ps < 1.0:
                    crop = cv2.resize(crop,(int(cw*ps),int(ch*ps)))
            pose_results.append(pose_est.estimate(crop) if crop.size else None)
        _cached_pose = pose_results
    else:
        pose_results = _cached_pose

    # ── Face recognition every 20th frame ────────────────────────────
    if _frame_count % 20 == 0 and face_mgr.available:
        for det in detections:
            name = face_mgr.identify(frame_rgb, det["bbox"])
            if name:
                for identity in id_manager.get_all():
                    if _iou(identity.bbox, det["bbox"]) > 0.3:
                        identity.name = name
                        break

    active = id_manager.update(detections, pose_results)

    # ── Hand detection + gesture + drawing ───────────────────────────
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    hand_results = hands_detector.detect(mp_img)
    gestures_this_frame = []
    drawing_now = False
    draw_pt     = None

    if hand_results.hand_landmarks:
        for hand_lm, handedness in zip(
            hand_results.hand_landmarks,
            hand_results.handedness
        ):
            gesture_name, tip_lm, is_pinch = detect_gesture_new(hand_lm, handedness)
            gestures_this_frame.append(gesture_name)

            # Draw hand skeleton manually
            for a, b in HAND_CONNECTIONS:
                ax, ay = int(hand_lm[a].x*w), int(hand_lm[a].y*h)
                bx, by = int(hand_lm[b].x*w), int(hand_lm[b].y*h)
                cv2.line(frame_bgr, (ax,ay), (bx,by), (255,255,255), 1, cv2.LINE_AA)
            for lm in hand_lm:
                cv2.circle(frame_bgr, (int(lm.x*w), int(lm.y*h)), 4, (0,200,255), -1)

            # Gesture label near wrist
            wrist = hand_lm[0]
            gx, gy = int(wrist.x*w), int(wrist.y*h)
            cv2.putText(frame_bgr, gesture_name, (gx-40, gy+30),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75, (0,200,255), 2, cv2.LINE_AA)

            if is_pinch and tip_lm is not None:
                drawing_now = True
                draw_pt = (int(tip_lm.x*w), int(tip_lm.y*h))

    # Apply drawing
    if drawing_now and draw_pt:
        if _prev_draw_pt is not None:
            cv2.line(_canvas, _prev_draw_pt, draw_pt, _draw_color, _draw_thickness)
        _prev_draw_pt = draw_pt
    else:
        _prev_draw_pt = None

    # Overlay canvas
    canvas_gray = cv2.cvtColor(_canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(canvas_gray, 1, 255, cv2.THRESH_BINARY)
    frame_bgr[mask > 0] = _canvas[mask > 0]

    # ── Emotion detection every 5th frame ────────────────────────────
    emotions_this_frame = []
    if _frame_count % 5 == 0:
        mesh_res = face_mesh_detector.detect(mp_img)
        if mesh_res.face_landmarks:
            for i, face_lm_list in enumerate(mesh_res.face_landmarks):
                emotion = detect_emotion_new(face_lm_list, frame_bgr.shape)
                emotions_this_frame.append(emotion)
                # Draw emotion near top of face
                nose = face_lm_list[1]
                ex, ey = int(nose.x*w), int(nose.y*h) - 20
                cv2.putText(frame_bgr, emotion, (ex-60, ey),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,200,0), 2, cv2.LINE_AA)

    # ── Annotate persons ─────────────────────────────────────────────
    annotated = draw_frame(frame_bgr, active, pose_results, objects=_last_objects)

    # ── Update shared state for Gradio ───────────────────────────────
    if _frame_count % 5 == 0:
        with _lock:
            _shared["identities"] = id_manager.get_summary()
            _shared["gestures"]   = gestures_this_frame
            _shared["emotions"]   = emotions_this_frame if emotions_this_frame else _shared["emotions"]

    return annotated


# ── OpenCV loop ───────────────────────────────────────────────────────────────

def run_opencv():
    global _canvas, _draw_color

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS,          60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    cv2.namedWindow("Identity & Posture System", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Identity & Posture System", 1280, 720)

    fps_time  = time.time()
    fps_count = 0
    fps_val   = 0

    colors = [(0,255,128),(0,128,255),(255,100,0),(200,0,255),(0,255,255)]
    color_idx = 0

    print("\n[SYSTEM] Controls:")
    print("  Q       — quit")
    print("  C       — clear drawing canvas")
    print("  1-5     — change draw color")
    print("  Pinch   — draw on screen\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated = process_frame(frame)

        fps_count += 1
        if time.time() - fps_time >= 1.0:
            fps_val   = fps_count
            fps_count = 0
            fps_time  = time.time()
            with _lock:
                _shared["fps"] = fps_val

        cv2.putText(annotated, f"FPS: {fps_val}  |  Q=quit  C=clear  1-5=color",
                    (10, annotated.shape[0]-14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80,80,80), 1, cv2.LINE_AA)

        cv2.imshow("Identity & Posture System", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            _canvas = np.zeros_like(frame)
            print("[Drawing] Canvas cleared.")
        elif ord('1') <= key <= ord('5'):
            _draw_color = colors[key - ord('1')]
            print(f"[Drawing] Color changed.")

    cap.release()
    cv2.destroyAllWindows()


# ── Gradio panel ──────────────────────────────────────────────────────────────

def get_panel_html():
    with _lock:
        data       = dict(_shared)
        identities = data.get("identities", [])
        gestures   = data.get("gestures",   [])
        emotions   = data.get("emotions",   [])
        fps        = data.get("fps",        0)

    # Big font cards
    cards = ""
    for p in identities:
        score = p.get("posture_score") or 0
        sc    = "#00e676" if score>=75 else "#ffab00" if score>=50 else "#ff3d00"
        em    = p.get("emotion") or "—"
        pos   = p.get("body_position") or "—"
        cards += f"""
        <div style="background:#0f1629;border-radius:10px;padding:18px 20px;
                    border-left:4px solid #00b0ff;margin-bottom:14px">
          <div style="font-size:28px;font-weight:700;color:#00b0ff;margin-bottom:8px">
            {p['label']}
          </div>
          <div style="font-size:22px;color:{sc};margin-bottom:6px">
            {pos}
          </div>
          <div style="font-size:18px;color:#90caf9;margin-bottom:8px">
            {p.get('posture_label','—')}
          </div>
          <div style="background:#1e2a40;border-radius:4px;height:10px;margin-bottom:6px">
            <div style="width:{score}%;height:100%;background:{sc};border-radius:4px"></div>
          </div>
          <div style="font-size:20px;color:{sc};font-weight:700">{score}% posture</div>
          <div style="font-size:18px;color:#ffcc02;margin-top:6px">Emotion: {em}</div>
        </div>
        """

    if not cards:
        cards = '<div style="font-size:22px;color:#37474f;text-align:center;padding:40px">No persons detected</div>'

    gest_html = ""
    if gestures:
        gest_html = '<div style="font-size:24px;color:#00e5ff;margin-top:14px"><b>Gestures:</b> ' + \
                    " &nbsp; ".join(gestures) + "</div>"

    emo_html = ""
    if emotions:
        emo_html = '<div style="font-size:24px;color:#ffcc02;margin-top:8px"><b>Emotions:</b> ' + \
                   " &nbsp; ".join(emotions) + "</div>"

    fps_html = f'<div style="font-size:16px;color:#37474f;font-family:monospace;margin-top:10px">FPS: {fps}</div>'

    return f"""
    <div style="font-family:sans-serif;background:#0a0e1a;padding:14px;border-radius:10px">
      {cards}
      {gest_html}
      {emo_html}
      {fps_html}
    </div>
    """


def on_assign(uid_str, name):
    try:
        uid  = int(uid_str.strip())
        name = name.strip()
        if not name: return "Name cannot be empty."
        id_manager.assign_name(uid, name)
        return f"✅ Person {uid} -> '{name}'"
    except ValueError:
        return "Enter a valid numeric ID."


def on_register_face(uid_str, name):
    global _last_frame_rgb
    try:
        uid  = int(uid_str.strip())
        name = name.strip()
        if not name: return "Enter a name first."
        if not face_mgr.available: return "face-recognition not installed."
        if _last_frame_rgb is None: return "Camera not started yet."
        target = next((i for i in id_manager.get_all() if i.uid == uid), None)
        if target is None: return f"Person {uid} not visible right now."
        ok = face_mgr.register(_last_frame_rgb, target.bbox, name)
        if ok:
            id_manager.assign_name(uid, name)
            return f"✅ '{name}' saved permanently to face DB!"
        return "⚠️ No face detected in bbox. Make sure face is visible."
    except Exception as e:
        return f"Error: {str(e)}"


def on_clear_canvas():
    global _canvas
    _canvas = None   # gets re-initialised as zeros on next frame
    return "Canvas cleared."


DRAW_COLORS = {
    "Green":  (0, 255, 128),
    "Blue":   (0, 128, 255),
    "Red":    (0, 60, 255),
    "Yellow": (0, 220, 220),
    "White":  (255, 255, 255),
    "Purple": (200, 0, 255),
}

def on_set_color(color_name):
    global _draw_color, _draw_thickness
    _draw_color = DRAW_COLORS.get(color_name, (0,255,128))
    return f"Draw color: {color_name}"


def on_set_thickness(val):
    global _draw_thickness
    _draw_thickness = int(val)
    return f"Thickness: {val}"


def on_reset():
    id_manager._identities.clear()
    id_manager._next_uid = 1
    return "Reset done. Face DB preserved."


def on_list():
    names = face_mgr.known_names()
    return "Known: " + (", ".join(names) if names else "none yet")


def on_delete_face(name):
    name = name.strip()
    if not name: return "Enter name to delete."
    face_mgr.delete(name)
    return f"Deleted '{name}' from DB."


CSS = """
body, .gradio-container { background:#060b14 !important; color:#cdd9e5 !important; }
footer { display:none !important }
"""


def build_gradio():
    with gr.Blocks(title="Identity Panel") as app:
        gr.HTML("""
        <div style="text-align:center;padding:14px 0 10px;border-bottom:1px solid #0d1f35;margin-bottom:12px">
          <div style="font-size:20px;font-weight:700;color:#00b0ff;letter-spacing:3px;text-transform:uppercase">
            Identity · Posture · Gesture · Emotion
          </div>
          <div style="font-size:11px;color:#37474f;letter-spacing:2px;margin-top:4px;font-family:monospace">
            OpenCV window is the live feed · Pinch to draw · Press C to clear · Q to quit
          </div>
        </div>
        """)

        panel = gr.HTML(value=get_panel_html())

        gr.HTML('<div style="font-size:11px;color:#37474f;font-family:monospace;margin:12px 0 5px">ASSIGN NAME</div>')
        with gr.Row():
            uid_box  = gr.Textbox(placeholder="ID", label="", show_label=False, scale=1)
            name_box = gr.Textbox(placeholder="Name", label="", show_label=False, scale=2)
            btn_assign = gr.Button("Assign", scale=1)

        gr.HTML('<div style="font-size:11px;color:#37474f;font-family:monospace;margin:10px 0 5px">SAVE FACE TO DB</div>')
        with gr.Row():
            uid_face   = gr.Textbox(placeholder="ID", label="", show_label=False, scale=1)
            name_face  = gr.Textbox(placeholder="Name", label="", show_label=False, scale=2)
            btn_face   = gr.Button("Save Face", variant="primary", scale=1)

        gr.HTML('<div style="font-size:11px;color:#37474f;font-family:monospace;margin:10px 0 5px">DELETE FACE FROM DB</div>')
        with gr.Row():
            del_name = gr.Textbox(placeholder="Name to delete", label="", show_label=False, scale=3)
            btn_del  = gr.Button("Delete", variant="stop", scale=1)

        status = gr.Textbox(label="", value="", show_label=False, interactive=False, max_lines=1)

        gr.HTML('<div style="font-size:11px;color:#37474f;font-family:monospace;margin:10px 0 5px">DRAWING CONTROLS</div>')
        with gr.Row():
            color_dd  = gr.Dropdown(
                choices=["Green","Blue","Red","Yellow","White","Purple"],
                value="Green", label="Color", scale=2
            )
            thick_sl  = gr.Slider(minimum=2, maximum=20, value=6, step=1, label="Thickness", scale=2)
            btn_clear = gr.Button("Clear Canvas", variant="stop", scale=1)

        with gr.Row():
            btn_list  = gr.Button("List Faces", variant="secondary")
            btn_reset = gr.Button("Reset Tracking", variant="secondary")

        timer = gr.Timer(value=0.5)
        timer.tick(get_panel_html, outputs=panel)

        btn_assign.click(on_assign,       inputs=[uid_box, name_box], outputs=status)
        btn_face.click(on_register_face,  inputs=[uid_face, name_face], outputs=status)
        btn_del.click(on_delete_face,     inputs=del_name, outputs=status)
        btn_list.click(on_list,           outputs=status)
        btn_reset.click(on_reset,         outputs=status)
        btn_clear.click(on_clear_canvas,  outputs=status)
        color_dd.change(on_set_color,     inputs=color_dd,  outputs=status)
        thick_sl.change(on_set_thickness, inputs=thick_sl,  outputs=status)

    app.launch(server_name="127.0.0.1", server_port=7860,
               share=False, quiet=True, css=CSS)


if __name__ == "__main__":
    t = threading.Thread(target=build_gradio, daemon=True)
    t.start()
    print("[SYSTEM] Gradio panel → http://127.0.0.1:7860")
    run_opencv()