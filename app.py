import cv2
import numpy as np
import threading
import time
import gradio as gr

from detector import PersonDetector
from pose_estimator import PoseEstimator
from identity_manager import IdentityManager
from annotator import draw_frame
from qa_engine import QAEngine, _empty
import config

detector   = PersonDetector()
pose_est   = PoseEstimator()
id_manager = IdentityManager()
qa_engine  = QAEngine()

_lock         = threading.Lock()
_latest_frame = None
_latest_panel = None
_running      = False
_cap          = None


# ── Pipeline ──────────────────────────────────────────────────────────────────

def process_frame(frame_bgr):
    global _latest_frame, _latest_panel

    frame_rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    detections = detector.detect(frame_bgr)

    pose_results = []
    h, w = frame_bgr.shape[:2]
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        crop = frame_rgb[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
        pose_results.append(pose_est.estimate(crop) if crop.size else None)

    active     = id_manager.update(detections, pose_results)
    annotated  = draw_frame(frame_bgr, active, pose_results)
    annotated  = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    panel_data = qa_engine.generate_panel(id_manager.get_summary())
    panel_html = qa_engine.to_html(panel_data)

    with _lock:
        _latest_frame = annotated
        _latest_panel = panel_html


# ── Camera thread ─────────────────────────────────────────────────────────────

def _cam_loop(source):
    global _running, _cap
    _cap = cv2.VideoCapture(source)
    _cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    _cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    _cap.set(cv2.CAP_PROP_FPS,          config.MAX_FPS)
    while _running:
        ret, frame = _cap.read()
        if not ret:
            break
        process_frame(frame)
        time.sleep(1 / config.MAX_FPS)
    _cap.release()


def start_cam(source=0):
    global _running
    if _running:
        return
    _running = True
    threading.Thread(target=_cam_loop, args=(source,), daemon=True).start()


def stop_cam():
    global _running
    _running = False


# ── Gradio callbacks ──────────────────────────────────────────────────────────

def poll():
    with _lock:
        frame = _latest_frame
        panel = _latest_panel
    if frame is None:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "Waiting for feed...", (140, 240),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (40,70,100), 1)
        frame = blank
    return frame, panel or _empty()


def on_start():
    id_manager._identities.clear()
    id_manager._next_uid = 1
    start_cam(0)
    return gr.update(value="🟢 Camera Running", variant="secondary")


def on_stop():
    stop_cam()
    return gr.update(value="📷 Start Webcam", variant="primary")


def on_reset():
    stop_cam()
    id_manager._identities.clear()
    id_manager._next_uid = 1
    return gr.update(value="📷 Start Webcam", variant="primary"), "System reset."


def on_video(video_path):
    if not video_path:
        return None, _empty()
    id_manager._identities.clear()
    id_manager._next_uid = 1
    cap = cv2.VideoCapture(video_path)
    last_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        process_frame(frame)
        with _lock:
            last_frame = _latest_frame
    cap.release()
    with _lock:
        panel = _latest_panel
    return last_frame, panel or _empty()


def on_assign(uid_str, name):
    try:
        uid = int(uid_str.strip())
        name = name.strip()
        if not name:
            return "⚠️ Name cannot be empty."
        id_manager.assign_name(uid, name)
        return f"✅ Person {uid} → '{name}'"
    except ValueError:
        return "⚠️ Enter a valid numeric ID."


# ── UI ────────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Rajdhani:wght@500;700&display=swap');
body, .gradio-container { background:#060b14 !important; color:#cdd9e5 !important; }
.hdr { text-align:center; padding:24px 0 16px; border-bottom:1px solid #0d1f35; margin-bottom:18px }
.htitle { font-family:'Rajdhani',sans-serif; font-size:28px; font-weight:700; color:#00b0ff; letter-spacing:3px; text-transform:uppercase; margin:0 }
.hsub { font-family:'Space Mono',monospace; font-size:10px; color:#37474f; letter-spacing:2px; margin-top:4px }
.plabel { font-family:'Space Mono',monospace; font-size:10px; letter-spacing:2px; color:#37474f; text-transform:uppercase; margin-bottom:6px }
footer { display:none !important }
"""

with gr.Blocks(css=CSS, title="Identity & Posture System",
               theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate")) as demo:

    gr.HTML("""
    <div class="hdr">
      <div class="htitle">⬡ Identity & Posture Analysis</div>
      <div class="hsub">REAL-TIME · PERSON DETECTION · POSE ESTIMATION · IDENTITY TRACKING</div>
    </div>
    """)

    with gr.Row(equal_height=True):

        # Left: feed + controls
        with gr.Column(scale=3):
            gr.HTML('<div class="plabel">📡 Live Feed</div>')
            live_img = gr.Image(label="", height=460, show_label=False, interactive=False)

            with gr.Row():
                btn_start = gr.Button("📷 Start Webcam", variant="primary", scale=2)
                btn_stop  = gr.Button("⏹ Stop", variant="stop", scale=1)
                btn_reset = gr.Button("↺ Reset", variant="secondary", scale=1)

            gr.HTML('<div class="plabel" style="margin-top:14px">🎞 Upload Video</div>')
            video_upload = gr.Video(label="", height=140, show_label=False)
            btn_proc = gr.Button("▶ Process Video", variant="secondary")

            status = gr.Textbox(label="", value="System idle.", show_label=False,
                                interactive=False, max_lines=1)

        # Right: side panel + identity assign
        with gr.Column(scale=2):
            gr.HTML('<div class="plabel">🧠 Identity Panel</div>')
            side_panel = gr.HTML(value=_empty(), show_label=False)

            gr.HTML('<div class="plabel" style="margin-top:14px">✏️ Assign Identity</div>')
            with gr.Row():
                uid_box  = gr.Textbox(placeholder="ID (e.g. 1)", label="", show_label=False, scale=1)
                name_box = gr.Textbox(placeholder="Real name", label="", show_label=False, scale=2)
                btn_assign = gr.Button("Assign", scale=1)
            assign_status = gr.Textbox(label="", value="", show_label=False,
                                       interactive=False, max_lines=1)

    # ── Wiring ──────────────────────────────────────────────────────────────
    btn_start.click(on_start, outputs=btn_start)
    btn_stop.click(on_stop, outputs=btn_start)
    btn_reset.click(on_reset, outputs=[btn_start, status])
    btn_proc.click(on_video, inputs=video_upload, outputs=[live_img, side_panel])
    btn_assign.click(on_assign, inputs=[uid_box, name_box], outputs=assign_status)

    # Auto-poll every 100ms
    timer = gr.Timer(value=0.1)
    timer.tick(poll, outputs=[live_img, side_panel])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
