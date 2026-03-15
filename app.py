"""
app.py — Identity & Posture Analysis System
- 60fps: JS grabs webcam, sends base64 frames to Python, gets annotated base64 back
- Face recognition: persistent face_db.pkl, remembers faces across all runs
- Object detection: all COCO objects shown on frame
"""

import cv2
import numpy as np
import base64
import gradio as gr

from detector         import PersonDetector
from pose_estimator   import PoseEstimator
from identity_manager import IdentityManager
from face_manager     import FaceManager
from object_detector  import ObjectDetector
from annotator        import draw_frame
from qa_engine        import QAEngine, _empty
import config

detector     = PersonDetector()
obj_detector = ObjectDetector()
pose_est     = PoseEstimator()
id_manager   = IdentityManager()
face_mgr     = FaceManager()
qa_engine    = QAEngine()

_frame_count  = 0
_cached_pose  = []
_last_objects = []
_last_panel   = _empty()
_last_frame_rgb = None


def _iou(a, b):
    ix1,iy1 = max(a[0],b[0]), max(a[1],b[1])
    ix2,iy2 = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    if inter == 0: return 0.0
    return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter+1e-6)


def process_b64(b64_str: str):
    global _frame_count, _cached_pose, _last_objects, _last_panel, _last_frame_rgb

    if not b64_str:
        return "", _last_panel

    try:
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_str)
        arr = np.frombuffer(img_bytes, np.uint8)
        frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return "", _last_panel
    except Exception:
        return "", _last_panel

    h, w = frame_bgr.shape[:2]
    _last_frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Scale down for detection
    scale = min(1.0, 640 / max(h, w, 1))
    small = cv2.resize(frame_bgr, (int(w*scale), int(h*scale))) if scale < 1.0 else frame_bgr

    # Person detection
    dets_small = detector.detect(small)
    detections = []
    for d in dets_small:
        x1,y1,x2,y2 = d["bbox"]
        if scale < 1.0:
            x1,y1,x2,y2 = int(x1/scale),int(y1/scale),int(x2/scale),int(y2/scale)
        detections.append({"bbox":[x1,y1,x2,y2],"conf":d["conf"]})

    _frame_count += 1

    # Object detection every 3rd frame
    if _frame_count % 3 == 0:
        objs = obj_detector.detect(small)
        if scale < 1.0:
            for o in objs:
                ox1,oy1,ox2,oy2 = o["bbox"]
                o["bbox"] = [int(ox1/scale),int(oy1/scale),int(ox2/scale),int(oy2/scale)]
        _last_objects = objs

    # Pose every 2nd frame
    if _frame_count % 2 == 0 or not _cached_pose:
        pose_results = []
        for det in detections:
            x1,y1,x2,y2 = det["bbox"]
            crop = _last_frame_rgb[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
            if crop.size:
                ch,cw = crop.shape[:2]
                ps = min(1.0, 256/max(ch,cw,1))
                if ps < 1.0:
                    crop = cv2.resize(crop,(int(cw*ps),int(ch*ps)))
            pose_results.append(pose_est.estimate(crop) if crop.size else None)
        _cached_pose = pose_results
    else:
        pose_results = _cached_pose

    # Face recognition every 5th frame
    if _frame_count % 5 == 0 and face_mgr.available and _last_frame_rgb is not None:
        for det in detections:
            name = face_mgr.identify(_last_frame_rgb, det["bbox"])
            if name:
                for identity in id_manager.get_all():
                    if _iou(identity.bbox, det["bbox"]) > 0.35:
                        identity.name = name
                        break

    active    = id_manager.update(detections, pose_results)
    annotated = draw_frame(frame_bgr, active, pose_results, objects=_last_objects)

    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
    out_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

    if _frame_count % 3 == 0:
        panel_data = qa_engine.generate_panel(id_manager.get_summary())
        _last_panel = qa_engine.to_html(panel_data)

    return out_b64, _last_panel


def on_frame(b64):
    return process_b64(b64 or "")


def on_assign(uid_str, name):
    try:
        uid  = int(uid_str.strip())
        name = name.strip()
        if not name:
            return "⚠️ Name cannot be empty."
        id_manager.assign_name(uid, name)
        return f"✅ Person {uid} → '{name}'"
    except ValueError:
        return "⚠️ Enter a valid numeric ID."


def on_register_face(uid_str, name):
    global _last_frame_rgb
    try:
        uid  = int(uid_str.strip())
        name = name.strip()
        if not name:
            return "⚠️ Enter a name first."
        if not face_mgr.available:
            return "⚠️ Run: pip install face-recognition"
        if _last_frame_rgb is None:
            return "⚠️ Start camera first."
        target = next((i for i in id_manager.get_all() if i.uid == uid), None)
        if target is None:
            return f"⚠️ Person {uid} not currently visible."
        ok = face_mgr.register(_last_frame_rgb, target.bbox, name)
        if ok:
            id_manager.assign_name(uid, name)
            return f"✅ '{name}' saved permanently — will be recognised on next run!"
        return "⚠️ No face detected in bounding box."
    except Exception as e:
        return f"⚠️ {e}"


def on_reset():
    id_manager._identities.clear()
    id_manager._next_uid = 1
    return "Reset done. Face DB preserved."


def on_list_known():
    names = face_mgr.known_names()
    return "Known: " + (", ".join(names) if names else "none yet")


# ── UI ────────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Rajdhani:wght@500;700&display=swap');
body, .gradio-container { background:#060b14 !important; color:#cdd9e5 !important; }
.hdr { text-align:center; padding:20px 0 14px; border-bottom:1px solid #0d1f35; margin-bottom:16px }
.htitle { font-family:'Rajdhani',sans-serif; font-size:26px; font-weight:700; color:#00b0ff; letter-spacing:3px; text-transform:uppercase }
.hsub { font-family:'Space Mono',monospace; font-size:10px; color:#37474f; letter-spacing:2px; margin-top:4px }
.plabel { font-family:'Space Mono',monospace; font-size:10px; letter-spacing:2px; color:#37474f; text-transform:uppercase; margin-bottom:5px }
footer { display:none !important }
#cam-canvas { width:100%; border-radius:8px; border:1px solid #1e2a40; background:#000; display:block; min-height:460px }
"""

# JS: grab webcam at up to 60fps, send raw frame to Python, render annotated back
JS_START = """
async () => {
    if (window._camRunning) return;
    window._camRunning = true;

    const canvas  = document.getElementById('cam-canvas');
    const ctx     = canvas.getContext('2d');
    const offscr  = document.createElement('canvas');

    let stream;
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { width:{ideal:1280}, height:{ideal:720}, frameRate:{ideal:60,max:60} }
        });
    } catch(e) {
        console.error('Camera denied:', e); return;
    }

    const video = document.createElement('video');
    video.srcObject = stream;
    video.play();
    window._stopCam = () => {
        window._camRunning = false;
        stream.getTracks().forEach(t=>t.stop());
    };

    // Wait for metadata
    await new Promise(r => video.addEventListener('loadedmetadata', r, {once:true}));
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    offscr.width  = video.videoWidth;
    offscr.height = video.videoHeight;

    let sending = false;

    function getHiddenInput(elemId) {
        const wrap = document.getElementById(elemId);
        if (!wrap) return null;
        return wrap.querySelector('textarea') || wrap.querySelector('input');
    }

    function setInput(inp, val) {
        const proto = inp.tagName === 'TEXTAREA'
            ? window.HTMLTextAreaElement.prototype
            : window.HTMLInputElement.prototype;
        const setter = Object.getOwnPropertyDescriptor(proto, 'value');
        setter.set.call(inp, val);
        inp.dispatchEvent(new Event('input', {bubbles:true}));
    }

    // Watch output textbox → draw on canvas
    const outWrap = document.getElementById('frame-output');
    if (outWrap) {
        new MutationObserver(() => {
            const ta = outWrap.querySelector('textarea') || outWrap.querySelector('input');
            if (!ta || !ta.value || !ta.value.startsWith('data:')) return;
            const img = new Image();
            img.onload = () => {
                canvas.width  = img.naturalWidth  || canvas.width;
                canvas.height = img.naturalHeight || canvas.height;
                ctx.drawImage(img, 0, 0);
            };
            img.src = ta.value;
            ta.value = '';   // clear so next mutation fires
        }).observe(outWrap, {subtree:true, characterData:true, childList:true, attributes:true});
    }

    async function loop() {
        if (!window._camRunning) return;
        if (!sending) {
            sending = true;
            offscr.getContext('2d').drawImage(video, 0, 0);
            const b64 = offscr.toDataURL('image/jpeg', 0.82);
            const inp = getHiddenInput('frame-input');
            if (inp) setInput(inp, b64);
            sending = false;
        }
        requestAnimationFrame(loop);
    }
    loop();
}
"""

JS_STOP = "() => { if(window._stopCam) window._stopCam(); window._camRunning=false; }"

with gr.Blocks(title="Identity & Posture System") as demo:

    gr.HTML("""
    <div class="hdr">
      <div class="htitle">⬡ Identity · Posture · Object Detection</div>
      <div class="hsub">60FPS · PERSISTENT FACE RECOGNITION · ALL COCO OBJECTS · POSE ANALYSIS</div>
    </div>
    """)

    with gr.Row(equal_height=True):

        with gr.Column(scale=3):
            gr.HTML('<div class="plabel">📡 Live Feed</div>')
            gr.HTML('<canvas id="cam-canvas"></canvas>')

            frame_input  = gr.Textbox(visible=False, elem_id="frame-input")
            frame_output = gr.Textbox(visible=False, elem_id="frame-output")

            with gr.Row():
                btn_start = gr.Button("📷 Start Camera", variant="primary", scale=3)
                btn_stop  = gr.Button("⏹ Stop", variant="stop", scale=1)
                btn_reset = gr.Button("↺ Reset", variant="secondary", scale=1)

            status = gr.Textbox(label="", value="Click 'Start Camera'",
                                show_label=False, interactive=False, max_lines=1)

        with gr.Column(scale=2):
            gr.HTML('<div class="plabel">🧠 Identity Panel</div>')
            side_panel = gr.HTML(value=_empty(), show_label=False)

            gr.HTML('<div class="plabel" style="margin-top:12px">✏️ Assign Name (this session)</div>')
            with gr.Row():
                uid_box    = gr.Textbox(placeholder="ID", label="", show_label=False, scale=1)
                name_box   = gr.Textbox(placeholder="Name", label="", show_label=False, scale=2)
                btn_assign = gr.Button("Assign", scale=1)

            gr.HTML('<div class="plabel" style="margin-top:10px">💾 Save Face to DB (permanent)</div>')
            with gr.Row():
                uid_face   = gr.Textbox(placeholder="ID", label="", show_label=False, scale=1)
                name_face  = gr.Textbox(placeholder="Name", label="", show_label=False, scale=2)
                btn_face   = gr.Button("Save Face", variant="primary", scale=1)

            assign_status = gr.Textbox(label="", value="",
                                       show_label=False, interactive=False, max_lines=1)

            with gr.Row():
                btn_list = gr.Button("📋 List Known Faces", variant="secondary")

    # Wiring
    frame_input.change(on_frame, inputs=frame_input, outputs=[frame_output, side_panel])
    btn_start.click(None, js=JS_START)
    btn_start.click(lambda: "🟢 Camera running", outputs=status)
    btn_stop.click(None, js=JS_STOP)
    btn_stop.click(lambda: "⏹ Stopped.", outputs=status)
    btn_reset.click(on_reset, outputs=status)
    btn_assign.click(on_assign, inputs=[uid_box, name_box], outputs=assign_status)
    btn_face.click(on_register_face, inputs=[uid_face, name_face], outputs=assign_status)
    btn_list.click(on_list_known, outputs=assign_status)


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,
        css=CSS,
    )