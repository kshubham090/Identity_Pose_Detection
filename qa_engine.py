import time
import random

_ID_Q = [
    "Who is {label}? Assign their name below.",
    "Do you recognise {label}? ({frames} frames tracked)",
    "New person detected: {label}. Enter a name to identify them.",
]
_BAD_Q = [
    "Should {label} be notified about their {issue}?",
    "{label}'s posture score is {score}% — take action?",
    "Alert: {label} shows {issue}. Intervene?",
]
_GOOD_Q = [
    "{label} has excellent posture! ({score}%)",
    "Great alignment for {label} — score: {score}%.",
]


class QAEngine:
    def __init__(self):
        self._last: dict[int, dict] = {}
        self._cooldown = 4.0

    def generate_panel(self, summaries: list[dict]) -> list[dict]:
        now = time.time()
        panel = []
        for s in summaries:
            uid, label = s["uid"], s["label"]
            score, issues, frames = s["posture_score"], s["posture_issues"], s["frames_tracked"]

            last = self._last.get(uid, {})
            question = last.get("text", "")

            if now - last.get("ts", 0) > self._cooldown:
                if label.startswith("Person "):
                    question = random.choice(_ID_Q).format(label=label, frames=frames)
                elif issues:
                    question = random.choice(_BAD_Q).format(label=label, score=score, issue=issues[0].lower())
                else:
                    question = random.choice(_GOOD_Q).format(label=label, score=score)
                self._last[uid] = {"text": question, "ts": now}

            panel.append({**s, "question": question})
        return panel

    def to_html(self, panel: list[dict]) -> str:
        if not panel:
            return _empty()

        cards = ""
        for p in panel:
            score = p["posture_score"] or 0
            is_unknown = p["label"].startswith("Person ")
            sc = "#00e676" if score >= 75 else "#ffab00" if score >= 50 else "#ff3d00"
            border = "#00b0ff" if is_unknown else sc
            issues_html = "".join(f'<span class="itag">{i}</span>' for i in p["posture_issues"])
            metrics = ""
            if p["shoulder_tilt"] is not None:
                metrics = f"""
                <div class="mrow"><span>Shoulder Tilt</span><span>{p['shoulder_tilt']}°</span></div>
                <div class="mrow"><span>Spine Angle</span><span>{p['spine_angle']}°</span></div>
                <div class="mrow"><span>Fwd Head</span><span>{p['fwd_head']}</span></div>
                """
            cards += f"""
            <div class="card" style="border-left:3px solid {border}">
              <div class="cheader">
                <span style="color:{border};font-weight:700;font-size:15px">{p['label']}</span>
                <span style="color:#37474f;font-size:11px">#{p['frames_tracked']} frames</span>
              </div>
              <div class="qbubble">
                <span class="qico">?</span>
                <span style="color:#90caf9;font-size:12.5px">{p['question']}</span>
              </div>
              <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
                <span style="color:{sc};font-size:12px;font-weight:600;min-width:130px">{p['posture_label']}</span>
                <div style="flex:1;height:6px;background:#1e2a40;border-radius:3px;overflow:hidden">
                  <div style="width:{score}%;height:100%;background:{sc};border-radius:3px"></div>
                </div>
                <span style="color:{sc};font-size:12px;font-weight:700">{score}%</span>
              </div>
              <div style="display:flex;flex-wrap:wrap;gap:5px;margin-bottom:6px">{issues_html}</div>
              <div class="metrics">{metrics}</div>
            </div>
            """

        return f"""
        <style>
          .panel {{ font-family:'JetBrains Mono',monospace; background:#0a0e1a; padding:12px; border-radius:10px; display:flex; flex-direction:column; gap:12px }}
          .card  {{ background:#0f1629; border-radius:8px; padding:14px 16px; border:1px solid #1e2a40 }}
          .cheader {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:10px }}
          .qbubble {{ background:#131d33; border-radius:6px; padding:8px 12px; margin-bottom:12px; display:flex; gap:8px; align-items:flex-start }}
          .qico {{ background:#1565c0; color:#fff; border-radius:50%; width:18px; height:18px; display:inline-flex; align-items:center; justify-content:center; font-size:11px; font-weight:700; flex-shrink:0; margin-top:1px }}
          .itag {{ background:#ff3d0022; color:#ff6e40; border:1px solid #ff3d0055; border-radius:4px; font-size:10px; padding:2px 7px }}
          .metrics {{ border-top:1px solid #1e2a40; padding-top:6px }}
          .mrow {{ display:flex; justify-content:space-between; font-size:11px; color:#546e7a; margin-bottom:3px }}
        </style>
        <div class="panel">{cards}</div>
        """


def _empty():
    return """
    <div style="font-family:monospace;background:#0a0e1a;border-radius:10px;
                padding:40px 20px;text-align:center;color:#263348">
        <div style="font-size:48px">👁</div>
        <div style="font-size:13px;color:#37474f;margin-top:12px">No persons detected</div>
    </div>
    """
