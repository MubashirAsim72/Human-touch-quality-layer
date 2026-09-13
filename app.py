import json
import html
import os
import textwrap
from typing import Any, Dict, List

import streamlit as st
from groq import Groq

# ============================================================
# HELPERS
# ============================================================
def safe_text(value: Any) -> str:
    return html.escape(str(value or "").strip())


def clamp_score(value: Any) -> int:
    try:
        return max(0, min(100, int(round(float(value)))))
    except Exception:
        return 0


def get_api_key() -> str:
    try:
        value = st.secrets.get("GROQ_API_KEY")
        if value:
            return str(value).strip()
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY", "").strip()


def get_model() -> str:
    try:
        value = st.secrets.get("GROQ_MODEL")
        if value:
            return str(value).strip()
    except Exception:
        pass
    return os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()




def recommendation_card(item: Dict[str, Any], number: int) -> str:
    title = safe_text(item.get("title", "Recommended improvement"))
    explanation = safe_text(item.get("explanation", ""))

    return f"""
    <div class="recommendation">
        <span class="recommendation-number">0{number}</span>
        <b>{title}</b>
        <div style="color:#aab4c8;margin-top:8px;line-height:1.6;">
            {explanation}
        </div>
    </div>
    """

# ============================================================
# GROQ PROMPT
# ============================================================
SYSTEM_PROMPT = """
You are the Communication Quality Evaluator for a product called
"Human Touch Quality Layer".

Evaluate a message from the recipient's perspective.

This is NOT an AI detector, authorship detector, grammar checker, or
sentiment classifier.

Core question:
"Would the recipient reasonably feel that this message was thoughtfully
written for their specific situation?"

Use only the supplied message and context. Never invent names, dates,
policies, discounts, delivery dates, feelings, commitments, background,
or other unsupported facts.

Score these five dimensions from 0 to 100:
1. Empathy
2. Naturalness
3. Personalization
4. Context Awareness
5. Brand Voice

High scores require evidence. Do not reward generic politeness or polished
wording by itself.

Also provide:
- overall_human_touch_score
- verdict
- summary
- recipient_perspective
- potential_concerns
- recommended_changes
- recommended_rewrite

Potential concern objects must contain:
issue, dimension, evidence, impact_on_recipient, severity

Severity must be one of:
Low, Medium, High

Recommended change objects must contain:
title, explanation

The rewrite must preserve the original meaning and supported factual claims.
Do not invent missing information.

Return VALID JSON ONLY with this exact structure:

{
  "overall_human_touch_score": 0,
  "verdict": "",
  "summary": "",
  "scores": {
    "empathy": 0,
    "naturalness": 0,
    "personalization": 0,
    "context_awareness": 0,
    "brand_voice": 0
  },
  "recipient_perspective": "",
  "potential_concerns": [
    {
      "issue": "",
      "dimension": "",
      "evidence": "",
      "impact_on_recipient": "",
      "severity": "Low"
    }
  ],
  "recommended_changes": [
    {
      "title": "",
      "explanation": ""
    }
  ],
  "recommended_rewrite": ""
}
""".strip()


def build_user_prompt(
    message: str,
    content_type: str,
    audience: str,
    purpose: str,
    brand_voice: str,
    additional_context: str,
) -> str:
    return f"""
Communication type:
{content_type or "[Not provided]"}

Message:
{message or "[Not provided]"}

Audience / Recipient:
{audience or "[Not provided]"}

Purpose:
{purpose or "[Not provided]"}

Brand Voice:
{brand_voice or "[Not provided]"}

Additional Context:
{additional_context or "[Not provided]"}

Evaluate only what is supported above.
""".strip()


def extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object was found in the model response.")

    depth = 0
    in_string = False
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed

    raise ValueError("The model response contained incomplete JSON.")


def normalize_result(data: Dict[str, Any]) -> Dict[str, Any]:
    scores = data.get("scores") or {}

    result = {
        "scores": {
            "empathy": clamp_score(scores.get("empathy")),
            "naturalness": clamp_score(scores.get("naturalness")),
            "personalization": clamp_score(scores.get("personalization")),
            "context_awareness": clamp_score(scores.get("context_awareness")),
            "brand_voice": clamp_score(scores.get("brand_voice")),
        },
        "verdict": str(data.get("verdict", "")).strip(),
        "summary": str(data.get("summary", "")).strip(),
        "recipient_perspective": str(
            data.get("recipient_perspective", "")
        ).strip(),
        "potential_concerns": data.get("potential_concerns") or [],
        "recommended_changes": data.get("recommended_changes") or [],
        "recommended_rewrite": str(
            data.get("recommended_rewrite", "")
        ).strip(),
    }

    result["overall_human_touch_score"] = round(
        sum(result["scores"].values()) / 5
    )

    if not result["verdict"]:
        score = result["overall_human_touch_score"]
        if score >= 90:
            result["verdict"] = "Excellent human touch"
        elif score >= 75:
            result["verdict"] = "Strong human touch"
        elif score >= 60:
            result["verdict"] = "Needs some refinement"
        else:
            result["verdict"] = "Needs significant refinement"

    return result


def evaluate_message(
    message: str,
    content_type: str,
    audience: str,
    purpose: str,
    brand_voice: str,
    additional_context: str,
) -> Dict[str, Any]:

    api_key = get_api_key()

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it in Streamlit → Settings → Secrets."
        )

    client = Groq(api_key=api_key)

    user_prompt = build_user_prompt(
        message=message,
        content_type=content_type,
        audience=audience,
        purpose=purpose,
        brand_voice=brand_voice,
        additional_context=additional_context,
    )

    response = client.chat.completions.create(
        model=get_model(),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=3000,
        response_format={"type": "json_object"},
    )

    if not response.choices:
        raise RuntimeError("Groq returned no choices.")

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError("Groq returned an empty response.")

    return normalize_result(extract_json(content))

# ============================================================
# HUMAN TOUCH — 3D SIGNAL LAB
# 3D product UI built inside Streamlit using CSS 3D + WebGL.
# ============================================================

st.set_page_config(
    page_title="Human Touch · 3D Signal Lab",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- Helpers ----------
def quality_color(score: int) -> str:
    score = clamp_score(score)
    if score >= 85:
        return "#B8FF72"
    if score >= 70:
        return "#A78BFA"
    if score >= 55:
        return "#FFD166"
    return "#FF6B86"


def metric_3d(label: str, value: Any, code: str) -> str:
    score = clamp_score(value)
    color = quality_color(score)
    return f"""
    <div class="metric-3d">
        <div class="metric-face">
            <div class="metric-code">{safe_text(code)}</div>
            <div class="metric-label">{safe_text(label)}</div>
            <div class="metric-score" style="color:{color};">{score}</div>
            <div class="metric-progress"><i style="width:{score}%;background:{color};"></i></div>
        </div>
        <div class="metric-depth"></div>
    </div>
    """


def concern_3d(item: Dict[str, Any], index: int) -> str:
    severity = safe_text(item.get("severity", "Low"))
    sev = severity.lower() if severity.lower() in {"low", "medium", "high"} else "low"
    return f"""
    <div class="issue-3d">
        <div class="issue-face">
            <div class="issue-index">0{index}</div>
            <div>
                <div class="issue-title">{safe_text(item.get("issue", "Potential concern"))}</div>
                <div class="issue-tag">{safe_text(item.get("dimension", ""))}</div>
                <div class="issue-copy"><b>Evidence</b>{safe_text(item.get("evidence", ""))}</div>
                <div class="issue-copy"><b>Recipient impact</b>{safe_text(item.get("impact_on_recipient", ""))}</div>
            </div>
            <span class="severity {sev}">{severity}</span>
        </div>
    </div>
    """


def improvement_3d(item: Dict[str, Any], index: int) -> str:
    return f"""
    <div class="improve-3d">
        <div class="improve-face">
            <div class="issue-index">0{index}</div>
            <div>
                <div class="issue-title">{safe_text(item.get("title", "Recommended improvement"))}</div>
                <div class="issue-copy">{safe_text(item.get("explanation", ""))}</div>
            </div>
            <div class="improve-arrow">↗</div>
        </div>
    </div>
    """


def results_3d(result: Dict[str, Any], original: str) -> str:
    overall = clamp_score(result.get("overall_human_touch_score", 0))
    color = quality_color(overall)
    circumference = 314.159
    dash = circumference * overall / 100

    metrics = "".join([
        metric_3d("Empathy", result["scores"]["empathy"], "EMP"),
        metric_3d("Naturalness", result["scores"]["naturalness"], "NAT"),
        metric_3d("Personalization", result["scores"]["personalization"], "PER"),
        metric_3d("Context awareness", result["scores"]["context_awareness"], "CTX"),
        metric_3d("Brand voice", result["scores"]["brand_voice"], "VOI"),
    ])

    concerns = result.get("potential_concerns", [])
    recs = result.get("recommended_changes", [])

    issues = "".join(concern_3d(x, i + 1) for i, x in enumerate(concerns))
    if not issues:
        issues = """
        <div class="empty-3d">
            <span>✓</span><div><b>Clean signal.</b><br>No major friction detected.</div>
        </div>
        """

    improvements = "".join(improvement_3d(x, i + 1) for i, x in enumerate(recs))
    if not improvements:
        improvements = """
        <div class="empty-3d">
            <span>✦</span><div><b>Ready to send.</b><br>No major intervention required.</div>
        </div>
        """

    return f"""
    <div class="results-3d">

        <div class="result-orbit-panel">
            <div class="result-copy">
                <div class="eyebrow-3d">HUMAN-TOUCH FIELD</div>
                <div class="result-heading">How will this land?</div>
                <div class="result-summary">{safe_text(result.get("summary", ""))}</div>
                <div class="verdict-3d" style="color:{color};border-color:{color}45;">
                    <i style="background:{color};"></i>
                    {safe_text(result.get("verdict", ""))}
                </div>
            </div>

            <div class="score-orb">
                <div class="orb-glow"></div>
                <div class="orb-ring">
                    <svg viewBox="0 0 120 120">
                        <circle class="orb-track" cx="60" cy="60" r="50"></circle>
                        <circle class="orb-value" cx="60" cy="60" r="50"
                            stroke="{color}"
                            stroke-dasharray="{dash:.2f} {circumference:.2f}"></circle>
                    </svg>
                </div>
                <div class="orb-core">
                    <strong>{overall}</strong>
                    <span>HUMAN TOUCH</span>
                </div>
            </div>

            <div class="orb-caption">
                <span>QUALITY FIELD</span>
                <b>0 — 100</b>
            </div>
        </div>

        <div class="metrics-3d">{metrics}</div>

        <div class="perspective-3d">
            <div class="perspective-depth"></div>
            <div class="perspective-face">
                <div class="perspective-icon">◌</div>
                <div>
                    <div class="eyebrow-3d">RECIPIENT LENS</div>
                    <div class="perspective-title">If I received this…</div>
                    <div class="perspective-text">{safe_text(result.get("recipient_perspective", ""))}</div>
                </div>
            </div>
        </div>

        <div class="analysis-3d">
            <div class="section-head-3d">
                <span>01</span>
                <div>
                    <div class="eyebrow-3d">FRICTION MAP</div>
                    <h3>Where the signal drops</h3>
                    <p>Evidence from the message, not generic AI-detector heuristics.</p>
                </div>
            </div>
            {issues}
        </div>

        <div class="analysis-3d">
            <div class="section-head-3d">
                <span>02</span>
                <div>
                    <div class="eyebrow-3d">IMPROVEMENT PLAN</div>
                    <h3>Where a small change matters</h3>
                    <p>Practical changes grounded in the supplied context.</p>
                </div>
            </div>
            {improvements}
        </div>

        <div class="analysis-3d">
            <div class="section-head-3d">
                <span>03</span>
                <div>
                    <div class="eyebrow-3d">REWRITE LAB</div>
                    <h3>Preserve meaning. Raise the signal.</h3>
                    <p>Supported claims stay supported; invented personal details are avoided.</p>
                </div>
            </div>

            <div class="rewrite-3d">
                <div class="message-3d">
                    <div class="message-head"><span>ORIGINAL</span><b>INPUT</b></div>
                    <div class="message-text">{safe_text(original).replace(chr(10), "<br>")}</div>
                </div>
                <div class="message-3d improved">
                    <div class="message-head"><span>RECOMMENDED</span><b>HUMANIZED</b></div>
                    <div class="message-text">{safe_text(result.get("recommended_rewrite", "")).replace(chr(10), "<br>")}</div>
                </div>
            </div>
        </div>
    </div>
    """


# ---------- WebGL hero ----------
# This is an intentionally small, lightweight Three.js scene. It is decorative
# and does not contain application state, so the Groq workflow remains reliable.
def render_3d_hero():
    components.html(
        """
        <html>
        <head>
        <style>
            html,body{margin:0;width:100%;height:100%;overflow:hidden;background:#07090d}
            #scene{width:100%;height:100%}
            .label{
                position:absolute;left:22px;bottom:18px;
                color:rgba(205,216,226,.48);
                font:700 8px/1.2 Arial,sans-serif;
                letter-spacing:.16em;text-transform:uppercase;
                pointer-events:none
            }
        </style>
        </head>
        <body>
        <div id="scene"></div>
        <div class="label">LIVE 3D HUMAN-TOUCH FIELD</div>
        <script src="https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.min.js"></script>
        <script>
        (() => {
          const host = document.getElementById("scene");
          const scene = new THREE.Scene();
          scene.background = new THREE.Color(0x07090d);

          const camera = new THREE.PerspectiveCamera(
            38, host.clientWidth / Math.max(host.clientHeight, 1), 0.1, 100
          );
          camera.position.set(0, 0.15, 5.8);

          const renderer = new THREE.WebGLRenderer({antialias:true, alpha:true});
          renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.7));
          renderer.setSize(host.clientWidth, host.clientHeight);
          renderer.outputColorSpace = THREE.SRGBColorSpace;
          host.appendChild(renderer.domElement);

          const group = new THREE.Group();
          scene.add(group);

          const core = new THREE.Mesh(
            new THREE.IcosahedronGeometry(1.12, 5),
            new THREE.MeshPhysicalMaterial({
              color: 0x8b5cf6,
              emissive: 0x35146e,
              emissiveIntensity: 1.7,
              roughness: 0.18,
              metalness: 0.42,
              transparent:true,
              opacity:.92
            })
          );
          group.add(core);

          const shell = new THREE.Mesh(
            new THREE.IcosahedronGeometry(1.42, 2),
            new THREE.MeshBasicMaterial({
              color:0x5ee7f5,
              wireframe:true,
              transparent:true,
              opacity:.18
            })
          );
          group.add(shell);

          const ringMaterial = new THREE.MeshBasicMaterial({
            color:0xb9f66d,
            transparent:true,
            opacity:.62
          });

          for(let i=0;i<3;i++){
            const ring = new THREE.Mesh(
              new THREE.TorusGeometry(1.72 + i*.17, .008, 8, 120),
              ringMaterial
            );
            ring.rotation.x = (i+1)*0.63;
            ring.rotation.y = (i+1)*0.47;
            group.add(ring);
          }

          const particles = new THREE.BufferGeometry();
          const count = 360;
          const positions = new Float32Array(count*3);
          for(let i=0;i<count;i++){
            const r = 2.2 + Math.random()*2.0;
            const a = Math.random()*Math.PI*2;
            const y = (Math.random()-.5)*2.6;
            positions[i*3] = Math.cos(a)*r;
            positions[i*3+1] = y;
            positions[i*3+2] = Math.sin(a)*r;
          }
          particles.setAttribute("position", new THREE.BufferAttribute(positions,3));
          const dots = new THREE.Points(
            particles,
            new THREE.PointsMaterial({
              color:0x8b9bb0,size:.018,transparent:true,opacity:.46
            })
          );
          scene.add(dots);

          scene.add(new THREE.AmbientLight(0x7282ff, 1.25));
          const key = new THREE.PointLight(0xb9f66d, 14, 12);
          key.position.set(2.4,1.8,3.3);
          scene.add(key);
          const fill = new THREE.PointLight(0x7c3aed, 16, 10);
          fill.position.set(-2.7,-1.3,2.4);
          scene.add(fill);
          const cyan = new THREE.PointLight(0x22d3ee, 9, 9);
          cyan.position.set(0,2.5,-1);
          scene.add(cyan);

          let tx=0, ty=0;
          host.addEventListener("pointermove", (e)=>{
            const r=host.getBoundingClientRect();
            tx=((e.clientX-r.left)/r.width-.5)*.55;
            ty=((e.clientY-r.top)/r.height-.5)*.35;
          });

          const clock = new THREE.Clock();
          function animate(){
            requestAnimationFrame(animate);
            const t=clock.getElapsedTime();

            core.rotation.x += .0022;
            core.rotation.y += .0033;
            shell.rotation.x -= .0014;
            shell.rotation.y += .0020;
            dots.rotation.y += .00028;

            group.rotation.y += (tx-group.rotation.y)*.018;
            group.rotation.x += (-ty-group.rotation.x)*.018;

            const pulse = 1 + Math.sin(t*1.5)*.025;
            core.scale.setScalar(pulse);

            renderer.render(scene,camera);
          }
          animate();

          const resize=()=>{
            const w=host.clientWidth, h=Math.max(host.clientHeight,1);
            camera.aspect=w/h;
            camera.updateProjectionMatrix();
            renderer.setSize(w,h);
          };
          new ResizeObserver(resize).observe(host);
        })();
        </script>
        </body>
        </html>
        """,
        height=330,
        scrolling=False,
    )


# ---------- CSS / 3D system ----------
st.html(r"""
<style>
:root{
 --bg:#07090D;--panel:#0C1017;--panel2:#101721;
 --line:rgba(255,255,255,.085);--line2:rgba(255,255,255,.13);
 --text:#F4F7F2;--muted:#86919E;--dim:#4D5864;
 --lime:#B8FF72;--violet:#A78BFA;--cyan:#5EE7F5;
}
html,body,[data-testid="stAppViewContainer"]{
 background:
 radial-gradient(700px 500px at 5% -8%,rgba(167,139,250,.10),transparent 65%),
 radial-gradient(700px 500px at 100% 8%,rgba(94,231,245,.055),transparent 66%),
 #07090D!important;color:var(--text)!important;
}
[data-testid="stHeader"]{background:transparent!important}
.block-container{max-width:1260px!important;padding:15px 28px 90px!important}

/* NAV */
.nav3d{
 display:flex;justify-content:space-between;align-items:center;
 height:42px;border-bottom:1px solid var(--line);
}
.brand3d{display:flex;align-items:center;gap:9px}
.logo3d{
 width:27px;height:27px;display:grid;place-items:center;
 border-radius:8px;color:#10150B;background:var(--lime);
 font-size:13px;font-weight:900;
 box-shadow:0 0 28px rgba(184,255,114,.17);
}
.brand3d-name{
 color:#E9EDE7;font-size:10px;font-weight:900;
 letter-spacing:.13em;text-transform:uppercase;
}
.brand3d-name span{color:#56616D;font-weight:600}
.live3d{
 color:#5D6875;font-size:7px;font-weight:800;
 letter-spacing:.13em;display:flex;align-items:center;gap:6px;
}
.live3d i{
 width:5px;height:5px;border-radius:50%;background:var(--lime);
 box-shadow:0 0 9px var(--lime);
}

/* HERO */
.hero3d{
 display:grid;grid-template-columns:1fr 420px;
 gap:42px;align-items:center;padding:65px 0 30px;
}
.kicker3d{
 color:var(--lime);font-size:8px;font-weight:900;letter-spacing:.18em;
}
.hero3d h1{
 margin:14px 0 0;max-width:780px;
 color:#F7F9F5;font-size:clamp(50px,6.8vw,88px);
 line-height:.90;font-weight:900;letter-spacing:-.075em;
}
.hero3d h1 span{color:#66717C}
.author3d{margin-top:18px;color:#626D79;font-size:10px}
.hero3d-side{
 color:#7F8A97;font-size:11px;line-height:1.8;
}
.hero3d-side strong{color:#C8D0D7}
.hero3d-side-rule{
 margin-top:14px;padding-top:11px;border-top:1px solid var(--line);
 color:#4F5965;font-size:7px;letter-spacing:.05em;
}

/* 3D HERO FRAME */
.hero3d-canvas{
 margin-top:8px;
 border:1px solid rgba(255,255,255,.09);
 border-radius:22px;
 overflow:hidden;
 background:#07090D;
 box-shadow:
   0 30px 90px rgba(0,0,0,.42),
   inset 0 0 60px rgba(94,231,245,.025);
}

/* STEPS */
.steps3d{
 display:grid;grid-template-columns:repeat(3,1fr);
 border-top:1px solid var(--line);border-bottom:1px solid var(--line);
}
.step3d{padding:15px 17px;border-right:1px solid var(--line)}
.step3d:last-child{border-right:0}
.step3d-code{color:#505A66;font-size:7px;font-weight:900;letter-spacing:.14em}
.step3d-title{margin-top:6px;color:#DCE2DC;font-size:10px;font-weight:730}
.step3d-copy{margin-top:3px;color:#626D79;font-size:8px}

/* WORKSPACE */
.workspace-label3d{
 margin:40px 0 11px;color:#525D69;font-size:8px;font-weight:900;
 letter-spacing:.18em;
}
.panel3d{
 border:1px solid var(--line);border-radius:19px;padding:21px;
 background:linear-gradient(145deg,rgba(255,255,255,.03),rgba(255,255,255,.008)),var(--panel);
 box-shadow:0 18px 55px rgba(0,0,0,.12);
}
.panel3d-label{color:#82909D;font-size:8px;font-weight:900;letter-spacing:.15em}
.panel3d-title{margin-top:6px;color:#EEF2ED;font-size:18px;font-weight:770;letter-spacing:-.025em}
.panel3d-copy{margin-top:4px;color:#66717D;font-size:9px;line-height:1.6}

/* CONTROLS */
[data-baseweb="textarea"]>div,[data-baseweb="input"]>div,[data-baseweb="select"]>div{
 background:#080B10!important;border:1px solid rgba(255,255,255,.085)!important;
 border-radius:12px!important;
}
[data-baseweb="textarea"]>div:focus-within,[data-baseweb="input"]>div:focus-within,[data-baseweb="select"]>div:focus-within{
 border-color:rgba(184,255,114,.32)!important;
 box-shadow:0 0 0 3px rgba(184,255,114,.035)!important;
}
textarea,input{color:#F3F6F1!important}
textarea::placeholder,input::placeholder{color:#47515D!important}
[data-baseweb="select"] *{color:#F3F6F1!important}
label,[data-testid="stWidgetLabel"] p{
 color:#7C8793!important;font-size:9px!important;font-weight:700!important;
}
[data-testid="stRadio"] label{color:#77828F!important}
[data-testid="stRadio"] [role="radiogroup"]{gap:3px}
.count3d{margin:-4px 0 0;color:#46515D;font-size:8px}
.count3d b{color:#AAB4BF}
.stButton>button{
 min-height:58px!important;border-radius:13px!important;
 border:1px solid rgba(184,255,114,.27)!important;
 background:linear-gradient(100deg,#A8E85C,#C5F77B,#A8E85C)!important;
 color:#10140C!important;font-size:12px!important;font-weight:900!important;
 box-shadow:0 18px 50px rgba(184,255,114,.10)!important;
}
.stButton>button:hover{
 filter:brightness(1.04);transform:translateY(-1px);
 box-shadow:0 23px 60px rgba(184,255,114,.15)!important;
}
.disclaimer3d{text-align:center;margin-top:7px;color:#47515C;font-size:8px}

/* RESULTS */
.result-head3d{
 margin:56px 0 20px;padding-top:23px;border-top:1px solid var(--line);
 display:flex;justify-content:space-between;align-items:end;
}
.result-head3d h2{
 margin:6px 0 0;color:#EEF2ED;font-size:31px;font-weight:850;
 letter-spacing:-.05em;
}
.result-side3d{color:#4D5864;font-size:7px;font-weight:900;letter-spacing:.14em}
.result-orbit-panel{
 position:relative;display:grid;grid-template-columns:1fr 250px;
 gap:25px;align-items:center;
 min-height:285px;padding:28px;
 border:1px solid var(--line2);border-radius:22px;
 background:
 radial-gradient(circle at 80% 50%,rgba(167,139,250,.06),transparent 31%),
 radial-gradient(circle at 10% 100%,rgba(184,255,114,.025),transparent 35%),
 #0B0F15;
 box-shadow:0 30px 90px rgba(0,0,0,.25);
 overflow:hidden;
}
.result-orbit-panel:before{
 content:"";position:absolute;inset:10px;border:1px solid rgba(255,255,255,.025);
 border-radius:17px;pointer-events:none;
}
.eyebrow-3d{color:#687480;font-size:7px;font-weight:900;letter-spacing:.17em}
.result-heading{margin-top:8px;color:#F3F6F1;font-size:34px;font-weight:850;letter-spacing:-.05em}
.result-summary{max-width:650px;margin-top:12px;color:#7D8996;font-size:11px;line-height:1.8}
.verdict-3d{
 display:inline-flex;align-items:center;gap:7px;margin-top:16px;
 padding:7px 10px;border:1px solid;border-radius:999px;
 background:rgba(255,255,255,.018);font-size:8px;font-weight:800;
}
.verdict-3d i{width:5px;height:5px;border-radius:50%;box-shadow:0 0 9px currentColor}
.score-caption{
 margin-top:20px;padding-top:11px;border-top:1px solid var(--line);
 color:#4F5965;font-size:7px;letter-spacing:.04em;
}

/* 3D SCORE ORB */
.score-orb{
 width:220px;height:220px;position:relative;margin:auto;
 perspective:700px;transform-style:preserve-3d;
}
.orb-glow{
 position:absolute;inset:32px;border-radius:50%;
 background:radial-gradient(circle,rgba(167,139,250,.23),rgba(94,231,245,.06),transparent 68%);
 filter:blur(10px);
 animation:orbPulse 4s ease-in-out infinite;
}
@keyframes orbPulse{0%,100%{transform:scale(.92);opacity:.72}50%{transform:scale(1.08);opacity:1}}
.orb-ring{position:absolute;inset:0;animation:orbitSpin 14s linear infinite}
.orb-ring:after{
 content:"";position:absolute;inset:21px;border-radius:50%;
 border:1px solid rgba(94,231,245,.12);
 box-shadow:inset 0 0 30px rgba(167,139,250,.05);
}
@keyframes orbitSpin{to{transform:rotate(360deg)}}
.orb-ring svg{width:220px;height:220px;transform:rotate(-90deg)}
.orb-track{fill:none;stroke:rgba(255,255,255,.06);stroke-width:4}
.orb-value{fill:none;stroke-width:4;stroke-linecap:round;filter:drop-shadow(0 0 5px currentColor)}
.orb-core{
 position:absolute;inset:0;display:flex;flex-direction:column;
 justify-content:center;align-items:center;z-index:3;
}
.orb-core strong{
 color:#F5F8F3;font-size:54px;font-weight:900;
 letter-spacing:-.08em;line-height:1;
 text-shadow:0 0 25px rgba(255,255,255,.08);
}
.orb-core span{margin-top:7px;color:#596570;font-size:7px;font-weight:900;letter-spacing:.15em}
.orb-caption{
 position:absolute;right:28px;bottom:20px;
 display:flex;gap:7px;color:#4E5965;font-size:7px;font-weight:800;letter-spacing:.10em;
}
.orb-caption b{color:#75808C}

/* METRICS */
.metrics-3d{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-top:8px}
.metric-3d{position:relative;height:122px;perspective:700px}
.metric-face{
 position:absolute;inset:0;padding:15px;border:1px solid var(--line);
 border-radius:14px;background:#0C1016;
 transform:translateZ(12px);
 box-shadow:0 10px 30px rgba(0,0,0,.17);
}
.metric-depth{
 position:absolute;inset:5px;border-radius:14px;
 border:1px solid rgba(255,255,255,.025);
 background:#080B10;transform:translateZ(-4px);
}
.metric-code{color:#4E5965;font-size:7px;font-weight:900;letter-spacing:.11em}
.metric-label{margin-top:9px;color:#7F8A96;font-size:9px}
.metric-score{margin-top:7px;font-size:26px;font-weight:880;letter-spacing:-.05em}
.metric-progress{height:3px;margin-top:11px;border-radius:99px;background:rgba(255,255,255,.05);overflow:hidden}
.metric-progress i{display:block;height:100%;border-radius:99px}

/* PERSPECTIVE */
.perspective-3d{position:relative;margin-top:10px;padding:0 0 5px;perspective:900px}
.perspective-depth{
 position:absolute;inset:5px 0 0;border:1px solid rgba(167,139,250,.07);
 border-radius:16px;background:#080B10;transform:translateZ(-10px);
}
.perspective-face{
 position:relative;display:grid;grid-template-columns:45px 1fr;gap:13px;
 padding:24px 27px;border:1px solid rgba(167,139,250,.14);
 border-radius:16px;background:linear-gradient(135deg,rgba(167,139,250,.06),rgba(94,231,245,.018)),#0D1118;
 transform:translateZ(9px);box-shadow:0 20px 45px rgba(0,0,0,.20);
}
.perspective-icon{
 width:35px;height:35px;display:grid;place-items:center;border-radius:10px;
 color:#B9A9FF;background:rgba(167,139,250,.08);font-size:18px;
}
.perspective-title{margin-top:5px;color:#E9EDE8;font-size:14px;font-weight:760}
.perspective-text{max-width:900px;margin-top:8px;color:#98A3AE;font-size:10px;line-height:1.8}

/* ANALYSIS */
.analysis-3d{margin-top:45px}
.section-head-3d{display:grid;grid-template-columns:34px 1fr;gap:11px;margin-bottom:14px}
.section-head-3d>span{color:#414B56;font-size:8px;font-weight:900}
.section-head-3d h3{margin:6px 0 0;color:#E8ECE7;font-size:21px;font-weight:800;letter-spacing:-.035em}
.section-head-3d p{margin:4px 0 0;color:#626D79;font-size:8px}

/* 3D ROWS */
.issue-3d,.improve-3d{perspective:900px;margin:7px 0}
.issue-face,.improve-face{
 position:relative;display:grid;grid-template-columns:35px 1fr auto;gap:13px;
 padding:18px;border:1px solid var(--line);border-radius:14px;
 background:#0C1016;transform:translateZ(7px);
 box-shadow:0 8px 25px rgba(0,0,0,.12);
}
.issue-3d:after,.improve-3d:after{
 content:"";display:block;height:4px;margin:-3px 6px 0;
 border:1px solid rgba(255,255,255,.025);border-top:0;
 border-radius:0 0 12px 12px;background:#080B10;
}
.issue-index{color:#4B5662;font-size:8px;font-weight:900;letter-spacing:.10em}
.issue-title{color:#DDE3DE;font-size:10px;font-weight:750}
.issue-tag{margin-top:4px;color:#626D79;font-size:7px;text-transform:uppercase;letter-spacing:.07em}
.issue-copy{margin-top:10px;color:#76818D;font-size:8px;line-height:1.65}
.issue-copy b{display:block;margin-bottom:2px;color:#9DA7B0;font-size:7px;text-transform:uppercase;letter-spacing:.08em}
.severity{
 align-self:start;padding:4px 7px;border-radius:99px;
 font-size:6px;font-weight:900;text-transform:uppercase;
}
.severity.low{color:#69DDA7;background:rgba(105,221,167,.055)}
.severity.medium{color:#F2C96A;background:rgba(242,201,106,.055)}
.severity.high{color:#FF7F97;background:rgba(255,127,151,.055)}
.improve-face{align-items:start}
.improve-arrow{color:#6C7884;font-size:14px}

/* REWRITE */
.rewrite-3d{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.message-3d{
 min-height:225px;padding:19px;border:1px solid var(--line);
 border-radius:15px;background:#090C11;
 box-shadow:0 12px 30px rgba(0,0,0,.12);
}
.message-3d.improved{
 border-color:rgba(184,255,114,.14);
 background:radial-gradient(circle at 100% 0%,rgba(184,255,114,.045),transparent 45%),#090C11;
}
.message-head{display:flex;justify-content:space-between;color:#58636F;font-size:7px;font-weight:900;letter-spacing:.13em}
.message-head b{color:#3F4955}
.message-text{margin-top:12px;color:#AAB4BE;font-size:9px;line-height:1.85}
.empty-3d{
 display:flex;align-items:center;gap:10px;padding:17px;
 border:1px dashed rgba(255,255,255,.09);border-radius:13px;
 color:#697580;font-size:9px;line-height:1.6;
}
.empty-3d span{color:var(--lime);font-size:15px}
.empty-3d b{color:#AAB4BE}
.footer3d{
 display:flex;justify-content:space-between;gap:20px;
 margin-top:60px;padding-top:17px;border-top:1px solid var(--line);
 color:#414B57;font-size:7px;letter-spacing:.08em;
}

/* RESPONSIVE */
@media(max-width:950px){
 .hero3d{grid-template-columns:1fr;gap:28px}
 .workspace{grid-template-columns:1fr!important}
 .result-orbit-panel{grid-template-columns:1fr}
 .score-orb{order:-1}
 .metrics-3d{grid-template-columns:repeat(2,1fr)}
}
@media(max-width:650px){
 .block-container{padding:11px 12px 55px!important}
 .hero3d{padding:52px 0 30px}
 .hero3d h1{font-size:53px}
 .steps3d{grid-template-columns:1fr}
 .step3d{border-right:0;border-bottom:1px solid var(--line)}
 .step3d:last-child{border-bottom:0}
 .metrics-3d{grid-template-columns:1fr}
 .rewrite-3d{grid-template-columns:1fr}
 .issue-face,.improve-face{grid-template-columns:28px 1fr}
 .issue-face .severity{grid-column:2;justify-self:start}
 .footer3d{flex-direction:column}
}
</style>
""")

# ---------- Navigation ----------
st.html("""
<div class="nav3d">
    <div class="brand3d">
        <div class="logo3d">✦</div>
        <div class="brand3d-name">Human Touch <span>/ 3D Signal Lab</span></div>
    </div>
    <div class="live3d"><i></i> LIVE QUALITY LAYER</div>
</div>
""")

# ---------- Hero copy ----------
st.html("""
<div class="hero3d">
    <div>
        <div class="kicker3d">Communication intelligence · spatial interface</div>
        <h1>Before you send it,<br><span>feel it.</span></h1>
        <div class="author3d">By Engr. Muhammad Mubashir Asim</div>
    </div>
    <div class="hero3d-side">
        <strong>Human Touch</strong> turns communication quality into a visual
        signal — helping you see whether another person would feel understood,
        respected, and thoughtfully addressed.
        <div class="hero3d-side-rule">
            3D interface · recipient perspective · evidence-based quality control
        </div>
    </div>
</div>
""")

# ---------- 3D WebGL object ----------
st.html('<div class="hero3d-canvas">')
render_3d_hero()
st.html('</div>')

# ---------- Workflow ----------
st.html("""
<div class="steps3d" style="margin-top:10px;">
    <div class="step3d">
        <div class="step3d-code">01 · MESSAGE</div>
        <div class="step3d-title">Bring the communication</div>
        <div class="step3d-copy">Paste the exact message.</div>
    </div>
    <div class="step3d">
        <div class="step3d-code">02 · CONTEXT</div>
        <div class="step3d-title">Frame the situation</div>
        <div class="step3d-copy">Recipient, purpose, voice, context.</div>
    </div>
    <div class="step3d">
        <div class="step3d-code">03 · SIGNAL</div>
        <div class="step3d-title">See the human layer</div>
        <div class="step3d-copy">Score, friction, improvements, rewrite.</div>
    </div>
</div>
""")

# ---------- Workspace ----------
st.html('<div class="workspace-label3d">COMPOSE / CONTEXT</div>')

left, right = st.columns([1.25, .75], gap="small")

with left:
    st.html("""
    <div class="panel3d">
        <div class="panel3d-label">01 · MESSAGE</div>
        <div class="panel3d-title">What are you sending?</div>
        <div class="panel3d-copy">Use the exact wording the recipient will see.</div>
    </div>
    """)
    content_type = st.radio(
        "Communication type",
        ["Email", "Customer Service", "Business", "Social Post", "Other"],
        index=0,
        horizontal=True,
    )
    message = st.text_area(
        "Message",
        placeholder="Write or paste the message here…",
        height=245,
        label_visibility="collapsed",
    )
    st.html(
        f'<div class="count3d"><b>{len(message or "")}</b> characters · '
        'evaluated for communication quality</div>'
    )

with right:
    st.html("""
    <div class="panel3d">
        <div class="panel3d-label">02 · CONTEXT</div>
        <div class="panel3d-title">Who is receiving it?</div>
        <div class="panel3d-copy">Context helps the quality layer reason about the situation.</div>
    </div>
    """)
    audience = st.text_input("Audience / Recipient", placeholder="Existing customer")
    purpose = st.text_input("Purpose", placeholder="Confirmation")
    brand_voice = st.text_input("Brand Voice", placeholder="Warm, professional")
    additional_context = st.text_input("Additional Context", placeholder="Relevant situation…")

st.write("")
analyze = st.button("✦  Analyze Human Touch  →", type="primary", use_container_width=True)

st.html("""
<div class="disclaimer3d">
    Communication-quality analysis · not a definitive AI detector
</div>
""")

# ---------- Evaluation ----------
if analyze:
    if not message.strip():
        st.error("Please paste a message before analyzing.")
        st.session_state.result = None
    else:
        with st.spinner("Reading the message through the recipient's eyes…"):
            try:
                result = evaluate_message(
                    message=message.strip(),
                    content_type=content_type,
                    audience=audience.strip(),
                    purpose=purpose.strip(),
                    brand_voice=brand_voice.strip(),
                    additional_context=additional_context.strip(),
                )
                st.session_state.result = result
                st.session_state.analyzed_message = message.strip()
            except Exception as exc:
                st.session_state.result = None
                error_text = str(exc)
                lower = error_text.lower()

                if any(token in lower for token in ["401", "authentication", "api key", "unauthorized"]):
                    st.error("Groq authentication failed. Check GROQ_API_KEY in Streamlit Secrets.")
                elif "rate" in lower and "limit" in lower:
                    st.error("Groq rate limit reached. Please wait a moment and try again.")
                elif "model" in lower and ("not found" in lower or "does not exist" in lower):
                    st.error(
                        f"Groq model error. Current model: {get_model()}. "
                        "Set GROQ_MODEL to a currently supported Groq model in Streamlit Secrets."
                    )
                else:
                    st.error(f"Evaluation failed: {error_text}")

# ---------- Results ----------
if "result" not in st.session_state:
    st.session_state.result = None
if "analyzed_message" not in st.session_state:
    st.session_state.analyzed_message = ""

if st.session_state.result:
    st.html("""
    <div class="result-head3d">
        <div>
            <div class="eyebrow-3d">03 · SIGNAL REPORT</div>
            <h2>Your human-touch readout</h2>
        </div>
        <div class="result-side3d">RECIPIENT-FIRST ANALYSIS</div>
    </div>
    """)
    st.html(results_3d(st.session_state.result, st.session_state.analyzed_message))

st.html("""
<div class="footer3d">
    <div>HUMAN TOUCH · 3D SIGNAL LAB</div>
    <div>Communication quality · Recipient perspective · Human connection</div>
</div>
""")
