import html
import json
import os
from typing import Any, Dict

import streamlit as st
from groq import Groq

# ============================================================
# PAGE
# ============================================================
st.set_page_config(
    page_title="Human Touch — Quality Layer",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
                parsed = json.loads(text[start : i + 1])
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
        "recipient_perspective": str(data.get("recipient_perspective", "")).strip(),
        "potential_concerns": data.get("potential_concerns") or [],
        "recommended_changes": data.get("recommended_changes") or [],
        "recommended_rewrite": str(data.get("recommended_rewrite", "")).strip(),
    }
    result["overall_human_touch_score"] = round(sum(result["scores"].values()) / 5)

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


# ============================================================
# GROQ ENGINE — KEEPING THE PRODUCT LOGIC INTACT
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


def build_user_prompt(message, content_type, audience, purpose, brand_voice, additional_context):
    return f"""
Communication type:
{content_type or '[Not provided]'}

Message:
{message or '[Not provided]'}

Audience / Recipient:
{audience or '[Not provided]'}

Purpose:
{purpose or '[Not provided]'}

Brand Voice:
{brand_voice or '[Not provided]'}

Additional Context:
{additional_context or '[Not provided]'}

Evaluate only what is supported above.
""".strip()


def evaluate_message(message, content_type, audience, purpose, brand_voice, additional_context):
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it in Streamlit → Settings → Secrets.")

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=get_model(),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_prompt(
                    message, content_type, audience, purpose, brand_voice, additional_context
                ),
            },
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
# VISUAL SYSTEM
# ============================================================
st.html("""
<style>
:root {
  --bg: #0b0d12;
  --surface: #11141b;
  --surface-2: #151923;
  --border: #252b36;
  --muted: #8991a1;
  --text: #f5f7fb;
  --accent: #8b7cff;
  --accent-2: #6d5ce7;
  --green: #38d39f;
  --red: #ff6b7a;
  --amber: #f2bf66;
}

#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stAppViewContainer"] { background: var(--bg); }
[data-testid="stSidebar"] {
  background: #0f1218;
  border-right: 1px solid var(--border);
}
[data-testid="stSidebarContent"] { padding: 24px 20px 30px; }
.block-container { padding-top: 0.8rem; padding-bottom: 4rem; max-width: 1180px; }

/* Native Streamlit controls */
.stTextArea textarea, .stTextInput input, .stSelectbox [data-baseweb="select"] > div {
  background: #0f1218 !important;
  border-color: #2a303c !important;
  color: #f5f7fb !important;
  border-radius: 10px !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
  border-color: #6f63d9 !important;
  box-shadow: 0 0 0 1px #6f63d9 !important;
}
.stButton > button[kind="primary"] {
  border: 0 !important;
  border-radius: 10px !important;
  min-height: 48px !important;
  font-weight: 750 !important;
  letter-spacing: .1px;
  background: linear-gradient(135deg, #8b7cff, #6d5ce7) !important;
}
.stButton > button[kind="primary"]:hover { filter: brightness(1.08); }

.brand { padding: 4px 2px 0; }
.brand-mark {
  width: 38px; height: 38px; border-radius: 11px;
  display: inline-flex; align-items:center; justify-content:center;
  background: linear-gradient(135deg,#9b8cff,#6253db);
  color:white; font-weight:900; font-size:19px;
  box-shadow: 0 8px 25px rgba(120,100,230,.25);
}
.brand-name { font-size: 18px; font-weight: 800; margin-left: 9px; vertical-align: 10px; color:#f5f7fb; }
.brand-sub { color:#858d9e; font-size:12px; margin-top:7px; }
.sidebar-note { color:#6f7787; font-size:11px; line-height:1.55; margin-top:22px; }

.hero { padding: 52px 0 30px; }
.eyebrow {
  display:inline-flex; align-items:center; gap:8px; color:#a9a1ff; font-size:11px;
  font-weight:800; letter-spacing:1.5px; text-transform:uppercase;
  background:#17152b; border:1px solid #302b55; padding:7px 10px; border-radius:999px;
}
.hero h1 { font-size: clamp(38px,5vw,64px); line-height: .98; letter-spacing:-2.8px; margin:18px 0 13px; color:#fff; }
.hero h1 span { color:#9185ff; }
.hero p { max-width:690px; color:#9aa2b2; font-size:17px; line-height:1.7; margin:0; }
.author { color:#697283; font-size:12px; margin-top:18px; }

.workflow { display:flex; gap:9px; margin:25px 0 8px; flex-wrap:wrap; }
.step { display:flex; align-items:center; gap:8px; color:#8991a1; font-size:12px; }
.step b { width:25px; height:25px; border-radius:8px; display:flex; align-items:center; justify-content:center; background:#171b24; border:1px solid #2a303c; color:#cbd0da; font-size:10px; }
.step.active b { background:#211e42; color:#a69cff; border-color:#423b73; }
.arrow { color:#4c5361; }

.card { background:var(--surface); border:1px solid var(--border); border-radius:16px; padding:22px; }
.card-head { display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom:16px; }
.card-title { color:#f2f4f8; font-weight:780; font-size:15px; }
.card-note { color:#687183; font-size:11px; }

.counter { color:#6f7889; font-size:11px; text-align:right; margin-top:-6px; margin-bottom:12px; }
.hint { color:#697282; font-size:12px; line-height:1.6; margin-top:9px; }

.result-top { margin-top:30px; padding-top:30px; border-top:1px solid #202631; }
.score-wrap { display:flex; align-items:center; gap:25px; }
.score-ring {
  width:145px; height:145px; border-radius:50%; display:flex; align-items:center; justify-content:center;
  background: conic-gradient(#8b7cff var(--score), #252a34 0);
  position:relative; flex:0 0 auto;
}
.score-ring:after { content:""; position:absolute; inset:9px; background:#11141b; border-radius:50%; }
.score-inner { position:relative; z-index:1; text-align:center; }
.score-num { font-size:40px; line-height:1; font-weight:850; color:#fff; letter-spacing:-2px; }
.score-den { color:#727b8b; font-size:11px; }
.score-label { color:#727b8b; font-size:10px; letter-spacing:1.4px; font-weight:800; margin-top:6px; }
.verdict { font-size:23px; font-weight:800; color:#fff; margin-bottom:8px; }
.summary { color:#9ca4b3; line-height:1.7; font-size:14px; max-width:690px; }

.section { margin-top:34px; }
.section-title { font-size:19px; font-weight:800; color:#f5f7fb; letter-spacing:-.3px; margin-bottom:13px; }
.section-sub { color:#687182; font-size:12px; margin-top:-8px; margin-bottom:15px; }
.metric-grid { display:grid; grid-template-columns:repeat(5,1fr); gap:10px; }
.metric { background:#11141b; border:1px solid #242a35; border-radius:13px; padding:16px; }
.metric-name { color:#8d96a6; font-size:11px; min-height:29px; }
.metric-value { color:#f5f7fb; font-size:24px; font-weight:850; margin:7px 0 9px; }
.track { height:4px; border-radius:8px; background:#272d37; overflow:hidden; }
.fill { height:100%; border-radius:8px; background:linear-gradient(90deg,#7769ec,#a095ff); }

.quote { background:#151923; border-left:3px solid #7c70ee; border-radius:0 12px 12px 0; padding:18px 20px; color:#c1c7d2; line-height:1.75; font-size:14px; }
.concern { padding:16px 0; border-bottom:1px solid #252b35; }
.concern:last-child { border-bottom:0; }
.concern-title { color:#f2f4f8; font-weight:750; font-size:14px; }
.concern-meta { color:#6f7888; font-size:11px; margin-top:5px; }
.concern-body { color:#a2a9b7; font-size:12px; line-height:1.65; margin-top:8px; }
.badge { display:inline-block; border-radius:999px; padding:4px 8px; font-size:9px; font-weight:800; letter-spacing:.7px; text-transform:uppercase; margin-left:7px; }
.badge-low { color:#77dcb8; background:#10251f; }
.badge-medium { color:#f0c36f; background:#292113; }
.badge-high { color:#ff8791; background:#2c171b; }

.change { display:flex; gap:13px; padding:15px 0; border-bottom:1px solid #252b35; }
.change:last-child { border-bottom:0; }
.change-num { color:#9186ff; font-size:11px; font-weight:850; padding-top:2px; }
.change-title { color:#f2f4f8; font-weight:750; font-size:13px; }
.change-text { color:#929baa; font-size:12px; line-height:1.65; margin-top:4px; }
.rewrite { background:#0e1117; border:1px solid #2b313c; border-radius:13px; padding:19px; color:#d7dbe4; line-height:1.8; font-size:14px; white-space:pre-wrap; }
.copy-note { color:#667081; font-size:11px; margin-top:9px; }

.empty { text-align:center; padding:30px 15px; color:#737c8d; border:1px dashed #2a303a; border-radius:14px; }
.footer { color:#4f5868; text-align:center; font-size:11px; padding:45px 0 8px; }

@media (max-width: 900px) {
  .metric-grid { grid-template-columns:repeat(2,1fr); }
  .hero { padding-top:30px; }
}
@media (max-width: 600px) {
  .block-container { padding-left:14px; padding-right:14px; }
  .metric-grid { grid-template-columns:1fr 1fr; }
  .score-wrap { align-items:flex-start; flex-direction:column; }
  .hero h1 { letter-spacing:-1.8px; }
}
</style>
""")

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.html("""
    <div class="brand">
      <span class="brand-mark">✦</span><span class="brand-name">Human Touch</span>
      <div class="brand-sub">Quality Layer</div>
    </div>
    <div class="sidebar-note">Communication quality control for messages that need to feel considered, not generated.</div>
    """)

    st.markdown("### Evaluation")
    st.selectbox("AI Model", [get_model()], disabled=True, label_visibility="visible")

    st.markdown("### Context")
    content_type = st.selectbox(
        "Content type",
        ["Email", "Customer-service response", "Business message", "Social-media post", "Other"],
    )
    audience = st.text_input("Recipient", placeholder="Customer, client, colleague…")
    purpose = st.text_input("Purpose", placeholder="What should this achieve?")
    brand_voice = st.text_input("Brand voice", placeholder="Warm, professional, direct…")
    additional_context = st.text_area(
        "Extra context",
        placeholder="Anything important the evaluator should know…",
        height=90,
    )

    st.html("<div class='sidebar-note'>Tip: context makes the evaluation more useful. You can leave optional fields empty.</div>")

# ============================================================
# HERO
# ============================================================
st.html("""
<div class="hero">
  <div class="eyebrow">● Human intelligence layer active</div>
  <h1>Make every message feel <span>human.</span></h1>
  <p>Paste a message. Give us a little context. Get a clear quality score and practical edits before you hit send.</p>
  <div class="author">By Engr. Muhammad Mubashir Asim</div>
  <div class="workflow">
    <div class="step active"><b>01</b> Message</div><div class="arrow">→</div>
    <div class="step"><b>02</b> Context</div><div class="arrow">→</div>
    <div class="step"><b>03</b> Human-touch review</div>
  </div>
</div>
""")

# ============================================================
# INPUT
# ============================================================
st.html("""
<div class="card">
  <div class="card-head">
    <div class="card-title">Your message</div>
    <div class="card-note">Private evaluation • no AI detection</div>
  </div>
</div>
""")

sample = st.button("Try a sample message", use_container_width=False)
if sample:
    st.session_state["message_input"] = (
        "Hi Sarah, just checking in on the proposal. Let me know if you have any questions. Thanks!"
    )

message = st.text_area(
    "Message",
    value=st.session_state.get("message_input", ""),
    key="message_input",
    placeholder="Paste your email, customer reply, business message, social post, or any communication here…",
    height=230,
    label_visibility="collapsed",
)

word_count = len(message.split()) if message.strip() else 0
st.html(f'<div class="counter">{word_count} words · Add context on the left for a sharper review.</div>')

col_a, col_b = st.columns([5, 1.4])
with col_a:
    analyze = st.button("✦  Analyze Human Touch", type="primary", use_container_width=True)
with col_b:
    clear = st.button("Clear", use_container_width=True)

if clear:
    st.session_state.pop("result", None)
    st.session_state["message_input"] = ""
    st.rerun()

if analyze:
    if not message.strip():
        st.warning("Paste a message first — then we can evaluate it.")
    else:
        with st.spinner("Reading the message from the recipient's perspective…"):
            try:
                st.session_state["result"] = evaluate_message(
                    message,
                    content_type,
                    audience,
                    purpose,
                    brand_voice,
                    additional_context,
                )
                st.session_state["original"] = message
            except Exception as exc:
                st.error(f"We could not complete the evaluation: {exc}")

# ============================================================
# RESULTS
# ============================================================
result = st.session_state.get("result")
if result:
    overall = clamp_score(result.get("overall_human_touch_score"))
    score_css = f"{overall}%"

    st.html(f"""
    <div class="result-top">
      <div class="score-wrap">
        <div class="score-ring" style="--score:{score_css};">
          <div class="score-inner">
            <div class="score-num">{overall}</div>
            <div class="score-den">/ 100</div>
            <div class="score-label">HUMAN TOUCH</div>
          </div>
        </div>
        <div>
          <div class="verdict">{safe_text(result.get('verdict'))}</div>
          <div class="summary">{safe_text(result.get('summary'))}</div>
        </div>
      </div>
    </div>
    """)

    st.html('<div class="section"><div class="section-title">Quality dimensions</div><div class="section-sub">Five signals that shape how the message may land with a real recipient.</div></div>')

    dimensions = [
        ("Empathy", "empathy"),
        ("Naturalness", "naturalness"),
        ("Personalization", "personalization"),
        ("Context awareness", "context_awareness"),
        ("Brand voice", "brand_voice"),
    ]
    cards = []
    for label, key in dimensions:
        value = clamp_score(result["scores"].get(key))
        cards.append(
            f'''<div class="metric"><div class="metric-name">{label}</div><div class="metric-value">{value}</div><div class="track"><div class="fill" style="width:{value}%"></div></div></div>'''
        )
    st.html('<div class="metric-grid">' + ''.join(cards) + '</div>')

    st.html('<div class="section"><div class="section-title">Recipient perspective</div><div class="section-sub">What the other person may reasonably take away from the message.</div></div>')
    st.html(f'<div class="quote">{safe_text(result.get("recipient_perspective"))}</div>')

    concerns = result.get("potential_concerns") or []
    st.html('<div class="section"><div class="section-title">Potential concerns</div></div>')
    if not concerns:
        st.success("No major concerns identified. The message is in good shape.")
    else:
        concern_html = []
        for item in concerns:
            severity = str(item.get("severity", "Low")).strip().lower()
            badge_class = {"high": "badge-high", "medium": "badge-medium", "low": "badge-low"}.get(severity, "badge-low")
            concern_html.append(
                f'''<div class="concern">
                  <div class="concern-title">{safe_text(item.get("issue", "Potential concern"))}<span class="badge {badge_class}">{safe_text(item.get("severity", "Low"))}</span></div>
                  <div class="concern-meta">{safe_text(item.get("dimension"))}</div>
                  <div class="concern-body"><b>Evidence:</b> {safe_text(item.get("evidence"))}<br><b>Recipient impact:</b> {safe_text(item.get("impact_on_recipient"))}</div>
                </div>'''
            )
        st.html('<div class="card">' + ''.join(concern_html) + '</div>')

    changes = result.get("recommended_changes") or []
    st.html('<div class="section"><div class="section-title">Recommended changes</div></div>')
    if not changes:
        st.info("No major changes needed.")
    else:
        change_html = []
        for i, item in enumerate(changes, 1):
            change_html.append(
                f'''<div class="change"><div class="change-num">0{i}</div><div><div class="change-title">{safe_text(item.get("title", "Recommended improvement"))}</div><div class="change-text">{safe_text(item.get("explanation"))}</div></div></div>'''
            )
        st.html('<div class="card">' + ''.join(change_html) + '</div>')

    rewrite = result.get("recommended_rewrite") or "No rewrite was provided."
    st.html('<div class="section"><div class="section-title">Recommended rewrite</div><div class="section-sub">Meaning and supported factual claims are preserved.</div></div>')
    st.html(f'<div class="rewrite">{safe_text(rewrite)}</div><div class="copy-note">Use this as a draft — keep your own final judgment before sending.</div>')
else:
    st.html('''
    <div class="section">
      <div class="empty">
        <div style="font-size:20px;color:#9a91ff;margin-bottom:7px;">✦</div>
        <div style="color:#c2c7d1;font-weight:700;">Your quality review will appear here</div>
        <div style="margin-top:5px;">One message in Clear human-touch signals out.</div>
      </div>
    </div>
    ''')

st.html('<div class="footer">Human Touch Quality Layer · Communication quality before send</div>')
