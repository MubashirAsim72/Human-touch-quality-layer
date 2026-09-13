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
# HUMAN TOUCH — SIMPLE PREMIUM UI
# ============================================================

st.set_page_config(
    page_title="Human Touch · Quality Layer",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

def score_color(score):
    score = clamp_score(score)
    if score >= 85:
        return "#72E6A1"
    if score >= 70:
        return "#9B8AFB"
    if score >= 55:
        return "#F2C66D"
    return "#F2778C"

def metric_card(label, value):
    score = clamp_score(value)
    color = score_color(score)
    return f"""
    <div class="metric-card">
        <div class="metric-label">{safe_text(label)}</div>
        <div class="metric-score" style="color:{color};">{score}</div>
        <div class="metric-bar"><span style="width:{score}%;background:{color};"></span></div>
    </div>
    """

def concern_card(item, number):
    severity = safe_text(item.get("severity", "Low"))
    cls = severity.lower() if severity.lower() in {"low", "medium", "high"} else "low"
    return f"""
    <div class="concern-card">
        <div class="number">0{number}</div>
        <div>
            <div class="card-title">{safe_text(item.get("issue", "Potential concern"))}</div>
            <div class="tag">{safe_text(item.get("dimension", ""))}</div>
            <p><b>Evidence:</b> {safe_text(item.get("evidence", ""))}</p>
            <p><b>Recipient impact:</b> {safe_text(item.get("impact_on_recipient", ""))}</p>
        </div>
        <div class="severity {cls}">{severity}</div>
    </div>
    """

def recommendation_card_simple(item, number):
    return f"""
    <div class="recommendation-card">
        <div class="number">0{number}</div>
        <div>
            <div class="card-title">{safe_text(item.get("title", "Recommended improvement"))}</div>
            <p>{safe_text(item.get("explanation", ""))}</p>
        </div>
        <div class="arrow">→</div>
    </div>
    """

def results_html(result, original):
    overall = clamp_score(result.get("overall_human_touch_score", 0))
    color = score_color(overall)

    metrics = "".join([
        metric_card("Empathy", result["scores"]["empathy"]),
        metric_card("Naturalness", result["scores"]["naturalness"]),
        metric_card("Personalization", result["scores"]["personalization"]),
        metric_card("Context awareness", result["scores"]["context_awareness"]),
        metric_card("Brand voice", result["scores"]["brand_voice"]),
    ])

    concerns = result.get("potential_concerns", [])
    recommendations = result.get("recommended_changes", [])

    concerns_html = "".join(
        concern_card(item, i + 1) for i, item in enumerate(concerns)
    )
    if not concerns_html:
        concerns_html = '<div class="empty">✓ No major concerns identified.</div>'

    recommendations_html = "".join(
        recommendation_card_simple(item, i + 1)
        for i, item in enumerate(recommendations)
    )
    if not recommendations_html:
        recommendations_html = '<div class="empty">✦ No major changes needed.</div>'

    original_html = safe_text(original).replace("\n", "<br>")
    rewrite_html = safe_text(result.get("recommended_rewrite", "")).replace("\n", "<br>")

    return f"""
    <div class="results">
        <div class="score-card">
            <div>
                <div class="eyebrow">HUMAN-TOUCH SCORE</div>
                <h2>{overall}<span>/100</span></h2>
                <div class="verdict" style="color:{color};">
                    <i style="background:{color};"></i>
                    {safe_text(result.get("verdict", ""))}
                </div>
                <p class="summary">{safe_text(result.get("summary", ""))}</p>
            </div>
            <div class="score-ring" style="--score:{overall};--ring:{color};">
                <div><strong>{overall}</strong><small>QUALITY</small></div>
            </div>
        </div>

        <div class="metrics">{metrics}</div>

        <div class="perspective">
            <div class="eyebrow">RECIPIENT PERSPECTIVE</div>
            <h3>How might this feel on the other side?</h3>
            <p>{safe_text(result.get("recipient_perspective", ""))}</p>
        </div>

        <div class="section">
            <div class="eyebrow">01 · CONCERNS</div>
            <h3>What could feel less human?</h3>
            {concerns_html}
        </div>

        <div class="section">
            <div class="eyebrow">02 · RECOMMENDATIONS</div>
            <h3>What should change?</h3>
            {recommendations_html}
        </div>

        <div class="section">
            <div class="eyebrow">03 · RECOMMENDED REWRITE</div>
            <h3>Keep the meaning. Improve the human touch.</h3>
            <div class="rewrite-grid">
                <div class="message-box">
                    <div class="box-label">ORIGINAL</div>
                    <div>{original_html}</div>
                </div>
                <div class="message-box improved">
                    <div class="box-label">RECOMMENDED</div>
                    <div>{rewrite_html}</div>
                </div>
            </div>
        </div>
    </div>
    """

st.html(r"""
<style>
:root{
    --bg:#080B12;
    --panel:#10151F;
    --panel2:#0D121A;
    --line:rgba(255,255,255,.09);
    --text:#F5F7FB;
    --muted:#8994A5;
    --dim:#5E6979;
    --accent:#8B7CF6;
    --accent2:#42C7E8;
}
html,body,[data-testid="stAppViewContainer"]{
    background:
        radial-gradient(700px 400px at 15% -5%,rgba(91,76,190,.16),transparent 65%),
        radial-gradient(650px 420px at 100% 15%,rgba(40,170,210,.08),transparent 65%),
        var(--bg)!important;
    color:var(--text)!important;
}
[data-testid="stHeader"]{background:transparent!important}
.block-container{
    max-width:1100px!important;
    padding:18px 28px 70px!important;
}
.nav{
    display:flex;justify-content:space-between;align-items:center;
    padding-bottom:16px;border-bottom:1px solid var(--line);
}
.brand{display:flex;align-items:center;gap:9px}
.logo{
    width:29px;height:29px;display:grid;place-items:center;
    border-radius:9px;color:white;
    background:linear-gradient(135deg,#7565F4,#2AAED0);
    font-weight:900;
}
.brand-name{
    font-size:10px;font-weight:800;letter-spacing:.12em;
    text-transform:uppercase;
}
.nav-status{
    color:#697487;font-size:8px;font-weight:700;
    letter-spacing:.12em;text-transform:uppercase;
}
.nav-status i{
    display:inline-block;width:5px;height:5px;margin-right:6px;
    border-radius:50%;background:#72E6A1;
    box-shadow:0 0 9px #72E6A1;
}
.hero{padding:65px 0 45px;text-align:center}
.eyebrow{
    color:#8E82F7;font-size:8px;font-weight:850;
    letter-spacing:.16em;text-transform:uppercase;
}
.hero h1{
    margin:14px 0 0;font-size:clamp(48px,7vw,78px);
    line-height:.94;letter-spacing:-.065em;font-weight:850;
}
.hero h1 span{
    background:linear-gradient(90deg,#A99EFF,#42C7E8);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}
.author{margin-top:15px;color:#697487;font-size:10px}
.hero-copy{
    max-width:650px;margin:16px auto 0;
    color:#818C9D;font-size:12px;line-height:1.8;
}
.pill{
    display:inline-block;margin-top:18px;padding:7px 11px;
    border:1px solid rgba(139,124,246,.25);
    border-radius:99px;background:rgba(139,124,246,.07);
    color:#AAA1FF;font-size:8px;font-weight:700;
}
.steps{
    display:grid;grid-template-columns:repeat(3,1fr);
    border:1px solid var(--line);border-radius:15px;
    overflow:hidden;background:rgba(255,255,255,.015);
}
.step{padding:15px 17px;border-right:1px solid var(--line)}
.step:last-child{border-right:0}
.step b{display:block;color:#8B7CF6;font-size:8px;letter-spacing:.12em}
.step strong{display:block;margin-top:6px;font-size:10px;color:#DCE2EA}
.step span{display:block;margin-top:3px;color:#657184;font-size:8px}
.panel{
    margin-top:22px;padding:22px;
    border:1px solid var(--line);border-radius:18px;
    background:linear-gradient(145deg,rgba(255,255,255,.025),rgba(255,255,255,.01));
}
.panel h3{margin:5px 0;color:#EDF1F7;font-size:18px}
.panel p{margin:0 0 14px;color:#697587;font-size:9px}
[data-baseweb="textarea"]>div,
[data-baseweb="input"]>div,
[data-baseweb="select"]>div{
    background:#090D14!important;
    border:1px solid rgba(255,255,255,.09)!important;
    border-radius:11px!important;
}
[data-baseweb="textarea"]>div:focus-within,
[data-baseweb="input"]>div:focus-within,
[data-baseweb="select"]>div:focus-within{
    border-color:rgba(139,124,246,.55)!important;
    box-shadow:0 0 0 3px rgba(139,124,246,.08)!important;
}
textarea,input{color:#F5F7FB!important}
textarea::placeholder,input::placeholder{color:#4E596A!important}
[data-baseweb="select"] *{color:#F5F7FB!important}
label,[data-testid="stWidgetLabel"] p{
    color:#7F8B9D!important;font-size:9px!important;font-weight:650!important;
}
.stButton>button{
    min-height:56px!important;border-radius:12px!important;
    border:0!important;color:#fff!important;font-weight:800!important;
    background:linear-gradient(100deg,#725EF1,#8D73F7,#269EC2)!important;
    box-shadow:0 15px 40px rgba(112,91,235,.22)!important;
}
.stButton>button:hover{filter:brightness(1.06);transform:translateY(-1px)}
.note{text-align:center;margin-top:8px;color:#4F5B6C;font-size:8px}
.results{margin-top:52px}
.score-card{
    display:grid;grid-template-columns:1fr 190px;
    gap:20px;align-items:center;padding:27px;
    border:1px solid rgba(139,124,246,.2);
    border-radius:20px;
    background:
        radial-gradient(circle at 80% 50%,rgba(139,124,246,.10),transparent 32%),
        var(--panel);
}
.score-card h2{
    margin:8px 0 0;font-size:52px;line-height:1;
    letter-spacing:-.06em;
}
.score-card h2 span{font-size:14px;color:#667285;margin-left:5px}
.verdict{display:inline-flex;align-items:center;gap:6px;margin-top:12px;font-size:9px;font-weight:750}
.verdict i{width:5px;height:5px;border-radius:50%}
.summary{max-width:650px;color:#8490A1;font-size:10px;line-height:1.75;margin-top:12px}
.score-ring{
    width:160px;height:160px;margin:auto;border-radius:50%;
    display:grid;place-items:center;
    background:conic-gradient(var(--ring) calc(var(--score)*1%),rgba(255,255,255,.07) 0);
    position:relative;
}
.score-ring:after{
    content:"";position:absolute;inset:8px;border-radius:50%;
    background:#0B1017;
}
.score-ring>div{position:relative;z-index:2;text-align:center}
.score-ring strong{display:block;font-size:38px;letter-spacing:-.06em}
.score-ring small{display:block;margin-top:3px;color:#677386;font-size:7px;letter-spacing:.13em}
.metrics{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-top:8px}
.metric-card{
    padding:16px;border:1px solid var(--line);border-radius:14px;
    background:rgba(255,255,255,.015);
}
.metric-label{color:#7D899A;font-size:8px}
.metric-score{margin-top:7px;font-size:24px;font-weight:800}
.metric-bar{height:3px;margin-top:10px;background:rgba(255,255,255,.06);border-radius:99px;overflow:hidden}
.metric-bar span{display:block;height:100%;border-radius:99px}
.perspective,.section{margin-top:32px}
.perspective{
    padding:22px;border:1px solid rgba(139,124,246,.13);
    border-radius:16px;background:rgba(139,124,246,.045);
}
.perspective h3,.section h3{margin:6px 0 13px;color:#E9EDF4;font-size:21px;letter-spacing:-.03em}
.perspective p{margin:0;color:#8994A4;font-size:10px;line-height:1.8}
.concern-card,.recommendation-card{
    display:grid;grid-template-columns:35px 1fr auto;gap:13px;
    padding:17px;margin:7px 0;
    border:1px solid var(--line);border-radius:13px;
    background:rgba(255,255,255,.012);
}
.number{color:#515D6E;font-size:8px;font-weight:800}
.card-title{color:#E0E5EC;font-size:10px;font-weight:750}
.tag{
    display:inline-block;margin-top:4px;color:#657184;
    font-size:7px;text-transform:uppercase;letter-spacing:.08em;
}
.concern-card p,.recommendation-card p{
    margin:9px 0 0;color:#778396;font-size:8px;line-height:1.65;
}
.concern-card p b{display:block;color:#9DA8B7;font-size:7px}
.severity{
    align-self:start;padding:4px 7px;border-radius:99px;
    font-size:6px;font-weight:800;text-transform:uppercase;
}
.severity.low{color:#72E6A1;background:rgba(114,230,161,.06)}
.severity.medium{color:#F2C66D;background:rgba(242,198,109,.06)}
.severity.high{color:#F2778C;background:rgba(242,119,140,.06)}
.arrow{color:#778396}
.empty{
    padding:17px;border:1px dashed rgba(255,255,255,.09);
    border-radius:12px;color:#728094;font-size:9px
}
.rewrite-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.message-box{
    min-height:210px;padding:18px;border:1px solid var(--line);
    border-radius:14px;background:#0B0F16;
    color:#AAB4C2;font-size:9px;line-height:1.8;
}
.message-box.improved{border-color:rgba(114,230,161,.16)}
.box-label{margin-bottom:10px;color:#5E6A7B;font-size:7px;font-weight:800;letter-spacing:.12em}
.footer{
    margin-top:55px;padding-top:16px;border-top:1px solid var(--line);
    display:flex;justify-content:space-between;color:#475365;font-size:7px;
}
@media(max-width:800px){
    .metrics{grid-template-columns:repeat(2,1fr)}
    .score-card{grid-template-columns:1fr}
    .score-ring{order:-1}
}
@media(max-width:600px){
    .block-container{padding:12px 12px 50px!important}
    .steps{grid-template-columns:1fr}
    .step{border-right:0;border-bottom:1px solid var(--line)}
    .step:last-child{border-bottom:0}
    .metrics{grid-template-columns:1fr}
    .rewrite-grid{grid-template-columns:1fr}
    .concern-card,.recommendation-card{grid-template-columns:28px 1fr}
    .severity{grid-column:2;justify-self:start}
    .footer{flex-direction:column;gap:7px}
}
</style>
""")

st.html("""
<div class="nav">
    <div class="brand">
        <div class="logo">✦</div>
        <div class="brand-name">Human Touch · Quality Layer</div>
    </div>
    <div class="nav-status"><i></i> Communication intelligence</div>
</div>
""")

st.html("""
<div class="hero">
    <div class="eyebrow">A QUALITY LAYER FOR HUMAN COMMUNICATION</div>
    <h1>Make every message<br><span>feel considered.</span></h1>
    <div class="author">By Engr. Muhammad Mubashir Asim</div>
    <div class="hero-copy">
        Check whether your message feels thoughtful, natural, personal,
        context-aware, and aligned with the intended voice — before you send it.
    </div>
    <div class="pill">● Human intelligence layer active</div>
</div>
""")

st.html("""
<div class="steps">
    <div class="step">
        <b>01 · MESSAGE</b>
        <strong>Write or paste</strong>
        <span>Start with the exact communication.</span>
    </div>
    <div class="step">
        <b>02 · CONTEXT</b>
        <strong>Tell us the situation</strong>
        <span>Recipient, purpose, voice, background.</span>
    </div>
    <div class="step">
        <b>03 · ANALYSIS</b>
        <strong>Improve the human touch</strong>
        <span>Score, evidence, recommendations, rewrite.</span>
    </div>
</div>
""")

left, right = st.columns([1.35, .65], gap="small")

with left:
    st.html("""
    <div class="panel">
        <div class="eyebrow">01 · MESSAGE</div>
        <h3>Your message</h3>
        <p>Paste the exact message your recipient will see.</p>
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
        placeholder="Write or paste your message here…",
        height=250,
        label_visibility="collapsed",
    )
    st.html(f'<div class="note" style="text-align:left;">{len(message or "")} characters</div>')

with right:
    st.html("""
    <div class="panel">
        <div class="eyebrow">02 · CONTEXT</div>
        <h3>Recipient context</h3>
        <p>Help the evaluation understand the situation.</p>
    </div>
    """)
    audience = st.text_input("Audience / Recipient", placeholder="Existing customer")
    purpose = st.text_input("Purpose", placeholder="Confirmation")
    brand_voice = st.text_input("Brand Voice", placeholder="Warm, professional")
    additional_context = st.text_input("Additional Context", placeholder="Optional")

st.write("")
analyze = st.button("✦  Analyze Human Touch", type="primary", use_container_width=True)

st.html("""
<div class="note">
    Evaluates communication quality — not whether the text was written by a human or AI.
</div>
""")

if analyze:
    if not message.strip():
        st.error("Please paste a message before analyzing.")
        st.session_state.result = None
    else:
        with st.spinner("Analyzing the message…"):
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
                if any(x in lower for x in ["401", "authentication", "api key", "unauthorized"]):
                    st.error("Groq authentication failed. Check GROQ_API_KEY in Streamlit Secrets.")
                elif "rate" in lower and "limit" in lower:
                    st.error("Groq rate limit reached. Please wait and try again.")
                elif "model" in lower and ("not found" in lower or "does not exist" in lower):
                    st.error(
                        f"Groq model error. Current model: {get_model()}. "
                        "Set GROQ_MODEL to a supported model in Streamlit Secrets."
                    )
                else:
                    st.error(f"Evaluation failed: {error_text}")

if "result" not in st.session_state:
    st.session_state.result = None
if "analyzed_message" not in st.session_state:
    st.session_state.analyzed_message = ""

if st.session_state.result:
    st.html("""
    <div style="margin-top:50px;padding-top:24px;border-top:1px solid rgba(255,255,255,.09);">
        <div class="eyebrow">03 · RESULTS</div>
        <h2 style="margin:7px 0 0;color:#EEF2F7;font-size:30px;letter-spacing:-.04em;">
            Your human-touch readout
        </h2>
    </div>
    """)
    st.html(results_html(st.session_state.result, st.session_state.analyzed_message))

st.html("""
<div class="footer">
    <div>HUMAN TOUCH · QUALITY LAYER</div>
    <div>Communication quality · Recipient perspective · Human connection</div>
</div>
""")
