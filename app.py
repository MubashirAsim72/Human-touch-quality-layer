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
# HUMAN TOUCH — QUALITY CONTROL CENTER
# Production SaaS dashboard UI
# ============================================================

st.set_page_config(
    page_title="Human Touch · Quality Control Center",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Design helpers ----------
def score_color(score):
    score = clamp_score(score)
    if score >= 85:
        return "#18B981"
    if score >= 70:
        return "#6957E8"
    if score >= 55:
        return "#D99A27"
    return "#DC5C6D"


def kpi_card(title, value, note, icon, color):
    return f"""
    <div class="kpi-card">
        <div class="kpi-icon" style="color:{color};background:{color}12;">{icon}</div>
        <div class="kpi-main">
            <div class="kpi-label">{safe_text(title)}</div>
            <div class="kpi-value">{safe_text(value)}</div>
            <div class="kpi-note">{safe_text(note)}</div>
        </div>
    </div>
    """


def metric_card(label, value):
    score = clamp_score(value)
    color = score_color(score)
    return f"""
    <div class="metric-card">
        <div class="metric-head">
            <span>{safe_text(label)}</span>
            <strong style="color:{color};">{score}</strong>
        </div>
        <div class="metric-track">
            <i style="width:{score}%;background:{color};"></i>
        </div>
    </div>
    """


def concern_card(item, number):
    severity = safe_text(item.get("severity", "Low"))
    cls = severity.lower() if severity.lower() in {"low", "medium", "high"} else "low"
    return f"""
    <div class="detail-row">
        <div class="row-number">0{number}</div>
        <div class="row-content">
            <div class="row-title">{safe_text(item.get("issue", "Potential concern"))}</div>
            <div class="row-tag">{safe_text(item.get("dimension", ""))}</div>
            <div class="row-text"><b>Evidence</b>{safe_text(item.get("evidence", ""))}</div>
            <div class="row-text"><b>Recipient impact</b>{safe_text(item.get("impact_on_recipient", ""))}</div>
        </div>
        <span class="severity {cls}">{severity}</span>
    </div>
    """


def recommendation_card(item, number):
    return f"""
    <div class="detail-row recommendation">
        <div class="row-number">0{number}</div>
        <div class="row-content">
            <div class="row-title">{safe_text(item.get("title", "Recommended improvement"))}</div>
            <div class="row-text">{safe_text(item.get("explanation", ""))}</div>
        </div>
        <div class="row-arrow">↗</div>
    </div>
    """


def render_results(result, original):
    overall = clamp_score(result.get("overall_human_touch_score", 0))
    color = score_color(overall)

    circumference = 2 * 3.1415926535 * 48
    dash = circumference * overall / 100

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
        concern_card(item, i + 1)
        for i, item in enumerate(concerns)
    ) or """
    <div class="empty-state">
        <span>✓</span>
        <div><b>No major concerns.</b><br>The message is showing strong human-touch signals.</div>
    </div>
    """

    recommendations_html = "".join(
        recommendation_card(item, i + 1)
        for i, item in enumerate(recommendations)
    ) or """
    <div class="empty-state">
        <span>✦</span>
        <div><b>No major changes needed.</b><br>The communication is already relatively strong.</div>
    </div>
    """

    original_html = safe_text(original).replace("\n", "<br>")
    rewrite_html = safe_text(result.get("recommended_rewrite", "")).replace("\n", "<br>")

    return f"""
    <div class="result-wrapper">

        <div class="result-hero">
            <div class="result-copy">
                <div class="eyebrow">QUALITY REPORT</div>
                <h2>Your human-touch assessment</h2>
                <p>{safe_text(result.get("summary", ""))}</p>
                <div class="verdict" style="color:{color};border-color:{color}35;">
                    <i style="background:{color};"></i>
                    {safe_text(result.get("verdict", ""))}
                </div>
            </div>

            <div class="score-dial">
                <svg viewBox="0 0 110 110">
                    <circle class="dial-bg" cx="55" cy="55" r="48"></circle>
                    <circle class="dial-value" cx="55" cy="55" r="48"
                        stroke="{color}"
                        stroke-dasharray="{dash:.1f} {circumference:.1f}"></circle>
                </svg>
                <div class="dial-number">
                    <strong>{overall}</strong>
                    <span>/100</span>
                </div>
                <small>HUMAN TOUCH</small>
            </div>
        </div>

        <div class="dimension-grid">{metrics}</div>

        <div class="recipient-panel">
            <div class="recipient-icon">◎</div>
            <div>
                <div class="eyebrow">RECIPIENT PERSPECTIVE</div>
                <h3>How might this feel on the other side?</h3>
                <p>{safe_text(result.get("recipient_perspective", ""))}</p>
            </div>
        </div>

        <div class="report-section">
            <div class="section-heading">
                <div>
                    <div class="section-number">01</div>
                    <div class="eyebrow">POTENTIAL CONCERNS</div>
                    <h3>Signals worth improving</h3>
                </div>
                <span>Evidence based</span>
            </div>
            {concerns_html}
        </div>

        <div class="report-section">
            <div class="section-heading">
                <div>
                    <div class="section-number">02</div>
                    <div class="eyebrow">RECOMMENDED CHANGES</div>
                    <h3>Practical improvements</h3>
                </div>
                <span>Actionable</span>
            </div>
            {recommendations_html}
        </div>

        <div class="report-section">
            <div class="section-heading">
                <div>
                    <div class="section-number">03</div>
                    <div class="eyebrow">RECOMMENDED REWRITE</div>
                    <h3>Keep the meaning. Improve the connection.</h3>
                </div>
                <span>Claim-safe</span>
            </div>

            <div class="rewrite-grid">
                <div class="rewrite-box">
                    <div class="rewrite-label">ORIGINAL MESSAGE</div>
                    <div class="rewrite-text">{original_html}</div>
                </div>
                <div class="rewrite-box improved">
                    <div class="rewrite-label">RECOMMENDED VERSION</div>
                    <div class="rewrite-text">{rewrite_html}</div>
                </div>
            </div>
        </div>
    </div>
    """


# ---------- Professional dashboard CSS ----------
st.html(r"""
<style>
:root{
    --bg:#F5F7FA;
    --surface:#FFFFFF;
    --surface-soft:#F8FAFC;
    --border:#E7EBF0;
    --border-dark:#DDE3EA;
    --text:#172033;
    --muted:#687386;
    --subtle:#9AA4B2;
    --primary:#6957E8;
    --primary-soft:#F0EEFF;
    --success:#18B981;
    --warning:#D99A27;
    --danger:#DC5C6D;
    --shadow:0 8px 30px rgba(31,41,55,.045);
}
html,body,[data-testid="stAppViewContainer"]{
    background:var(--bg)!important;
    color:var(--text)!important;
}
[data-testid="stHeader"]{background:transparent!important}
.block-container{
    max-width:1320px!important;
    padding:0 30px 70px!important;
}

/* SIDEBAR */
[data-testid="stSidebar"]{
    background:#101725!important;
    border-right:0!important;
}
[data-testid="stSidebar"]>div{
    padding:0 17px 20px!important;
}
.sidebar-brand{
    height:72px;display:flex;align-items:center;gap:10px;
    border-bottom:1px solid rgba(255,255,255,.07);
    margin-bottom:20px;
}
.sidebar-logo{
    width:31px;height:31px;border-radius:9px;
    display:grid;place-items:center;
    background:linear-gradient(135deg,#7868EF,#4A9DDF);
    color:#fff;font-weight:900;
}
.sidebar-title{
    color:#F5F7FB;font-size:11px;font-weight:800;
    letter-spacing:.04em;
}
.sidebar-sub{color:#748095;font-size:7px;margin-top:2px}
.sidebar-section{
    color:#58667A;font-size:7px;font-weight:800;
    letter-spacing:.13em;text-transform:uppercase;
    margin:20px 7px 7px;
}
.nav-item{
    display:flex;align-items:center;gap:10px;
    padding:10px 10px;margin:3px 0;border-radius:8px;
    color:#8D99AA;font-size:9px;font-weight:600;
}
.nav-item.active{
    color:#fff;background:rgba(105,87,232,.16);
}
.nav-icon{
    width:18px;text-align:center;color:#7E8AA0;font-size:11px;
}
.nav-item.active .nav-icon{color:#A89EFF}
.sidebar-bottom{
    margin-top:30px;padding:14px 9px;
    border-top:1px solid rgba(255,255,255,.07);
}
.sidebar-user{
    display:flex;align-items:center;gap:9px;
}
.avatar{
    width:29px;height:29px;border-radius:50%;
    display:grid;place-items:center;
    background:#252F42;color:#B9C3D2;font-size:8px;font-weight:800;
}
.user-name{color:#D7DEE8;font-size:8px;font-weight:700}
.user-role{color:#657287;font-size:7px;margin-top:2px}

/* TOP HEADER */
.topbar{
    height:72px;display:flex;align-items:center;
    justify-content:space-between;
    border-bottom:1px solid var(--border);
}
.breadcrumb{
    color:#98A2B1;font-size:9px;
}
.breadcrumb b{color:#3E4858}
.top-actions{display:flex;align-items:center;gap:9px}
.search{
    width:190px;height:32px;padding:0 12px;
    display:flex;align-items:center;gap:7px;
    border:1px solid var(--border);border-radius:8px;
    background:#fff;color:#98A2B1;font-size:8px;
}
.icon-button{
    width:32px;height:32px;display:grid;place-items:center;
    border:1px solid var(--border);border-radius:8px;
    background:#fff;color:#657184;font-size:12px;
}

/* PAGE HEADER */
.page-header{
    display:flex;align-items:end;justify-content:space-between;
    gap:20px;padding:30px 0 20px;
}
.page-title{
    margin:0;color:#172033;font-size:28px;
    font-weight:780;letter-spacing:-.035em;
}
.page-subtitle{
    margin-top:5px;color:#7B8696;font-size:10px;
}
.header-actions{display:flex;gap:8px}
.action-secondary{
    padding:9px 13px;border:1px solid var(--border);
    border-radius:8px;background:#fff;color:#4D596A;
    font-size:8px;font-weight:700;
}
.action-primary{
    padding:9px 14px;border:0;border-radius:8px;
    background:#6957E8;color:#fff;font-size:8px;font-weight:750;
    box-shadow:0 5px 14px rgba(105,87,232,.18);
}

/* DASHBOARD KPI STRIP */
.kpi-grid{
    display:grid;grid-template-columns:repeat(4,1fr);
    gap:10px;
}
.kpi-card{
    min-height:100px;padding:17px;
    border:1px solid var(--border);border-radius:11px;
    background:var(--surface);box-shadow:var(--shadow);
    display:flex;gap:12px;align-items:flex-start;
}
.kpi-icon{
    width:31px;height:31px;border-radius:8px;
    display:grid;place-items:center;font-size:11px;
}
.kpi-label{color:#7C8797;font-size:8px;font-weight:650}
.kpi-value{
    margin-top:4px;color:#182133;font-size:24px;
    font-weight:780;letter-spacing:-.04em;
}
.kpi-note{margin-top:3px;color:#A0A9B5;font-size:7px}

/* MAIN WORKSPACE */
.workspace-title{
    margin:31px 0 10px;color:#5C6879;font-size:8px;
    font-weight:800;letter-spacing:.11em;text-transform:uppercase;
}
.workspace{
    display:grid;grid-template-columns:1.4fr .8fr;gap:11px;
}
.card{
    border:1px solid var(--border);border-radius:12px;
    background:#fff;box-shadow:var(--shadow);
    padding:20px;
}
.card-header{
    display:flex;justify-content:space-between;gap:10px;
    margin-bottom:14px;
}
.card-eyebrow{
    color:#8175E8;font-size:7px;font-weight:800;
    letter-spacing:.12em;text-transform:uppercase;
}
.card-title{
    margin-top:5px;color:#1B2537;font-size:16px;font-weight:750;
}
.card-description{
    margin-top:4px;color:#8791A0;font-size:8px;line-height:1.6;
}
.badge{
    align-self:start;padding:5px 8px;border-radius:99px;
    color:#168B63;background:#EAF9F3;font-size:7px;font-weight:750;
}

/* STREAMLIT FORM CONTROLS */
[data-baseweb="textarea"]>div,
[data-baseweb="input"]>div,
[data-baseweb="select"]>div{
    background:#fff!important;
    border:1px solid #DDE3EA!important;
    border-radius:8px!important;
}
[data-baseweb="textarea"]>div:focus-within,
[data-baseweb="input"]>div:focus-within,
[data-baseweb="select"]>div:focus-within{
    border-color:#8D82EF!important;
    box-shadow:0 0 0 3px rgba(105,87,232,.08)!important;
}
textarea,input{color:#1D2738!important}
textarea::placeholder,input::placeholder{color:#A2ABB8!important}
[data-baseweb="select"] *{color:#1D2738!important}
label,[data-testid="stWidgetLabel"] p{
    color:#687487!important;font-size:8px!important;font-weight:650!important;
}
[data-testid="stRadio"] label{color:#687487!important}
[data-testid="stRadio"] [role="radiogroup"]{gap:2px}

/* CTA */
.stButton>button{
    min-height:52px!important;
    border-radius:8px!important;
    border:0!important;
    background:#6957E8!important;
    color:#fff!important;
    font-size:10px!important;
    font-weight:800!important;
    box-shadow:0 8px 20px rgba(105,87,232,.18)!important;
}
.stButton>button:hover{
    background:#5C4AD8!important;
    transform:translateY(-1px);
}
.disclaimer{
    margin-top:7px;text-align:center;
    color:#A0A9B5;font-size:7px;
}

/* RESULTS */
.result-wrapper{margin-top:34px}
.result-hero{
    display:grid;grid-template-columns:1fr 190px;
    gap:20px;align-items:center;
    padding:25px;
    border:1px solid #DED9FF;
    border-radius:13px;
    background:linear-gradient(135deg,#FBFAFF,#FFFFFF);
    box-shadow:var(--shadow);
}
.eyebrow{
    color:#766AE0;font-size:7px;font-weight:850;
    letter-spacing:.13em;text-transform:uppercase;
}
.result-copy h2{
    margin:6px 0 0;color:#182133;font-size:25px;
    font-weight:780;letter-spacing:-.035em;
}
.result-copy p{
    max-width:700px;margin:9px 0 0;color:#6D7889;
    font-size:9px;line-height:1.7;
}
.verdict{
    display:inline-flex;align-items:center;gap:6px;
    margin-top:12px;padding:6px 9px;
    border:1px solid;border-radius:99px;
    font-size:7px;font-weight:750;
}
.verdict i{width:5px;height:5px;border-radius:50%}
.score-dial{
    width:145px;height:145px;position:relative;margin:auto;
}
.score-dial svg{width:145px;height:145px;transform:rotate(-90deg)}
.dial-bg{fill:none;stroke:#E8EBF0;stroke-width:6}
.dial-value{fill:none;stroke-width:6;stroke-linecap:round}
.dial-number{
    position:absolute;inset:0;display:flex;
    align-items:center;justify-content:center;
    padding-bottom:5px;
}
.dial-number strong{
    color:#182133;font-size:37px;font-weight:820;
    letter-spacing:-.06em;
}
.dial-number span{margin-left:3px;color:#9AA3AF;font-size:8px}
.score-dial small{
    position:absolute;bottom:27px;left:0;right:0;
    text-align:center;color:#A1A9B4;font-size:6px;
    font-weight:800;letter-spacing:.13em;
}

/* DIMENSIONS */
.dimension-grid{
    display:grid;grid-template-columns:repeat(5,1fr);
    gap:8px;margin-top:8px;
}
.metric-card{
    padding:15px;border:1px solid var(--border);
    border-radius:10px;background:#fff;box-shadow:var(--shadow);
}
.metric-head{
    display:flex;justify-content:space-between;align-items:center;
}
.metric-head span{color:#778294;font-size:8px}
.metric-head strong{font-size:19px;font-weight:800}
.metric-track{
    height:3px;margin-top:10px;
    background:#EEF1F5;border-radius:99px;overflow:hidden;
}
.metric-track i{display:block;height:100%;border-radius:99px}

/* RECIPIENT */
.recipient-panel{
    display:grid;grid-template-columns:38px 1fr;gap:12px;
    margin-top:8px;padding:19px 20px;
    border:1px solid #E6E1FF;border-radius:10px;
    background:#FAF9FF;
}
.recipient-icon{
    width:33px;height:33px;display:grid;place-items:center;
    border-radius:8px;background:#EEEAFE;color:#6B5BE5;font-size:15px;
}
.recipient-panel h3{
    margin:5px 0 0;color:#263044;font-size:14px;font-weight:740;
}
.recipient-panel p{
    margin:6px 0 0;color:#727D8D;font-size:9px;line-height:1.7;
}

/* REPORT SECTIONS */
.report-section{margin-top:29px}
.section-heading{
    display:flex;justify-content:space-between;align-items:end;
    gap:20px;margin-bottom:11px;
}
.section-heading h3{
    margin:5px 0 0;color:#202A3B;font-size:18px;
    font-weight:750;letter-spacing:-.025em;
}
.section-heading>span{
    color:#A0A9B5;font-size:7px;font-weight:700;
}
.section-number{color:#B0B8C4;font-size:7px;font-weight:800;margin-bottom:3px}

/* DETAIL ROWS */
.detail-row{
    display:grid;grid-template-columns:32px 1fr auto;
    gap:12px;align-items:start;
    padding:16px;margin:5px 0;
    border:1px solid var(--border);border-radius:9px;
    background:#fff;box-shadow:0 2px 10px rgba(31,41,55,.025);
}
.row-number{color:#A3ACB8;font-size:7px;font-weight:800}
.row-title{color:#273246;font-size:9px;font-weight:750}
.row-tag{
    display:inline-block;margin-top:4px;
    color:#7D8898;font-size:6px;text-transform:uppercase;
    letter-spacing:.08em;
}
.row-text{
    margin-top:8px;color:#768193;font-size:8px;line-height:1.6;
}
.row-text b{
    display:block;margin-bottom:2px;color:#8B95A3;
    font-size:6px;letter-spacing:.07em;text-transform:uppercase;
}
.severity{
    padding:4px 7px;border-radius:99px;
    font-size:6px;font-weight:800;text-transform:uppercase;
}
.severity.low{color:#168B63;background:#EAF9F3}
.severity.medium{color:#A66A0A;background:#FFF5DD}
.severity.high{color:#B84355;background:#FFF0F2}
.recommendation{align-items:center}
.recommendation .row-text{margin-top:6px}
.row-arrow{color:#8B82DF;font-size:12px}

/* REWRITE */
.rewrite-grid{
    display:grid;grid-template-columns:1fr 1fr;gap:8px;
}
.rewrite-box{
    min-height:210px;padding:18px;
    border:1px solid var(--border);border-radius:10px;
    background:#fff;
}
.rewrite-box.improved{
    border-color:#CDEFE2;
    background:#FBFEFC;
}
.rewrite-label{
    color:#8B95A3;font-size:6px;font-weight:800;
    letter-spacing:.12em;
}
.rewrite-text{
    margin-top:10px;color:#5F6B7C;font-size:9px;line-height:1.8;
}

/* EMPTY / FOOTER */
.empty-state{
    display:flex;gap:9px;align-items:center;
    padding:15px;border:1px dashed #DDE3EA;
    border-radius:9px;color:#7C8796;font-size:8px;line-height:1.6;
}
.empty-state span{color:#18B981;font-size:14px}
.empty-state b{color:#4E5A6C}
.footer{
    display:flex;justify-content:space-between;
    margin-top:55px;padding-top:15px;
    border-top:1px solid var(--border);
    color:#A0A8B4;font-size:7px;
}

/* MOBILE */
@media(max-width:900px){
    .workspace{grid-template-columns:1fr}
    .kpi-grid{grid-template-columns:repeat(2,1fr)}
    .dimension-grid{grid-template-columns:repeat(2,1fr)}
    .result-hero{grid-template-columns:1fr}
    .score-dial{order:-1}
}
@media(max-width:600px){
    .block-container{padding:0 12px 50px!important}
    .top-actions .search{display:none}
    .page-header{align-items:flex-start;flex-direction:column}
    .kpi-grid,.dimension-grid{grid-template-columns:1fr}
    .rewrite-grid{grid-template-columns:1fr}
    .detail-row{grid-template-columns:25px 1fr}
    .detail-row .severity{grid-column:2;justify-self:start}
    .footer{flex-direction:column;gap:6px}
}
</style>
""")

# ---------- Sidebar ----------
st.html("""
<div class="sidebar-brand">
    <div class="sidebar-logo">✦</div>
    <div>
        <div class="sidebar-title">Human Touch</div>
        <div class="sidebar-sub">QUALITY LAYER</div>
    </div>
</div>

<div class="sidebar-section">Workspace</div>
<div class="nav-item active"><span class="nav-icon">⌂</span> Overview</div>
<div class="nav-item"><span class="nav-icon">✦</span> Analyze Message</div>
<div class="nav-item"><span class="nav-icon">◈</span> Quality Reports</div>

<div class="sidebar-section">Product</div>
<div class="nav-item"><span class="nav-icon">◎</span> Brand Voice</div>
<div class="nav-item"><span class="nav-icon">▤</span> Templates</div>
<div class="nav-item"><span class="nav-icon">◌</span> Insights</div>

<div class="sidebar-section">Settings</div>
<div class="nav-item"><span class="nav-icon">⚙</span> Preferences</div>
<div class="nav-item"><span class="nav-icon">?</span> Help & Support</div>

<div class="sidebar-bottom">
    <div class="sidebar-user">
        <div class="avatar">MA</div>
        <div>
            <div class="user-name">Engr. Muhammad Mubashir Asim</div>
            <div class="user-role">Product owner</div>
        </div>
    </div>
</div>
""")

# ---------- Header ----------
st.html("""
<div class="topbar">
    <div class="breadcrumb"><b>Workspace</b> &nbsp;/&nbsp; Human Touch</div>
    <div class="top-actions">
        <div class="search">⌕ &nbsp; Search</div>
        <div class="icon-button">⌁</div>
        <div class="icon-button">◔</div>
    </div>
</div>
""")

# ---------- Page header ----------
st.html("""
<div class="page-header">
    <div>
        <h1 class="page-title">Communication Quality Center</h1>
        <div class="page-subtitle">
            Evaluate how human, thoughtful, and context-aware your message feels before it reaches someone.
        </div>
    </div>
    <div class="header-actions">
        <div class="action-secondary">Documentation</div>
        <div class="action-primary">✦ New analysis</div>
    </div>
</div>
""")

# ---------- Dashboard overview ----------
st.html("""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-icon" style="color:#6957E8;background:#6957E812;">✦</div>
        <div class="kpi-main">
            <div class="kpi-label">QUALITY DIMENSIONS</div>
            <div class="kpi-value">05</div>
            <div class="kpi-note">Core human-touch signals</div>
        </div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon" style="color:#18B981;background:#18B98112;">✓</div>
        <div class="kpi-main">
            <div class="kpi-label">ANALYSIS MODE</div>
            <div class="kpi-value">Live</div>
            <div class="kpi-note">Recipient-first evaluation</div>
        </div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon" style="color:#D99A27;background:#D99A2712;">◈</div>
        <div class="kpi-main">
            <div class="kpi-label">SIGNAL TYPE</div>
            <div class="kpi-value">Human</div>
            <div class="kpi-note">Communication quality</div>
        </div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon" style="color:#42A8D0;background:#42A8D012;">↗</div>
        <div class="kpi-main">
            <div class="kpi-label">WORKFLOW</div>
            <div class="kpi-value">3 steps</div>
            <div class="kpi-note">Message → Context → Report</div>
        </div>
    </div>
</div>
""")

# ---------- Analyzer workspace ----------
st.html('<div class="workspace-title">Analyze a message</div>')

left, right = st.columns([1.35, .65], gap="small")

with left:
    st.html("""
    <div class="card">
        <div class="card-header">
            <div>
                <div class="card-eyebrow">MESSAGE INPUT</div>
                <div class="card-title">Analyze your communication</div>
                <div class="card-description">
                    Paste the exact message the recipient will see. The evaluator focuses on quality, not authorship detection.
                </div>
            </div>
            <div class="badge">Ready</div>
        </div>
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

    st.html(
        f'<div style="color:#A0A9B5;font-size:7px;">'
        f'{len(message or "")} characters · supported by evidence-based analysis</div>'
    )

with right:
    st.html("""
    <div class="card">
        <div class="card-header">
            <div>
                <div class="card-eyebrow">RECIPIENT CONTEXT</div>
                <div class="card-title">Add the situation</div>
                <div class="card-description">
                    More context helps distinguish generic polish from genuine consideration.
                </div>
            </div>
        </div>
    </div>
    """)

    audience = st.text_input("Audience / Recipient", placeholder="Existing customer")
    purpose = st.text_input("Purpose", placeholder="Confirmation")
    brand_voice = st.text_input("Brand Voice", placeholder="Warm, professional")
    additional_context = st.text_input("Additional Context", placeholder="Optional background")

st.write("")
analyze = st.button("✦  Analyze Human Touch", type="primary", use_container_width=True)

st.html("""
<div class="disclaimer">
    Human Touch evaluates communication quality. It is not a definitive AI detector.
</div>
""")

# ---------- Evaluation ----------
if analyze:
    if not message.strip():
        st.error("Please paste a message before analyzing.")
        st.session_state.result = None
    else:
        with st.spinner("Analyzing the message through the recipient's perspective…"):
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

# ---------- Results ----------
if "result" not in st.session_state:
    st.session_state.result = None
if "analyzed_message" not in st.session_state:
    st.session_state.analyzed_message = ""

if st.session_state.result:
    st.html("""
    <div style="margin-top:42px;padding-top:24px;border-top:1px solid #E7EBF0;">
        <div class="eyebrow">ANALYSIS OUTPUT</div>
        <div style="margin-top:5px;color:#172033;font-size:24px;font-weight:780;letter-spacing:-.035em;">
            Quality report
        </div>
    </div>
    """)
    st.html(render_results(st.session_state.result, st.session_state.analyzed_message))

# ---------- Footer ----------
st.html("""
<div class="footer">
    <div>HUMAN TOUCH · QUALITY CONTROL CENTER</div>
    <div>By Engr. Muhammad Mubashir Asim · Communication quality, recipient perspective, human connection</div>
</div>
""")
