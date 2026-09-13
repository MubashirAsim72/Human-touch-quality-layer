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
# HUMAN TOUCH — PREMIUM PRODUCT UI
# ============================================================

st.set_page_config(
    page_title="Human Touch · Quality Layer",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- Visual helpers ----------
def score_color(score: int) -> str:
    score = clamp_score(score)
    if score >= 85:
        return "#4ADE80"
    if score >= 70:
        return "#A78BFA"
    if score >= 55:
        return "#FBBF24"
    return "#FB7185"


def metric_card(label: str, score: Any, icon: str) -> str:
    value = clamp_score(score)
    color = score_color(value)
    return f"""
    <div class="metric">
        <div class="metric-icon">{icon}</div>
        <div class="metric-label">{safe_text(label)}</div>
        <div class="metric-score">{value}</div>
        <div class="metric-bar">
            <span style="width:{value}%;background:{color};"></span>
        </div>
    </div>
    """


def concern_card(item: Dict[str, Any], number: int) -> str:
    severity = safe_text(item.get("severity", "Low"))
    sev = severity.lower() if severity.lower() in {"low", "medium", "high"} else "low"

    return f"""
    <div class="list-card">
        <div class="list-number">0{number}</div>
        <div class="list-content">
            <div class="list-title">{safe_text(item.get("issue", "Potential concern"))}</div>
            <div class="list-meta">{safe_text(item.get("dimension", ""))}</div>
            <div class="list-copy"><b>Evidence:</b> {safe_text(item.get("evidence", ""))}</div>
            <div class="list-copy"><b>Impact:</b> {safe_text(item.get("impact_on_recipient", ""))}</div>
        </div>
        <span class="severity {sev}">{severity}</span>
    </div>
    """


def recommendation_card(item: Dict[str, Any], number: int) -> str:
    return f"""
    <div class="recommendation">
        <div class="list-number">0{number}</div>
        <div>
            <div class="list-title">{safe_text(item.get("title", "Recommended improvement"))}</div>
            <div class="list-copy">{safe_text(item.get("explanation", ""))}</div>
        </div>
        <div class="arrow">→</div>
    </div>
    """


def results_view(result: Dict[str, Any], original: str) -> str:
    overall = clamp_score(result.get("overall_human_touch_score", 0))
    tone = score_color(overall)
    circumference = 301.59
    dash = circumference * overall / 100

    metrics = "".join([
        metric_card("Empathy", result["scores"]["empathy"], "♥"),
        metric_card("Naturalness", result["scores"]["naturalness"], "✦"),
        metric_card("Personalization", result["scores"]["personalization"], "◎"),
        metric_card("Context", result["scores"]["context_awareness"], "◈"),
        metric_card("Brand Voice", result["scores"]["brand_voice"], "≋"),
    ])

    concerns = result.get("potential_concerns", [])
    recommendations = result.get("recommended_changes", [])

    concern_html = "".join(
        concern_card(x, i + 1) for i, x in enumerate(concerns)
    )
    if not concern_html:
        concern_html = """
        <div class="empty">
            <span>✓</span><div><b>No major concerns identified.</b><br>
            The message is already showing strong human-touch signals.</div>
        </div>
        """

    recommendation_html = "".join(
        recommendation_card(x, i + 1) for i, x in enumerate(recommendations)
    )
    if not recommendation_html:
        recommendation_html = """
        <div class="empty">
            <span>✦</span><div><b>No major changes needed.</b><br>
            The current communication is relatively strong.</div>
        </div>
        """

    return f"""
    <div class="results">

        <div class="score-card">
            <div class="score-left">
                <div class="tiny-label">HUMAN TOUCH SCORE</div>
                <div class="score-title">Does it feel human?</div>
                <div class="score-summary">{safe_text(result.get("summary", ""))}</div>
                <div class="status" style="color:{tone};border-color:{tone}35;">
                    <i style="background:{tone};"></i>
                    {safe_text(result.get("verdict", ""))}
                </div>
            </div>

            <div class="score-circle">
                <svg viewBox="0 0 110 110">
                    <circle class="circle-bg" cx="55" cy="55" r="48"></circle>
                    <circle class="circle-value" cx="55" cy="55" r="48"
                            stroke="{tone}"
                            stroke-dasharray="{dash:.1f} {circumference:.1f}"></circle>
                </svg>
                <div class="circle-center">
                    <strong>{overall}</strong>
                    <span>/100</span>
                </div>
            </div>
        </div>

        <div class="metrics">{metrics}</div>

        <div class="recipient-card">
            <div class="recipient-icon">♧</div>
            <div>
                <div class="tiny-label">RECIPIENT PERSPECTIVE</div>
                <div class="recipient-title">Through their eyes</div>
                <div class="recipient-copy">
                    {safe_text(result.get("recipient_perspective", ""))}
                </div>
            </div>
        </div>

        <div class="result-section">
            <div class="section-label">01 · FRICTION</div>
            <div class="section-title">What could feel less human?</div>
            <div class="section-sub">Specific signals found in the communication.</div>
            <div class="stack">{concern_html}</div>
        </div>

        <div class="result-section">
            <div class="section-label">02 · IMPROVE</div>
            <div class="section-title">Make the connection stronger.</div>
            <div class="section-sub">Actionable changes based on the evaluation.</div>
            <div class="stack">{recommendation_html}</div>
        </div>

        <div class="result-section">
            <div class="section-label">03 · REWRITE</div>
            <div class="section-title">A more human version.</div>
            <div class="section-sub">Meaning and supported facts are preserved.</div>

            <div class="rewrite">
                <div class="message-box">
                    <div class="box-label">ORIGINAL</div>
                    <div class="message-copy">{safe_text(original).replace(chr(10), "<br>")}</div>
                </div>
                <div class="message-box improved">
                    <div class="box-label">RECOMMENDED</div>
                    <div class="message-copy">{safe_text(result.get("recommended_rewrite", "")).replace(chr(10), "<br>")}</div>
                </div>
            </div>
        </div>
    </div>
    """


# ---------- Premium mobile-inspired design ----------
st.html(r"""
<style>
:root {
    --bg: #070B17;
    --bg2: #0A1020;
    --card: rgba(16, 24, 45, .82);
    --card2: rgba(13, 20, 38, .94);
    --line: rgba(148,163,184,.13);
    --white: #F8FAFC;
    --soft: #CBD5E1;
    --muted: #8290A8;
    --dim: #526078;
    --purple: #8B5CF6;
    --blue: #38BDF8;
    --cyan: #22D3EE;
}

/* Page */
html, body, [data-testid="stAppViewContainer"] {
    background:
        radial-gradient(700px 500px at 15% -5%, rgba(124,58,237,.18), transparent 60%),
        radial-gradient(700px 500px at 100% 10%, rgba(14,165,233,.12), transparent 62%),
        linear-gradient(180deg, #080C1A 0%, #060916 100%) !important;
    color: var(--white) !important;
}
[data-testid="stHeader"] { background: transparent !important; }
.block-container {
    max-width: 1160px !important;
    padding: 18px 25px 75px !important;
}

/* Header */
.nav {
    display:flex;
    align-items:center;
    justify-content:space-between;
    padding:8px 0 18px;
    border-bottom:1px solid var(--line);
}
.brand {
    display:flex;
    align-items:center;
    gap:10px;
}
.logo {
    width:31px;height:31px;display:grid;place-items:center;
    border-radius:10px;
    color:#fff;font-size:15px;font-weight:800;
    background:linear-gradient(135deg,#6D4AFF,#16B9E8);
    box-shadow:0 7px 24px rgba(99,102,241,.25);
}
.brand-name {
    font-size:12px;font-weight:800;letter-spacing:.08em;
    text-transform:uppercase;color:#F1F5F9;
}
.brand-sub {
    color:#64748B;font-size:9px;margin-left:3px;
}
.live {
    display:flex;align-items:center;gap:7px;
    color:#71809A;font-size:9px;letter-spacing:.08em;
    text-transform:uppercase;
}
.live i {
    width:6px;height:6px;border-radius:50%;
    background:#4ADE80;box-shadow:0 0 11px #4ADE80;
}

/* Hero */
.hero {
    position:relative;
    padding:70px 0 48px;
    overflow:hidden;
}
.hero:after {
    content:"";
    position:absolute;
    width:450px;height:450px;right:-160px;top:-100px;
    border-radius:50%;
    background:radial-gradient(circle,rgba(99,102,241,.13),transparent 67%);
    pointer-events:none;
}
.hero-kicker {
    color:#A78BFA;font-size:10px;font-weight:800;
    letter-spacing:.17em;text-transform:uppercase;
}
.hero-title {
    margin-top:14px;
    max-width:850px;
    font-size:clamp(48px,7vw,82px);
    line-height:.94;
    font-weight:850;
    letter-spacing:-.065em;
}
.hero-title span {
    background:linear-gradient(90deg,#A78BFA,#38BDF8);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}
.author {
    margin-top:18px;color:#73809A;font-size:11px;
}
.hero-copy {
    max-width:680px;margin-top:18px;
    color:#8794AB;font-size:14px;line-height:1.8;
}
.hero-pill {
    display:inline-flex;align-items:center;gap:7px;
    margin-top:22px;padding:8px 12px;
    border:1px solid rgba(129,140,248,.20);
    border-radius:999px;background:rgba(99,102,241,.07);
    color:#C4B5FD;font-size:9px;font-weight:700;
}
.hero-pill i {
    width:5px;height:5px;border-radius:50%;
    background:#22D3EE;box-shadow:0 0 10px #22D3EE;
}

/* Workflow */
.workflow {
    display:grid;grid-template-columns:repeat(3,1fr);
    gap:9px;margin-bottom:32px;
}
.step {
    padding:15px 16px;border:1px solid var(--line);
    border-radius:15px;background:rgba(255,255,255,.022);
}
.step-number {
    color:#6366F1;font-size:8px;font-weight:850;letter-spacing:.14em;
}
.step-title { margin-top:6px;color:#D8DEE9;font-size:11px;font-weight:700; }
.step-copy { margin-top:4px;color:#64748B;font-size:9px; }

/* Panels */
.panel {
    border:1px solid var(--line);
    border-radius:20px;
    background:linear-gradient(145deg,rgba(17,25,47,.82),rgba(9,14,28,.90));
    box-shadow:0 25px 70px rgba(0,0,0,.18);
    padding:23px;
}
.panel-head { margin-bottom:17px; }
.panel-label {
    color:#818CF8;font-size:9px;font-weight:800;
    letter-spacing:.15em;text-transform:uppercase;
}
.panel-title {
    margin-top:6px;color:#F1F5F9;
    font-size:20px;font-weight:750;letter-spacing:-.025em;
}
.panel-copy {
    margin-top:5px;color:#71809A;font-size:10px;line-height:1.6;
}

/* Streamlit controls */
[data-baseweb="textarea"] > div,
[data-baseweb="input"] > div,
[data-baseweb="select"] > div {
    background:rgba(5,10,24,.80) !important;
    border:1px solid rgba(148,163,184,.13) !important;
    border-radius:13px !important;
}
[data-baseweb="textarea"] > div:focus-within,
[data-baseweb="input"] > div:focus-within,
[data-baseweb="select"] > div:focus-within {
    border-color:rgba(139,92,246,.50) !important;
    box-shadow:0 0 0 3px rgba(139,92,246,.08) !important;
}
textarea,input { color:#F8FAFC !important; }
textarea::placeholder,input::placeholder { color:#536078 !important; }
[data-baseweb="select"] * { color:#F8FAFC !important; }
label,[data-testid="stWidgetLabel"] p {
    color:#A7B2C5 !important;
    font-size:10px !important;
    font-weight:650 !important;
}
[data-testid="stRadio"] label { color:#9AA7BC !important; }
[data-testid="stRadio"] [role="radiogroup"] { gap:4px; }

.stButton > button {
    min-height:59px !important;
    border-radius:14px !important;
    border:1px solid rgba(167,139,250,.25) !important;
    background:linear-gradient(100deg,#7C3AED,#8B5CF6 48%,#0891B2) !important;
    color:white !important;
    font-size:14px !important;
    font-weight:800 !important;
    box-shadow:0 17px 45px rgba(99,102,241,.22) !important;
    transition:.18s ease;
}
.stButton > button:hover {
    filter:brightness(1.07);transform:translateY(-1px);
    box-shadow:0 22px 55px rgba(99,102,241,.30) !important;
}
.counter {
    color:#526078;font-size:9px;margin:-5px 0 2px;
}
.counter b { color:#A78BFA; }

/* Results */
.results { margin-top:24px; }
.result-divider {
    height:1px;background:var(--line);margin:46px 0 28px;
}
.results-title {
    text-align:center;color:#F1F5F9;font-size:30px;
    font-weight:820;letter-spacing:-.045em;
}
.score-card {
    display:grid;grid-template-columns:1fr 180px;
    gap:25px;align-items:center;
    margin-top:25px;padding:29px;
    border:1px solid rgba(129,140,248,.18);
    border-radius:21px;
    background:
        radial-gradient(circle at 85% 50%,rgba(99,102,241,.12),transparent 35%),
        linear-gradient(145deg,rgba(17,25,47,.88),rgba(9,14,28,.93));
}
.tiny-label {
    color:#7C89A3;font-size:8px;font-weight:800;
    letter-spacing:.15em;
}
.score-title {
    margin-top:8px;color:#F8FAFC;font-size:35px;
    line-height:1.05;font-weight:820;letter-spacing:-.045em;
}
.score-summary {
    max-width:650px;margin-top:12px;color:#8390A6;
    font-size:11px;line-height:1.75;
}
.status {
    display:inline-flex;align-items:center;gap:7px;
    margin-top:16px;padding:7px 10px;border:1px solid;
    border-radius:999px;background:rgba(255,255,255,.025);
    font-size:9px;font-weight:750;
}
.status i { width:5px;height:5px;border-radius:50%;box-shadow:0 0 9px currentColor; }

.score-circle { width:160px;height:160px;position:relative;margin:auto; }
.score-circle svg { width:160px;height:160px;transform:rotate(-90deg); }
.circle-bg { fill:none;stroke:rgba(255,255,255,.07);stroke-width:6; }
.circle-value { fill:none;stroke-width:6;stroke-linecap:round; }
.circle-center {
    position:absolute;inset:0;display:flex;flex-direction:column;
    align-items:center;justify-content:center;
}
.circle-center strong {
    color:#F8FAFC;font-size:42px;font-weight:850;
    letter-spacing:-.07em;line-height:1;
}
.circle-center span { color:#68758D;font-size:9px;margin-top:5px; }

/* Metrics */
.metrics {
    display:grid;grid-template-columns:repeat(5,1fr);
    gap:8px;margin-top:9px;
}
.metric {
    padding:17px;border:1px solid var(--line);
    border-radius:15px;background:rgba(255,255,255,.022);
}
.metric-icon { color:#A78BFA;font-size:12px; }
.metric-label { margin-top:8px;color:#8996AA;font-size:9px;font-weight:650; }
.metric-score { margin-top:6px;color:#F1F5F9;font-size:25px;font-weight:800; }
.metric-bar { height:3px;margin-top:11px;border-radius:99px;background:rgba(255,255,255,.06);overflow:hidden; }
.metric-bar span { display:block;height:100%;border-radius:inherit; }

/* Recipient */
.recipient-card {
    display:grid;grid-template-columns:40px 1fr;gap:13px;
    margin-top:9px;padding:23px 25px;
    border:1px solid rgba(167,139,250,.15);
    border-radius:16px;
    background:linear-gradient(135deg,rgba(99,102,241,.08),rgba(14,165,233,.035));
}
.recipient-icon {
    width:36px;height:36px;display:grid;place-items:center;
    border-radius:11px;background:rgba(167,139,250,.10);
    color:#A78BFA;font-size:17px;
}
.recipient-title { margin-top:5px;color:#EEF2FF;font-size:15px;font-weight:730; }
.recipient-copy { max-width:900px;margin-top:8px;color:#9AA7BC;font-size:11px;line-height:1.75; }

/* Report sections */
.result-section { margin-top:43px; }
.section-label { color:#818CF8;font-size:8px;font-weight:850;letter-spacing:.16em; }
.section-title { margin-top:6px;color:#EDF2F7;font-size:22px;font-weight:780;letter-spacing:-.03em; }
.section-sub { margin-top:4px;color:#68758B;font-size:9px; }
.stack { margin-top:15px; }

.list-card,.recommendation {
    display:grid;grid-template-columns:40px 1fr auto;gap:13px;
    padding:17px 18px;margin:7px 0;
    border:1px solid var(--line);border-radius:14px;
    background:rgba(255,255,255,.018);
}
.list-number { color:#56647B;font-size:9px;font-weight:850;letter-spacing:.10em; }
.issue-head { display:flex;justify-content:space-between;gap:12px; }
.list-title { color:#E2E8F0;font-size:11px;font-weight:730; }
.list-meta { margin-top:3px;color:#64748B;font-size:8px; }
.list-copy { margin-top:10px;color:#748198;font-size:9px;line-height:1.65; }
.list-copy b { color:#A3AEC0; }
.severity {
    align-self:start;padding:4px 7px;border-radius:999px;
    font-size:7px;font-weight:800;text-transform:uppercase;
}
.severity.low { color:#4ADE80;background:rgba(74,222,128,.07); }
.severity.medium { color:#FBBF24;background:rgba(251,191,36,.07); }
.severity.high { color:#FB7185;background:rgba(251,113,133,.07); }
.recommendation { align-items:start; }
.recommend-copy { margin-top:6px;color:#748198;font-size:9px;line-height:1.65; }
.arrow { color:#71809A;font-size:14px; }

/* Rewrite */
.rewrite {
    display:grid;grid-template-columns:1fr 1fr;
    gap:9px;margin-top:15px;
}
.message-box {
    min-height:220px;padding:19px;
    border:1px solid var(--line);border-radius:15px;
    background:rgba(5,10,20,.72);
}
.message-box.improved {
    border-color:rgba(74,222,128,.15);
    background:radial-gradient(circle at 100% 0%,rgba(74,222,128,.05),transparent 45%),rgba(5,10,20,.72);
}
.box-label { color:#65728A;font-size:8px;font-weight:850;letter-spacing:.15em; }
.message-copy { margin-top:11px;color:#B4BFCE;font-size:10px;line-height:1.8; }

/* Empty / footer */
.empty {
    display:flex;gap:10px;align-items:center;
    padding:17px;border:1px dashed rgba(255,255,255,.10);
    border-radius:13px;color:#6F7C91;font-size:9px;line-height:1.6;
}
.empty span { color:#4ADE80;font-size:15px; }
.empty b { color:#AEB8C8; }

.footer {
    margin-top:58px;padding-top:18px;border-top:1px solid var(--line);
    display:flex;justify-content:space-between;gap:20px;
    color:#4F5C71;font-size:8px;line-height:1.6;
}

@media(max-width:900px) {
    .hero { padding-top:55px; }
    .workspace { grid-template-columns:1fr !important; }
    .metrics { grid-template-columns:repeat(2,1fr); }
    .score-card { grid-template-columns:1fr; }
    .score-circle { order:-1; }
}
@media(max-width:650px) {
    .block-container { padding:12px 12px 55px !important; }
    .hero-title { font-size:51px; }
    .workflow { grid-template-columns:1fr; }
    .metrics { grid-template-columns:1fr; }
    .rewrite { grid-template-columns:1fr; }
    .score-card,.panel { padding:20px; }
    .footer { flex-direction:column; }
}
</style>
""")

# ---------- Header ----------
st.html("""
<div class="nav">
    <div class="brand">
        <div class="logo">✦</div>
        <div class="brand-name">Human Touch</div>
        <div class="brand-sub">Quality Layer</div>
    </div>
    <div class="live"><i></i> AI communication intelligence</div>
</div>
""")

# ---------- Hero ----------
st.html("""
<div class="hero">
    <div class="hero-kicker">AI CAN WRITE. HUMAN TOUCH DECIDES.</div>
    <div class="hero-title">
        But does it<br><span>feel human?</span>
    </div>
    <div class="author">By Engr. Muhammad Mubashir Asim</div>
    <div class="hero-copy">
        Evaluate messages for empathy, naturalness, personalization,
        context-awareness, and brand voice — before they reach another person.
    </div>
    <div class="hero-pill"><i></i> Human intelligence layer active</div>
</div>
""")

# ---------- Workflow ----------
st.html("""
<div class="workflow">
    <div class="step">
        <div class="step-number">01 · MESSAGE</div>
        <div class="step-title">Write or paste</div>
        <div class="step-copy">Start with the communication.</div>
    </div>
    <div class="step">
        <div class="step-number">02 · CONTEXT</div>
        <div class="step-title">Add recipient context</div>
        <div class="step-copy">Give the situation meaning.</div>
    </div>
    <div class="step">
        <div class="step-number">03 · ANALYZE</div>
        <div class="step-title">Get human-touch insights</div>
        <div class="step-copy">Score, improve, and rewrite.</div>
    </div>
</div>
""")

# ---------- Input ----------
st.html("""
<div class="panel">
    <div class="panel-head">
        <div class="panel-label">MESSAGE</div>
        <div class="panel-title">Your message</div>
        <div class="panel-copy">Paste the communication exactly as the recipient would see it.</div>
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
    placeholder="Type or paste your message here…",
    height=250,
    label_visibility="collapsed",
)

st.html(
    f'<div class="counter"><b>{len(message or "")}</b> characters · '
    'communication quality analysis only</div>'
)

st.html("</div>")

st.write("")
st.html("""
<div class="panel">
    <div class="panel-label">CONTEXT</div>
    <div class="panel-title">Who is on the other side?</div>
    <div class="panel-copy">A little context helps the model judge whether the message fits the situation.</div>
</div>
""")

left, right = st.columns(2, gap="medium")
with left:
    audience = st.text_input("Audience / Recipient", placeholder="Existing customer")
    purpose = st.text_input("Purpose", placeholder="Confirmation")
with right:
    brand_voice = st.text_input("Brand Voice", placeholder="Warm, professional")
    additional_context = st.text_input(
        "Additional Context",
        placeholder="Relevant situation or prior interaction…",
    )

st.write("")
analyze = st.button("✦  Analyze Human Touch  →", type="primary", use_container_width=True)

st.html("""
<div style="text-align:center;margin-top:8px;color:#56647B;font-size:8px;">
    Your message is evaluated for communication quality — not treated as a definitive AI detector.
</div>
""")

# ---------- Analyze ----------
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
    <div class="result-divider"></div>
    <div class="results-title">Analysis Results</div>
    """)
    st.html(results_view(st.session_state.result, st.session_state.analyzed_message))

# ---------- Footer ----------
st.html("""
<div class="footer">
    <div>HUMAN TOUCH · QUALITY LAYER</div>
    <div>Communication quality, recipient perspective, human connection.</div>
</div>
""")
