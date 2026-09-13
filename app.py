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
# HUMAN TOUCH — SIGNATURE EDITORIAL CONSOLE
# ============================================================

st.set_page_config(
    page_title="Human Touch · Quality Layer",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- Result presentation ----------
def score_tone(score: int) -> str:
    score = clamp_score(score)
    if score >= 85:
        return "#C8F36B"
    if score >= 70:
        return "#A99BFF"
    if score >= 55:
        return "#F2C66D"
    return "#F27F91"


def metric_card(label: str, score: Any, code: str) -> str:
    score = clamp_score(score)
    tone = score_tone(score)
    return f"""
    <div class="metric-card">
        <div class="metric-code">{safe_text(code)}</div>
        <div class="metric-name">{safe_text(label)}</div>
        <div class="metric-value" style="color:{tone};">{score}<span>/100</span></div>
        <div class="metric-track"><i style="width:{score}%;background:{tone};"></i></div>
    </div>
    """


def issue_card(item: Dict[str, Any], number: int) -> str:
    severity = safe_text(item.get("severity", "Low"))
    sev = severity.lower() if severity.lower() in {"low", "medium", "high"} else "low"
    return f"""
    <div class="issue-card">
        <div class="index">0{number}</div>
        <div>
            <div class="issue-head">
                <div>
                    <div class="issue-title">{safe_text(item.get("issue", "Potential concern"))}</div>
                    <div class="issue-dim">{safe_text(item.get("dimension", ""))}</div>
                </div>
                <span class="severity {sev}">{severity}</span>
            </div>
            <div class="issue-detail"><b>Evidence</b>{safe_text(item.get("evidence", ""))}</div>
            <div class="issue-detail"><b>Recipient impact</b>{safe_text(item.get("impact_on_recipient", ""))}</div>
        </div>
    </div>
    """


def recommendation_card(item: Dict[str, Any], number: int) -> str:
    return f"""
    <div class="recommend-card">
        <div class="index">0{number}</div>
        <div>
            <div class="recommend-title">{safe_text(item.get("title", "Recommended improvement"))}</div>
            <div class="recommend-copy">{safe_text(item.get("explanation", ""))}</div>
        </div>
        <div class="recommend-arrow">↗</div>
    </div>
    """


def results_html(result: Dict[str, Any], original: str) -> str:
    overall = clamp_score(result.get("overall_human_touch_score", 0))
    scores = result["scores"]
    tone = score_tone(overall)

    metrics = "".join([
        metric_card("Empathy", scores["empathy"], "01"),
        metric_card("Naturalness", scores["naturalness"], "02"),
        metric_card("Personalization", scores["personalization"], "03"),
        metric_card("Context Awareness", scores["context_awareness"], "04"),
        metric_card("Brand Voice", scores["brand_voice"], "05"),
    ])

    concerns = result.get("potential_concerns", [])
    recommendations = result.get("recommended_changes", [])

    issue_html = "".join(issue_card(x, i + 1) for i, x in enumerate(concerns))
    if not issue_html:
        issue_html = """
        <div class="empty-state"><span>✓</span><div><b>No major friction detected.</b><br>
        The message is already showing strong recipient awareness.</div></div>
        """

    recommendation_html = "".join(
        recommendation_card(x, i + 1) for i, x in enumerate(recommendations)
    )
    if not recommendation_html:
        recommendation_html = """
        <div class="empty-state"><span>✦</span><div><b>No major intervention required.</b><br>
        The current communication is relatively strong.</div></div>
        """

    circumference = 326.73
    dash = circumference * overall / 100

    return f"""
    <div class="result-hero">
        <div>
            <div class="eyebrow">QUALITY SIGNAL · ANALYSIS COMPLETE</div>
            <div class="result-title">Would this feel considered?</div>
            <div class="result-summary">{safe_text(result.get("summary", ""))}</div>
            <div class="verdict" style="color:{tone};border-color:{tone}38;">
                <i style="background:{tone};"></i>{safe_text(result.get("verdict", ""))}
            </div>
        </div>

        <div class="score-ring">
            <svg viewBox="0 0 120 120">
                <circle class="ring-track" cx="60" cy="60" r="52"></circle>
                <circle class="ring-progress" cx="60" cy="60" r="52"
                    stroke="{tone}"
                    stroke-dasharray="{dash:.1f} {circumference:.1f}"></circle>
            </svg>
            <div class="ring-center">
                <strong>{overall}</strong>
                <span>HUMAN TOUCH</span>
            </div>
        </div>
    </div>

    <div class="metrics-grid">{metrics}</div>

    <div class="perspective">
        <div class="quote">“</div>
        <div>
            <div class="eyebrow">RECIPIENT PERSPECTIVE</div>
            <div class="perspective-copy">{safe_text(result.get("recipient_perspective", ""))}</div>
        </div>
    </div>

    <div class="result-section">
        <div class="section-marker">01</div>
        <div class="eyebrow">FRICTION MAP</div>
        <div class="section-title">What could weaken the human feeling?</div>
        <div class="section-desc">Specific evidence from the message and supplied context.</div>
        <div class="stack">{issue_html}</div>
    </div>

    <div class="result-section">
        <div class="section-marker">02</div>
        <div class="eyebrow">IMPROVEMENT PLAN</div>
        <div class="section-title">Small changes. Stronger connection.</div>
        <div class="section-desc">Practical recommendations grounded in the evaluation.</div>
        <div class="stack">{recommendation_html}</div>
    </div>

    <div class="result-section">
        <div class="section-marker">03</div>
        <div class="eyebrow">REWRITE LAB</div>
        <div class="section-title">Keep the meaning. Improve the signal.</div>
        <div class="section-desc">No unsupported personal details or factual claims are added.</div>

        <div class="rewrite-grid">
            <div class="copy-box">
                <div class="copy-label">ORIGINAL</div>
                <div class="copy-text">{safe_text(original).replace(chr(10), "<br>")}</div>
            </div>
            <div class="copy-box improved">
                <div class="copy-label">RECOMMENDED REWRITE</div>
                <div class="copy-text">{safe_text(result.get("recommended_rewrite", "")).replace(chr(10), "<br>")}</div>
            </div>
        </div>
    </div>
    """


# ---------- Design system ----------
st.html(r"""
<style>
:root {
    --bg: #080a08;
    --panel: #0e110f;
    --panel-soft: rgba(255,255,255,.025);
    --line: rgba(255,255,255,.085);
    --text: #f2f4ef;
    --muted: #858d86;
    --muted2: #555d57;
    --lime: #c8f36b;
    --violet: #a99bff;
}

/* GLOBAL */
html, body, [data-testid="stAppViewContainer"] {
    background:
        radial-gradient(900px 520px at 4% -8%, rgba(200,243,107,.055), transparent 62%),
        radial-gradient(900px 580px at 103% 4%, rgba(169,155,255,.07), transparent 62%),
        #080a08 !important;
    color: var(--text) !important;
}
[data-testid="stHeader"] { background: transparent !important; }
.block-container {
    max-width: 1280px !important;
    padding: 22px 32px 85px !important;
}

/* BRAND BAR */
.brandbar {
    display:flex;
    justify-content:space-between;
    align-items:center;
    padding: 8px 0 17px;
    border-bottom:1px solid var(--line);
}
.brand-left { display:flex; align-items:center; gap:10px; }
.brand-symbol {
    width:29px;height:29px;display:grid;place-items:center;
    border:1px solid rgba(200,243,107,.30);
    border-radius:8px;color:var(--lime);font-size:13px;font-weight:850;
}
.brand-name {
    color:#e9ece6;font-size:11px;font-weight:850;
    letter-spacing:.10em;text-transform:uppercase;
}
.brand-sub { color:#4e564f;font-size:9px;letter-spacing:.07em;margin-left:3px; }
.system-state {
    color:#707870;font-size:9px;letter-spacing:.12em;text-transform:uppercase;
    display:flex;align-items:center;gap:7px;
}
.system-state i {
    width:6px;height:6px;border-radius:50%;background:var(--lime);
    box-shadow:0 0 12px rgba(200,243,107,.8);
}

/* HERO */
.hero {
    display:grid;
    grid-template-columns:minmax(0,1.2fr) minmax(310px,.8fr);
    gap:70px;
    align-items:end;
    padding:82px 0 58px;
}
.eyebrow {
    color:#7b847c;font-size:9px;font-weight:850;
    letter-spacing:.18em;text-transform:uppercase;
}
.hero h1 {
    margin:16px 0 0;
    font-size:clamp(54px,7.5vw,98px);
    line-height:.88;
    letter-spacing:-.07em;
    font-weight:850;
}
.hero h1 span { color:#737c74; }
.author { margin-top:20px;color:#687169;font-size:11px;letter-spacing:.035em; }
.hero-right {
    max-width:445px;margin-left:auto;color:#89918a;
    font-size:13px;line-height:1.9;
}
.hero-right strong { color:#c4cbc4;font-weight:650; }
.hero-rule { margin-top:20px;padding-top:15px;border-top:1px solid var(--line);color:#555d56;font-size:9px; }

/* FLOW */
.flow {
    display:grid;grid-template-columns:repeat(3,1fr);
    border-top:1px solid var(--line);border-bottom:1px solid var(--line);
}
.flow-item { padding:17px 19px;border-right:1px solid var(--line); }
.flow-item:last-child { border-right:0; }
.flow-num { color:#4e5650;font-size:8px;font-weight:850;letter-spacing:.14em; }
.flow-title { margin-top:7px;color:#cfd4ce;font-size:11px;font-weight:700; }
.flow-copy { margin-top:4px;color:#606860;font-size:9px;line-height:1.55; }

/* COMPOSE */
.compose-label {
    margin-top:45px;margin-bottom:12px;
    color:#555e57;font-size:9px;font-weight:850;letter-spacing:.18em;
}
.workspace {
    display:grid;grid-template-columns:1.3fr .7fr;gap:11px;
}
.panel {
    border:1px solid var(--line);border-radius:18px;
    background:linear-gradient(145deg,rgba(255,255,255,.032),rgba(255,255,255,.012));
    padding:24px;
}
.panel-kicker { color:var(--lime);font-size:8px;font-weight:850;letter-spacing:.16em; }
.panel-title { margin-top:7px;color:#e8ebe6;font-size:19px;font-weight:770;letter-spacing:-.025em; }
.panel-copy { margin-top:5px;color:#6f786f;font-size:10px;line-height:1.6; }

[data-baseweb="textarea"] > div,
[data-baseweb="input"] > div,
[data-baseweb="select"] > div {
    background:#090c0a !important;
    border:1px solid rgba(255,255,255,.08) !important;
    border-radius:12px !important;
}
[data-baseweb="textarea"] > div:focus-within,
[data-baseweb="input"] > div:focus-within,
[data-baseweb="select"] > div:focus-within {
    border-color:rgba(200,243,107,.35) !important;
    box-shadow:0 0 0 3px rgba(200,243,107,.045) !important;
}
textarea,input { color:#eef1ec !important; }
textarea::placeholder,input::placeholder { color:#4b534d !important; }
[data-baseweb="select"] * { color:#eef1ec !important; }
label,[data-testid="stWidgetLabel"] p {
    color:#858e86 !important;font-size:10px !important;font-weight:700 !important;
}
[data-testid="stRadio"] label { color:#858e86 !important; }
[data-testid="stRadio"] [role="radiogroup"] { gap:3px; }
.char-count { margin:-5px 0 2px;color:#505850;font-size:9px; }
.char-count b { color:#a9b09e; }

/* CTA */
.stButton > button {
    min-height:58px !important;
    border-radius:12px !important;
    border:1px solid rgba(200,243,107,.28) !important;
    background:var(--lime) !important;
    color:#10140b !important;
    font-size:12px !important;font-weight:850 !important;
    box-shadow:0 15px 40px rgba(170,210,70,.10) !important;
}
.stButton > button:hover {
    filter:brightness(1.04);transform:translateY(-1px);
    box-shadow:0 18px 48px rgba(170,210,70,.16) !important;
}
.disclaimer {
    text-align:center;margin-top:8px;color:#4e564f;font-size:9px;
}

/* RESULTS */
.report-top {
    margin-top:58px;padding-top:25px;border-top:1px solid var(--line);
    text-align:center;
}
.report-title-main {
    margin-top:7px;color:#eef1eb;font-size:31px;font-weight:820;
    letter-spacing:-.045em;
}
.result-hero {
    display:grid;grid-template-columns:1fr 190px;gap:35px;align-items:center;
    margin-top:30px;padding:31px;border:1px solid var(--line);border-radius:21px;
    background:
        radial-gradient(circle at 85% 50%,rgba(200,243,107,.055),transparent 31%),
        linear-gradient(145deg,rgba(255,255,255,.033),rgba(255,255,255,.012));
}
.result-title {
    margin-top:9px;color:#edf0eb;font-size:clamp(27px,4vw,40px);
    font-weight:820;letter-spacing:-.045em;line-height:1.04;
}
.result-summary { max-width:680px;margin-top:13px;color:#7f887f;font-size:12px;line-height:1.75; }
.verdict {
    display:inline-flex;align-items:center;gap:7px;margin-top:17px;
    padding:7px 10px;border:1px solid;border-radius:999px;
    background:rgba(255,255,255,.018);font-size:9px;font-weight:750;
}
.verdict i { width:5px;height:5px;border-radius:50%;box-shadow:0 0 9px currentColor; }

.score-ring { width:170px;height:170px;position:relative;margin:auto; }
.score-ring svg { width:170px;height:170px;transform:rotate(-90deg); }
.ring-track { fill:none;stroke:rgba(255,255,255,.065);stroke-width:5; }
.ring-progress { fill:none;stroke-width:5;stroke-linecap:round; }
.ring-center {
    position:absolute;inset:0;display:flex;flex-direction:column;
    align-items:center;justify-content:center;
}
.ring-center strong { color:#f2f4ef;font-size:42px;line-height:1;font-weight:850;letter-spacing:-.06em; }
.ring-center span { margin-top:7px;color:#4f5851;font-size:7px;font-weight:850;letter-spacing:.14em; }

.metrics-grid { display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-top:9px; }
.metric-card {
    padding:17px;border:1px solid var(--line);border-radius:14px;background:rgba(255,255,255,.018);
}
.metric-code { color:#464e48;font-size:8px;font-weight:850;letter-spacing:.12em; }
.metric-name { margin-top:10px;color:#858e86;font-size:10px;font-weight:650; }
.metric-value { margin-top:7px;font-size:27px;font-weight:830;letter-spacing:-.05em; }
.metric-value span { color:#4b534d;font-size:8px;font-weight:600; }
.metric-track { height:3px;margin-top:12px;border-radius:99px;background:rgba(255,255,255,.05);overflow:hidden; }
.metric-track i { display:block;height:100%;border-radius:inherit; }

.perspective {
    display:grid;grid-template-columns:53px 1fr;gap:14px;margin-top:9px;
    padding:25px 28px;border:1px solid rgba(169,155,255,.12);
    border-radius:16px;background:linear-gradient(120deg,rgba(169,155,255,.045),rgba(255,255,255,.012));
}
.quote { color:#8e84df;font:65px/0.6 Georgia,serif; }
.perspective-copy { max-width:920px;margin-top:9px;color:#b2bab2;font-size:13px;line-height:1.8; }

.result-section { margin-top:43px; }
.result-section + .result-section { margin-top:48px; }
.section-marker { color:#454d47;font-size:9px;font-weight:850;margin-bottom:8px; }
.section-title { margin-top:6px;color:#e4e8e2;font-size:22px;font-weight:780;letter-spacing:-.035em; }
.section-desc { margin-top:4px;color:#626a63;font-size:9px; }
.stack { margin-top:16px; }

.issue-card,.recommend-card {
    display:grid;grid-template-columns:42px 1fr auto;gap:14px;
    padding:18px 19px;margin:7px 0;border:1px solid var(--line);
    border-radius:14px;background:rgba(255,255,255,.017);
}
.issue-card { grid-template-columns:42px 1fr; }
.index { color:#4b534d;font-size:9px;font-weight:850;letter-spacing:.12em; }
.issue-head { display:flex;justify-content:space-between;gap:15px; }
.issue-title,.recommend-title { color:#dce1db;font-size:12px;font-weight:750; }
.issue-dim { margin-top:4px;color:#626b63;font-size:8px; }
.issue-detail {
    margin-top:11px;color:#737c74;font-size:10px;line-height:1.65;
}
.issue-detail b {
    display:block;margin-bottom:2px;color:#9ba39c;font-size:8px;
    letter-spacing:.08em;text-transform:uppercase;
}
.severity {
    align-self:start;padding:4px 7px;border-radius:999px;
    font-size:7px;font-weight:850;text-transform:uppercase;
}
.severity.low { color:#68d8ac;background:rgba(69,211,166,.06); }
.severity.medium { color:#e8c46a;background:rgba(242,198,109,.06); }
.severity.high { color:#f18494;background:rgba(242,127,145,.06); }
.recommend-copy { margin-top:6px;color:#727b73;font-size:10px;line-height:1.65; }
.recommend-arrow { color:#68716a;font-size:15px; }
.empty-state {
    display:flex;align-items:center;gap:10px;padding:17px;
    border:1px dashed rgba(255,255,255,.09);border-radius:13px;
    color:#69726b;font-size:10px;line-height:1.6;
}
.empty-state span { color:var(--lime);font-size:16px; }
.empty-state b { color:#aab1ab; }

.rewrite-grid { display:grid;grid-template-columns:1fr 1fr;gap:9px;margin-top:17px; }
.copy-box {
    min-height:225px;padding:20px;border:1px solid var(--line);
    border-radius:15px;background:#0b0e0c;
}
.copy-box.improved {
    border-color:rgba(200,243,107,.13);
    background:radial-gradient(circle at 100% 0%,rgba(200,243,107,.045),transparent 48%),#0b0e0c;
}
.copy-label { color:#59615a;font-size:8px;font-weight:850;letter-spacing:.15em; }
.copy-text { margin-top:12px;color:#aeb6af;font-size:11px;line-height:1.8; }

.footer {
    margin-top:62px;padding-top:19px;border-top:1px solid var(--line);
    display:flex;justify-content:space-between;gap:20px;
    color:#4b534d;font-size:8px;letter-spacing:.04em;line-height:1.6;
}

@media (max-width: 950px) {
    .hero,.workspace { grid-template-columns:1fr; }
    .hero-right { margin-left:0;max-width:650px; }
    .metrics-grid { grid-template-columns:repeat(2,1fr); }
    .result-hero { grid-template-columns:1fr; }
    .score-ring { order:-1; }
}
@media (max-width: 650px) {
    .block-container { padding:15px 13px 55px !important; }
    .hero { padding:52px 0 40px;gap:30px; }
    .hero h1 { font-size:55px; }
    .flow { grid-template-columns:1fr; }
    .flow-item { border-right:0;border-bottom:1px solid var(--line); }
    .flow-item:last-child { border-bottom:0; }
    .metrics-grid { grid-template-columns:1fr; }
    .rewrite-grid { grid-template-columns:1fr; }
    .panel,.result-hero { padding:20px; }
    .footer { flex-direction:column; }
}
</style>
""")

# ---------- Header ----------
st.html("""
<div class="brandbar">
    <div class="brand-left">
        <div class="brand-symbol">✦</div>
        <div class="brand-name">Human Touch</div>
        <div class="brand-sub">/ Quality Layer</div>
    </div>
    <div class="system-state"><i></i> Intelligence layer active</div>
</div>
""")

# ---------- Hero ----------
st.html("""
<div class="hero">
    <div>
        <div class="eyebrow">Communication quality intelligence</div>
        <h1>Make every message<br><span>feel considered.</span></h1>
        <div class="author">By Engr. Muhammad Mubashir Asim</div>
    </div>
    <div class="hero-right">
        AI can produce fluent language instantly. <strong>Human Touch</strong>
        evaluates what happens after the words are generated: whether another
        person would feel understood, respected, and thoughtfully addressed.
        <div class="hero-rule">
            Recipient experience · Context awareness · Human-touch quality
        </div>
    </div>
</div>
""")

# ---------- Flow ----------
st.html("""
<div class="flow">
    <div class="flow-item">
        <div class="flow-num">01 · MESSAGE</div>
        <div class="flow-title">Bring the communication</div>
        <div class="flow-copy">Paste the exact message you intend to send.</div>
    </div>
    <div class="flow-item">
        <div class="flow-num">02 · CONTEXT</div>
        <div class="flow-title">Describe the situation</div>
        <div class="flow-copy">Recipient, purpose, brand voice, and useful context.</div>
    </div>
    <div class="flow-item">
        <div class="flow-num">03 · QUALITY SIGNAL</div>
        <div class="flow-title">See it through their eyes</div>
        <div class="flow-copy">Score, friction, recommendations, and rewrite.</div>
    </div>
</div>
""")

# ---------- Compose ----------
st.html("""
<div class="compose-label">COMPOSE / EVALUATE</div>
<div class="workspace">
    <div class="panel">
        <div class="panel-kicker">MESSAGE</div>
        <div class="panel-title">What are you about to send?</div>
        <div class="panel-copy">Use the message exactly as the recipient would receive it.</div>
    </div>
    <div class="panel">
        <div class="panel-kicker">CONTEXT</div>
        <div class="panel-title">Who is on the other side?</div>
        <div class="panel-copy">Context gives the quality layer something human to reason about.</div>
    </div>
</div>
""")

# Inputs live immediately under their visual panels.
content_type = st.radio(
    "Communication type",
    ["Email", "Customer Service", "Business", "Social Post", "Other"],
    index=0,
    horizontal=True,
)

message = st.text_area(
    "Message",
    placeholder="Start with the message you want a second pair of human eyes on…",
    height=270,
    label_visibility="collapsed",
)

st.html(
    f'<div class="char-count"><b>{len(message or "")}</b> characters · '
    'evaluated for communication quality only</div>'
)

left, right = st.columns(2, gap="medium")
with left:
    audience = st.text_input("Audience / Recipient", placeholder="Existing customer")
    purpose = st.text_input("Purpose", placeholder="Apologize for a delayed delivery")
with right:
    brand_voice = st.text_input("Brand Voice", placeholder="Warm, professional, concise")
    additional_context = st.text_input(
        "Additional Context",
        placeholder="The customer has already waited 10 days.",
    )

st.write("")
analyze = st.button("✦  Analyze Human Touch", type="primary", use_container_width=True)

st.html("""
<div class="disclaimer">
    Quality analysis from the recipient's perspective · not a definitive AI detector
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
    <div class="report-top">
        <div class="eyebrow">ANALYSIS COMPLETE</div>
        <div class="report-title-main">Your communication quality report</div>
    </div>
    """)
    st.html(results_html(st.session_state.result, st.session_state.analyzed_message))

# ---------- Footer ----------
st.html("""
<div class="footer">
    <div>HUMAN TOUCH QUALITY LAYER · COMMUNICATION QUALITY INTELLIGENCE</div>
    <div>Does not determine whether content was written by a human or AI.</div>
</div>
""")
