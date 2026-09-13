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


def build_results(result: Dict[str, Any], original_message: str) -> str:
    scores = result["scores"]

    concerns = result.get("potential_concerns", [])
    recommendations = result.get("recommended_changes", [])

    concern_html = "".join(concern_card(c) for c in concerns)

    recommendation_html = "".join(
        recommendation_card(r, i + 1)
        for i, r in enumerate(recommendations)
    )

    if not concern_html:
        concern_html = """
        <div class="insight">
            No major human-touch concerns were identified.
        </div>
        """

    if not recommendation_html:
        recommendation_html = """
        <div class="insight">
            The message is already relatively strong.
        </div>
        """

    return f"""
    <div class="score-hero">
        <div class="score-number">
            {result["overall_human_touch_score"]}
        </div>

        <div class="score-label">
            Human-Touch Score / 100
        </div>

        <div class="verdict">
            {safe_text(result["verdict"])}
        </div>
    </div>

    <br>

    <div class="section-title">Communication Intelligence</div>

    <div class="insight">
        <div class="insight-label">Summary</div>
        <div style="margin-top:10px;line-height:1.7;">
            {safe_text(result.get("summary", ""))}
        </div>
    </div>

    <div class="section-title" style="margin-top:28px;">
        Human-Touch Dimensions
    </div>

    <br>

    <div style="
        display:grid;
        grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
        gap:14px;
    ">
        {score_card("Empathy", scores["empathy"])}
        {score_card("Naturalness", scores["naturalness"])}
        {score_card("Personalization", scores["personalization"])}
        {score_card("Context Awareness", scores["context_awareness"])}
        {score_card("Brand Voice", scores["brand_voice"])}
    </div>

    <br>

    <div class="section-title">
        Through the Recipient's Eyes
    </div>

    <div class="insight">
        <div style="line-height:1.7;">
            {safe_text(result.get("recipient_perspective", ""))}
        </div>
    </div>

    <br>

    <div class="section-title">
        What Could Feel Less Human?
    </div>

    {concern_html}

    <br>

    <div class="section-title">
        How to Make It More Human
    </div>

    {recommendation_html}

    <br>

    <div class="section-title">
        Recommended Rewrite
    </div>

    <div style="color:#8f9bb3;margin:8px 0 18px;">
        Original vs. improved human-touch version
    </div>

    <div style="
        display:grid;
        grid-template-columns:repeat(auto-fit,minmax(300px,1fr));
        gap:16px;
    ">
        <div class="original-box">
            <div class="box-label">Original</div>
            {safe_text(original_message)}
        </div>

        <div class="rewrite-box">
            <div class="box-label">Human-Touch Rewrite</div>
            {safe_text(result.get("recommended_rewrite", ""))}
        </div>
    </div>
    """

# ============================================================
# HUMAN TOUCH — SIGNATURE PRODUCT UI
# ============================================================

def score_color(score: int) -> str:
    if score >= 85:
        return "#34d399"
    if score >= 70:
        return "#a78bfa"
    if score >= 55:
        return "#fbbf24"
    return "#fb7185"


def verdict_class(score: int) -> str:
    if score >= 85:
        return "good"
    if score >= 70:
        return "strong"
    if score >= 55:
        return "refine"
    return "weak"


def result_section_title(kicker: str, title: str, description: str = "") -> str:
    return f"""
    <div class="result-heading">
        <div class="result-kicker">{safe_text(kicker)}</div>
        <div class="result-title">{safe_text(title)}</div>
        {f'<div class="result-description">{safe_text(description)}</div>' if description else ""}
    </div>
    """


def dimension_card_v2(name: str, score: int, icon: str) -> str:
    score = clamp_score(score)
    return f"""
    <div class="metric-card">
        <div class="metric-top">
            <span class="metric-icon">{icon}</span>
            <span class="metric-name">{safe_text(name)}</span>
        </div>
        <div class="metric-value">{score}<span>/100</span></div>
        <div class="metric-track">
            <div class="metric-progress" style="width:{score}%; background:{score_color(score)};"></div>
        </div>
        <div class="metric-foot">
            <span>{'Strong' if score >= 80 else 'Healthy' if score >= 65 else 'Needs work'}</span>
            <span>{score}%</span>
        </div>
    </div>
    """


def concern_card_v2(item: Dict[str, Any], number: int) -> str:
    issue = safe_text(item.get("issue", "Potential concern"))
    dimension = safe_text(item.get("dimension", ""))
    evidence = safe_text(item.get("evidence", ""))
    impact = safe_text(item.get("impact_on_recipient", ""))
    severity = safe_text(item.get("severity", "Low"))
    severity_cls = {"High": "high", "Medium": "medium", "Low": "low"}.get(severity, "low")
    return f"""
    <div class="issue-card">
        <div class="issue-index">0{number}</div>
        <div class="issue-main">
            <div class="issue-row">
                <div class="issue-title">{issue}</div>
                <span class="severity {severity_cls}">{severity}</span>
            </div>
            <div class="issue-dimension">{dimension}</div>
            <div class="issue-detail"><b>Evidence</b><br>{evidence}</div>
            <div class="issue-detail"><b>Recipient impact</b><br>{impact}</div>
        </div>
    </div>
    """


def recommendation_card_v2(item: Dict[str, Any], number: int) -> str:
    title = safe_text(item.get("title", "Recommended improvement"))
    explanation = safe_text(item.get("explanation", ""))
    return f"""
    <div class="recommend-card">
        <div class="recommend-number">{number:02d}</div>
        <div>
            <div class="recommend-title">{title}</div>
            <div class="recommend-copy">{explanation}</div>
        </div>
    </div>
    """


def build_results_v2(result: Dict[str, Any], original_message: str) -> str:
    scores = result["scores"]
    overall = clamp_score(result["overall_human_touch_score"])
    concerns = result.get("potential_concerns", [])
    recommendations = result.get("recommended_changes", [])
    rewrite = result.get("recommended_rewrite", "")

    concern_html = "".join(
        concern_card_v2(c, i + 1) for i, c in enumerate(concerns)
    )
    if not concern_html:
        concern_html = """
        <div class="empty-state">
            <span class="empty-icon">✓</span>
            <div><b>No major human-touch concerns</b><br>
            <span>The message is already showing strong recipient awareness.</span></div>
        </div>
        """

    recommendation_html = "".join(
        recommendation_card_v2(r, i + 1)
        for i, r in enumerate(recommendations)
    )
    if not recommendation_html:
        recommendation_html = """
        <div class="empty-state">
            <span class="empty-icon">✦</span>
            <div><b>No major changes recommended</b><br>
            <span>The current communication is relatively strong.</span></div>
        </div>
        """

    circumference = 2 * 3.14159 * 48
    dash = circumference * overall / 100
    verdict_cls = verdict_class(overall)

    metrics = [
        ("Empathy", scores["empathy"], "♡"),
        ("Naturalness", scores["naturalness"], "◌"),
        ("Personalization", scores["personalization"], "◎"),
        ("Context Awareness", scores["context_awareness"], "⌁"),
        ("Brand Voice", scores["brand_voice"], "◈"),
    ]
    metric_html = "".join(
        dimension_card_v2(name, score, icon) for name, score, icon in metrics
    )

    return f"""
    <div class="report-shell">
        <div class="score-panel">
            <div class="score-copy">
                <div class="result-kicker">COMMUNICATION INTELLIGENCE</div>
                <div class="score-heading">How human does this message feel?</div>
                <div class="score-summary">{safe_text(result.get("summary", ""))}</div>
                <div class="verdict-pill {verdict_cls}">
                    <span class="pulse-dot"></span>{safe_text(result.get("verdict", ""))}
                </div>
            </div>
            <div class="score-ring-wrap">
                <svg class="score-ring" viewBox="0 0 120 120">
                    <circle class="ring-bg" cx="60" cy="60" r="48"></circle>
                    <circle class="ring-value" cx="60" cy="60" r="48"
                        stroke-dasharray="{dash:.1f} {circumference:.1f}"></circle>
                </svg>
                <div class="score-center">
                    <div class="score-number">{overall}</div>
                    <div class="score-denom">/ 100</div>
                </div>
            </div>
        </div>

        <div class="metrics-grid">{metric_html}</div>

        <div class="insight-grid">
            <div class="insight-panel">
                <div class="panel-label">RECIPIENT PERSPECTIVE</div>
                <div class="panel-title">Through their eyes</div>
                <div class="panel-copy">{safe_text(result.get("recipient_perspective", ""))}</div>
            </div>
            <div class="insight-panel accent-panel">
                <div class="panel-label">EXECUTIVE READOUT</div>
                <div class="panel-title">The signal</div>
                <div class="panel-copy">{safe_text(result.get("summary", ""))}</div>
            </div>
        </div>

        <div class="report-block">
            {result_section_title("01 · FRICTION MAP", "What could feel less human?",
                "Specific evidence that may weaken the recipient experience.")}
            <div class="issue-list">{concern_html}</div>
        </div>

        <div class="report-block">
            {result_section_title("02 · IMPROVEMENT PLAN", "How to make it more human",
                "Practical changes grounded in the message and supplied context.")}
            <div class="recommend-list">{recommendation_html}</div>
        </div>

        <div class="report-block">
            {result_section_title("03 · REWRITE", "A stronger human-touch version",
                "Meaning and factual claims are preserved; unsupported information is never added.")}
            <div class="rewrite-grid">
                <div class="copy-panel">
                    <div class="copy-label">ORIGINAL</div>
                    <div class="copy-text">{safe_text(original_message)}</div>
                </div>
                <div class="copy-panel rewrite-panel">
                    <div class="copy-label">RECOMMENDED REWRITE</div>
                    <div class="copy-text">{safe_text(rewrite)}</div>
                </div>
            </div>
        </div>
    </div>
    """


st.set_page_config(
    page_title="Human Touch · Quality Layer",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS_V2 = """
<style>
:root {
    --bg: #050816;
    --surface: rgba(13,18,38,.76);
    --line: rgba(255,255,255,.09);
    --text: #f6f7fb;
    --muted: #8993aa;
}
html, body, [data-testid="stAppViewContainer"] {
    background:
        radial-gradient(900px 500px at 8% -5%, rgba(98,80,245,.17), transparent 60%),
        radial-gradient(700px 450px at 96% 12%, rgba(28,190,224,.11), transparent 60%),
        linear-gradient(180deg,#070b1d 0%,#050816 55%,#040610 100%) !important;
    color: var(--text) !important;
}
[data-testid="stHeader"] { background: transparent !important; }
.block-container { max-width: 1240px !important; padding: 26px 28px 80px !important; }

.hero-v2 {
    position: relative; overflow: hidden; text-align: center; padding: 72px 30px 54px;
    border: 1px solid rgba(255,255,255,.07); border-radius: 32px;
    background: linear-gradient(135deg,rgba(255,255,255,.035),rgba(255,255,255,.012)),
                radial-gradient(circle at 50% 0%,rgba(139,124,246,.13),transparent 48%);
    box-shadow: 0 35px 100px rgba(0,0,0,.28);
}
.hero-v2:before {
    content:""; position:absolute; width:520px; height:1px; left:50%; top:0;
    transform:translateX(-50%); background:linear-gradient(90deg,transparent,#8b7cf6,#42d9ee,transparent);
}
.eyebrow { color:#a9a2ff; font-size:11px; font-weight:800; letter-spacing:.20em; text-transform:uppercase; }
.hero-title-v2 {
    margin-top:15px; font-size:clamp(48px,7vw,78px); line-height:.98; letter-spacing:-.055em; font-weight:850;
    background:linear-gradient(100deg,#fff 15%,#c9c2ff 58%,#71e7f5 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.hero-product-v2 { margin-top:12px; font-size:18px; font-weight:650; color:#c3c9da; }
.hero-author-v2 { margin-top:9px; color:#77839d; font-size:13px; letter-spacing:.035em; }
.hero-tagline-v2 { margin:28px auto 0; max-width:760px; font-size:clamp(21px,3vw,29px); line-height:1.3; font-weight:650; letter-spacing:-.025em; }
.hero-description-v2 { max-width:720px; margin:15px auto 0; color:#8c96ac; font-size:15px; line-height:1.75; }
.status-v2 {
    display:inline-flex; align-items:center; gap:8px; margin-top:24px; padding:9px 15px;
    border-radius:999px; color:#bceff5; background:rgba(66,217,238,.055);
    border:1px solid rgba(66,217,238,.16); font-size:12px; font-weight:650;
}
.live-dot { width:7px; height:7px; border-radius:50%; background:#42d9ee; box-shadow:0 0 14px #42d9ee; }

.flow-v2 { display:grid; grid-template-columns:1fr auto 1fr auto 1fr; align-items:center; gap:13px; margin:25px 0 32px; }
.flow-step { padding:16px 18px; border-radius:16px; border:1px solid rgba(255,255,255,.075); background:rgba(255,255,255,.025); }
.flow-num { color:#8f84ff; font-size:10px; font-weight:850; letter-spacing:.14em; }
.flow-name { margin-top:5px; color:#d9deea; font-size:13px; font-weight:650; }
.flow-arrow { color:#68728a; font-size:20px; }

.section-card-v2 {
    padding:27px; border-radius:24px; border:1px solid var(--line); background:var(--surface);
    box-shadow:0 24px 70px rgba(0,0,0,.17); margin-bottom:18px;
}
.card-kicker { color:#9289ff; font-size:10px; font-weight:850; letter-spacing:.16em; }
.card-title { margin-top:7px; color:#f4f5fa; font-size:22px; font-weight:750; letter-spacing:-.025em; }
.card-subtitle { margin-top:5px; color:#808ba3; font-size:13px; line-height:1.6; }

[data-baseweb="textarea"] > div,[data-baseweb="input"] > div,[data-baseweb="select"] > div {
    background:rgba(4,8,22,.78) !important; border:1px solid rgba(255,255,255,.085) !important;
    border-radius:14px !important; box-shadow:inset 0 1px 0 rgba(255,255,255,.025) !important;
}
[data-baseweb="textarea"] > div:focus-within,[data-baseweb="input"] > div:focus-within,[data-baseweb="select"] > div:focus-within {
    border-color:rgba(139,124,246,.55) !important; box-shadow:0 0 0 3px rgba(139,124,246,.08) !important;
}
textarea,input { color:#f5f7fb !important; }
textarea::placeholder,input::placeholder { color:#566179 !important; }
[data-baseweb="select"] * { color:#f5f7fb !important; }
label,[data-testid="stWidgetLabel"] p,[data-testid="stRadio"] label { color:#aeb7c9 !important; font-size:12px !important; font-weight:600 !important; }

.stButton > button {
    min-height:62px !important; border-radius:16px !important; border:1px solid rgba(255,255,255,.13) !important;
    background:linear-gradient(100deg,#7568f5 0%,#956cf4 48%,#18aeca 100%) !important;
    color:white !important; font-size:15px !important; font-weight:800 !important;
    box-shadow:0 18px 45px rgba(90,75,240,.23) !important;
}
.stButton > button:hover { filter:brightness(1.07); transform:translateY(-2px); }

.char-count { color:#59657d; font-size:11px; margin:-7px 0 18px; }
.char-count b { color:#a39aff; }
.disclaimer-v2 { margin-top:18px; text-align:center; color:#59647b; font-size:11px; line-height:1.6; }

.report-shell { margin-top:20px; }
.score-panel {
    display:grid; grid-template-columns:1fr 190px; align-items:center; gap:30px; padding:34px;
    border-radius:26px; border:1px solid rgba(139,124,246,.18);
    background:radial-gradient(circle at 75% 40%,rgba(139,124,246,.12),transparent 35%),
               linear-gradient(135deg,rgba(255,255,255,.045),rgba(255,255,255,.015));
    box-shadow:0 30px 90px rgba(0,0,0,.22);
}
.result-kicker { color:#8f84ff; font-size:10px; font-weight:850; letter-spacing:.18em; }
.score-heading { margin-top:10px; font-size:clamp(25px,4vw,38px); line-height:1.08; font-weight:780; letter-spacing:-.04em; }
.score-summary { max-width:650px; margin-top:13px; color:#8e99b0; line-height:1.7; font-size:14px; }
.verdict-pill { display:inline-flex; align-items:center; gap:8px; margin-top:18px; padding:8px 13px; border-radius:999px; font-size:12px; font-weight:700; border:1px solid rgba(255,255,255,.08); }
.verdict-pill.good { color:#a7f3d0; background:rgba(52,211,153,.08); }
.verdict-pill.strong { color:#ddd6fe; background:rgba(167,139,250,.08); }
.verdict-pill.refine { color:#fde68a; background:rgba(251,191,36,.07); }
.verdict-pill.weak { color:#fecdd3; background:rgba(251,113,133,.07); }
.pulse-dot { width:6px; height:6px; border-radius:50%; background:currentColor; box-shadow:0 0 10px currentColor; }

.score-ring-wrap { position:relative; width:170px; height:170px; margin:auto; }
.score-ring { width:170px; height:170px; transform:rotate(-90deg); }
.ring-bg { fill:none; stroke:rgba(255,255,255,.07); stroke-width:7; }
.ring-value { fill:none; stroke:#9b8cff; stroke-width:7; stroke-linecap:round; filter:drop-shadow(0 0 7px rgba(139,124,246,.45)); }
.score-center { position:absolute; inset:0; display:flex; flex-direction:column; justify-content:center; align-items:center; }
.score-number { font-size:43px; line-height:1; font-weight:850; letter-spacing:-.06em; }
.score-denom { margin-top:5px; color:#69748c; font-size:11px; }

.metrics-grid { display:grid; grid-template-columns:repeat(5,1fr); gap:12px; margin-top:13px; }
.metric-card { padding:19px; border:1px solid rgba(255,255,255,.075); border-radius:18px; background:rgba(255,255,255,.025); }
.metric-top { display:flex; align-items:center; gap:8px; }
.metric-icon { color:#a69cff; font-size:14px; }
.metric-name { color:#9ca6ba; font-size:11px; font-weight:650; }
.metric-value { margin-top:13px; font-size:28px; font-weight:800; letter-spacing:-.04em; }
.metric-value span { color:#566179; font-size:10px; font-weight:600; margin-left:2px; }
.metric-track { height:5px; margin-top:13px; border-radius:99px; background:rgba(255,255,255,.07); overflow:hidden; }
.metric-progress { height:100%; border-radius:inherit; }
.metric-foot { display:flex; justify-content:space-between; margin-top:8px; color:#566179; font-size:10px; }

.insight-grid { display:grid; grid-template-columns:1fr 1fr; gap:13px; margin-top:13px; }
.insight-panel { min-height:175px; padding:24px; border-radius:20px; border:1px solid rgba(255,255,255,.075); background:rgba(255,255,255,.025); }
.accent-panel { background:linear-gradient(135deg,rgba(139,124,246,.07),rgba(66,217,238,.025)); border-color:rgba(139,124,246,.13); }
.panel-label { color:#8179d8; font-size:9px; font-weight:850; letter-spacing:.16em; }
.panel-title { margin-top:9px; color:#eef0f7; font-size:17px; font-weight:720; }
.panel-copy { margin-top:10px; color:#8e99b0; line-height:1.75; font-size:13px; }

.report-block { margin-top:34px; }
.result-heading { margin-bottom:16px; }
.result-title { margin-top:7px; font-size:26px; font-weight:760; letter-spacing:-.035em; }
.result-description { margin-top:5px; color:#737f97; font-size:13px; }

.issue-card,.recommend-card {
    display:grid; grid-template-columns:42px 1fr; gap:15px; padding:19px; margin:10px 0;
    border-radius:17px; border:1px solid rgba(255,255,255,.07); background:rgba(255,255,255,.023);
}
.issue-index,.recommend-number { color:#7269c9; font-size:11px; font-weight:850; letter-spacing:.08em; }
.issue-row { display:flex; align-items:center; justify-content:space-between; gap:12px; }
.issue-title { color:#e8ebf3; font-size:14px; font-weight:720; }
.issue-dimension { margin-top:4px; color:#6f7a92; font-size:10px; }
.issue-detail { margin-top:11px; color:#7f8aa1; font-size:12px; line-height:1.65; }
.issue-detail b { color:#b4bdcd; font-weight:650; }
.severity { padding:4px 8px; border-radius:999px; font-size:9px; font-weight:750; border:1px solid rgba(255,255,255,.07); }
.severity.high { color:#fda4af; background:rgba(251,113,133,.07); }
.severity.medium { color:#fcd34d; background:rgba(251,191,36,.07); }
.severity.low { color:#86efac; background:rgba(52,211,153,.07); }

.recommend-title { color:#e9ecf4; font-size:14px; font-weight:720; }
.recommend-copy { margin-top:7px; color:#7f8aa1; font-size:12px; line-height:1.65; }
.empty-state {
    display:flex; align-items:center; gap:13px; padding:18px; border-radius:16px;
    border:1px dashed rgba(255,255,255,.09); color:#8b96ab; font-size:12px; line-height:1.55;
}
.empty-state b { color:#d8ddea; }
.empty-icon { width:30px; height:30px; display:grid; place-items:center; border-radius:9px; background:rgba(139,124,246,.08); color:#a39aff; }

.rewrite-grid { display:grid; grid-template-columns:1fr 1fr; gap:13px; }
.copy-panel { min-height:210px; padding:21px; border-radius:18px; border:1px solid rgba(255,255,255,.075); background:rgba(255,255,255,.023); }
.rewrite-panel { background:linear-gradient(135deg,rgba(139,124,246,.08),rgba(66,217,238,.035)); border-color:rgba(139,124,246,.15); }
.copy-label { color:#707b92; font-size:9px; font-weight:850; letter-spacing:.16em; }
.copy-text { margin-top:12px; color:#bfc6d5; line-height:1.75; font-size:13px; white-space:pre-wrap; }

.footer-v2 { margin-top:58px; padding-top:22px; border-top:1px solid rgba(255,255,255,.06); text-align:center; color:#505b71; font-size:11px; line-height:1.7; }
.footer-v2 b { color:#7e88a0; font-weight:650; }

@media (max-width:900px) {
    .metrics-grid { grid-template-columns:repeat(2,1fr); }
    .score-panel { grid-template-columns:1fr; }
    .score-ring-wrap { order:-1; }
    .insight-grid,.rewrite-grid { grid-template-columns:1fr; }
}
@media (max-width:650px) {
    .block-container { padding:14px 12px 55px !important; }
    .hero-v2 { padding:48px 18px 38px; border-radius:24px; }
    .hero-title-v2 { font-size:48px; }
    .flow-v2 { grid-template-columns:1fr; }
    .flow-arrow { display:none; }
    .metrics-grid { grid-template-columns:1fr; }
    .section-card-v2 { padding:20px; }
    .score-panel { padding:22px; }
}
</style>
"""

st.html(CUSTOM_CSS_V2)

if "result" not in st.session_state:
    st.session_state.result = None
if "analyzed_message" not in st.session_state:
    st.session_state.analyzed_message = ""

st.html("""
<div class="hero-v2">
    <div class="eyebrow">Communication Quality Intelligence</div>
    <div class="hero-title-v2">Human Touch</div>
    <div class="hero-product-v2">Quality Layer</div>
    <div class="hero-author-v2">By Engr. Muhammad Mubashir Asim</div>
    <div class="hero-tagline-v2">
        AI can write the message.<br>
        We measure whether it feels human.
    </div>
    <div class="hero-description-v2">
        A communication intelligence layer that evaluates the recipient experience
        before an email, support reply, business message, or social post reaches another person.
    </div>
    <div class="status-v2"><span class="live-dot"></span>Human Intelligence Layer Active</div>
</div>
""")

st.html("""
<div class="flow-v2">
    <div class="flow-step"><div class="flow-num">01</div><div class="flow-name">Message</div></div>
    <div class="flow-arrow">→</div>
    <div class="flow-step"><div class="flow-num">02</div><div class="flow-name">Recipient context</div></div>
    <div class="flow-arrow">→</div>
    <div class="flow-step"><div class="flow-num">03</div><div class="flow-name">Human-touch analysis</div></div>
</div>
""")

st.html("""
<div class="section-card-v2">
    <div class="card-kicker">STEP 01</div>
    <div class="card-title">Bring the message</div>
    <div class="card-subtitle">Paste the communication exactly as it would be sent.</div>
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
    placeholder="Paste the message you want to evaluate…",
    height=240,
    label_visibility="collapsed",
)

st.html(
    f'<div class="char-count"><b>{len(message or "")}</b> characters · '
    f'Only the supplied content is evaluated</div>'
)

st.html("""
<div class="section-card-v2">
    <div class="card-kicker">STEP 02</div>
    <div class="card-title">Add the human context</div>
    <div class="card-subtitle">
        Give the evaluator enough context to judge whether the message fits the person,
        purpose, and brand.
    </div>
</div>
""")

left, right = st.columns(2, gap="large")
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
<div class="disclaimer-v2">
    Communication quality analysis — <b>not</b> a definitive AI detector.
</div>
""")

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
                if any(token in error_text.lower() for token in ["401", "authentication", "api key", "unauthorized"]):
                    st.error("Groq authentication failed. Check GROQ_API_KEY in Streamlit Secrets.")
                elif "rate" in error_text.lower() and "limit" in error_text.lower():
                    st.error("Groq rate limit reached. Please wait a moment and try again.")
                elif "model" in error_text.lower() and (
                    "not found" in error_text.lower() or "does not exist" in error_text.lower()
                ):
                    st.error(
                        f"Groq model error. Current model: {get_model()}. "
                        "Set GROQ_MODEL to a currently supported Groq model in Streamlit Secrets."
                    )
                else:
                    st.error(f"Evaluation failed: {error_text}")

if st.session_state.result:
    result = st.session_state.result
    st.html("""
    <div style="height:1px;background:rgba(255,255,255,.06);margin:42px 0 34px;"></div>
    <div class="eyebrow" style="text-align:center;">ANALYSIS COMPLETE</div>
    <div style="text-align:center;font-size:32px;font-weight:800;letter-spacing:-.04em;margin-top:7px;">
        Your Human-Touch Report
    </div>
    """)
    st.html(build_results_v2(result, st.session_state.analyzed_message))

st.html("""
<div class="footer-v2">
    <b>Human Touch Quality Layer</b><br>
    Evaluates communication quality from the recipient's perspective.
    It does not determine whether content was written by a human or AI.
</div>
""")
