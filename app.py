import json
import html
import os
import textwrap
from typing import Any, Dict, List

import streamlit as st
from groq import Groq


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Human Touch Quality Layer",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# Streamlit treats heavily-indented HTML passed to render_markdown() as a
# code block. Dedent HTML before rendering so the premium UI displays
# as intended instead of showing raw <div> tags.
def render_markdown(content, **kwargs):
    """Render the app's HTML directly so Streamlit never shows raw HTML tags."""
    if isinstance(content, str):
        content = textwrap.dedent(content)
    kwargs.pop("unsafe_allow_html", None)
    return st.html(content)


# ============================================================
# PREMIUM DARK UI
# ============================================================
CUSTOM_CSS = """
<style>
:root {
    --bg: #070b18;
    --card: rgba(15,23,42,.72);
    --card-strong: rgba(15,23,42,.88);
    --border: rgba(148,163,184,.12);
    --text: #f5f7ff;
    --muted: #9ca3b8;
    --indigo: #6366f1;
    --violet: #8b5cf6;
    --cyan: #22d3ee;
}

html, body, [data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 15% 0%, rgba(99,102,241,.16), transparent 32%),
        radial-gradient(circle at 90% 10%, rgba(139,92,246,.12), transparent 30%),
        #070b18 !important;
    color: var(--text) !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

.block-container {
    max-width: 1180px !important;
    margin: auto !important;
    padding: 35px 24px 70px !important;
}

/* HERO */
.hero {
    text-align: center;
    padding: 55px 20px 35px;
}

.hero-title {
    font-size: 58px;
    font-weight: 800;
    letter-spacing: -2px;
    margin: 0;
    color: #f8fafc;
}

.hero-product {
    font-size: 19px;
    color: #a5b4fc;
    font-weight: 600;
    margin-top: 5px;
}

.hero-author {
    font-size: 14px;
    color: #8f9bb3;
    font-weight: 500;
    margin-top: 9px;
    letter-spacing: .02em;
}

.hero-tagline {
    font-size: 24px;
    font-weight: 600;
    margin-top: 25px;
    color: #f5f7ff;
}

.hero-description {
    max-width: 720px;
    margin: 15px auto;
    color: #9ca3b8;
    font-size: 16px;
    line-height: 1.7;
}

.status-pill {
    display: inline-block;
    margin-top: 18px;
    padding: 9px 17px;
    border-radius: 999px;
    background: rgba(99,102,241,.12);
    border: 1px solid rgba(129,140,248,.25);
    color: #c7d2fe;
    font-size: 13px;
}

/* WORKFLOW */
.workflow {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 14px;
    margin: 25px 0 35px;
    flex-wrap: wrap;
}

.workflow-step {
    padding: 11px 18px;
    border-radius: 12px;
    background: rgba(255,255,255,.035);
    border: 1px solid rgba(255,255,255,.08);
    color: #cbd5e1;
}

.workflow-arrow {
    color: #818cf8;
    font-size: 20px;
}

/* CARDS */
.saas-card {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 22px !important;
    padding: 28px !important;
    margin-bottom: 22px !important;
    box-shadow: 0 20px 60px rgba(0,0,0,.18);
}

.section-title {
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 5px;
    color: #f5f7ff;
}

.section-description {
    color: #8f9bb3;
    margin-bottom: 20px;
}

/* STREAMLIT INPUTS */
[data-baseweb="textarea"] > div,
[data-baseweb="input"] > div,
[data-baseweb="select"] > div {
    background: rgba(2,6,23,.7) !important;
    border-color: rgba(148,163,184,.16) !important;
    border-radius: 14px !important;
}

textarea,
input {
    color: #f8fafc !important;
}

textarea::placeholder,
input::placeholder {
    color: #64748b !important;
}

[data-baseweb="select"] * {
    color: #f8fafc !important;
}

label, [data-testid="stWidgetLabel"] p {
    color: #cbd5e1 !important;
}

/* RADIO */
[data-testid="stRadio"] label {
    color: #cbd5e1 !important;
}

/* BUTTON */
.stButton > button {
    min-height: 58px !important;
    border-radius: 15px !important;
    font-size: 17px !important;
    font-weight: 700 !important;
    border: 1px solid rgba(129,140,248,.2) !important;
    background: linear-gradient(135deg,#6366f1,#8b5cf6,#0891b2) !important;
    color: white !important;
    box-shadow: 0 12px 35px rgba(79,70,229,.22);
}

.stButton > button:hover {
    filter: brightness(1.08);
    transform: translateY(-1px);
}

/* SCORE */
.score-hero {
    text-align: center;
    padding: 35px;
    border-radius: 22px;
    background:
        radial-gradient(circle at center, rgba(99,102,241,.16), transparent 60%),
        rgba(15,23,42,.7);
    border: 1px solid rgba(129,140,248,.16);
}

.score-number {
    font-size: 72px;
    font-weight: 800;
    line-height: 1;
    color: #f8fafc;
}

.score-label {
    color: #9ca3b8;
    margin-top: 10px;
}

.verdict {
    display: inline-block;
    margin-top: 15px;
    padding: 8px 16px;
    border-radius: 999px;
    background: rgba(99,102,241,.14);
    color: #c7d2fe;
}

/* DIMENSIONS */
.dimension-card {
    background: rgba(255,255,255,.025);
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 17px;
    padding: 20px;
    min-height: 125px;
}

.dimension-name {
    color: #aab4c8;
    font-size: 14px;
}

.dimension-score {
    font-size: 29px;
    font-weight: 750;
    margin: 8px 0;
    color: #f8fafc;
}

.dimension-bar {
    height: 6px;
    background: rgba(255,255,255,.07);
    border-radius: 99px;
    overflow: hidden;
}

.dimension-fill {
    height: 100%;
    background: linear-gradient(90deg,#6366f1,#8b5cf6,#22d3ee);
    border-radius: 99px;
}

/* INSIGHTS */
.insight,
.concern,
.recommendation {
    background: rgba(255,255,255,.025);
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 16px;
    padding: 20px;
    margin: 10px 0;
    color: #e2e8f0;
}

.insight-label {
    color: #a5b4fc;
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .08em;
}

.concern-title {
    font-weight: 700;
    margin-bottom: 8px;
    color: #f1f5f9;
}

.concern-evidence,
.concern-impact {
    color: #aab4c8;
    line-height: 1.6;
    margin-top: 7px;
}

.recommendation-number {
    color: #818cf8;
    font-weight: 800;
    margin-right: 8px;
}

/* REWRITE */
.original-box,
.rewrite-box {
    padding: 22px;
    border-radius: 17px;
    line-height: 1.7;
    white-space: pre-wrap;
    min-height: 180px;
}

.original-box {
    background: rgba(255,255,255,.025);
    border: 1px solid rgba(255,255,255,.07);
}

.rewrite-box {
    background: rgba(99,102,241,.07);
    border: 1px solid rgba(129,140,248,.2);
}

.box-label {
    color: #9ca3b8;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: .08em;
    margin-bottom: 10px;
    font-weight: 700;
}

/* MOBILE */
@media (max-width: 700px) {
    .hero-title { font-size: 42px; }
    .hero-tagline { font-size: 20px; }
    .block-container { padding: 20px 14px 50px !important; }
    .saas-card { padding: 20px !important; }
}
</style>
"""

render_markdown(CUSTOM_CSS, unsafe_allow_html=True)


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
    return os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()


def score_card(name: str, score: Any) -> str:
    score = clamp_score(score)
    return f"""
    <div class="dimension-card">
        <div class="dimension-name">{safe_text(name)}</div>
        <div class="dimension-score">{score}</div>
        <div class="dimension-bar">
            <div class="dimension-fill" style="width:{score}%"></div>
        </div>
    </div>
    """


def concern_card(item: Dict[str, Any]) -> str:
    issue = safe_text(item.get("issue", "Potential concern"))
    dimension = safe_text(item.get("dimension", ""))
    evidence = safe_text(item.get("evidence", ""))
    impact = safe_text(item.get("impact_on_recipient", ""))
    severity = safe_text(item.get("severity", "Low"))

    return f"""
    <div class="concern">
        <div class="concern-title">
            {issue}
            <span style="color:#818cf8;">· {dimension}</span>
        </div>
        <div class="concern-evidence">
            <b>Evidence:</b> {evidence}
        </div>
        <div class="concern-impact">
            <b>Recipient impact:</b> {impact}
        </div>
        <div style="margin-top:10px;color:#94a3b8;font-size:12px;">
            Severity: {severity}
        </div>
    </div>
    """


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
# SESSION STATE
# ============================================================
if "result" not in st.session_state:
    st.session_state.result = None

if "analyzed_message" not in st.session_state:
    st.session_state.analyzed_message = ""


# ============================================================
# HERO
# ============================================================
render_markdown(
    """
    <div class="hero">

        <div class="hero-title">
            Human Touch
        </div>

        <div class="hero-product">
            Quality Layer
        </div>

        <div class="hero-author">
            Engr. Muhammad Mubashir Asim
        </div>

        <div class="hero-tagline">
            AI can write the message.
            We measure whether it feels human.
        </div>

        <div class="hero-description">
            Evaluate AI-generated or human-written communication
            for empathy, naturalness, personalization, context-awareness,
            and brand voice before it reaches another person.
        </div>

        <div class="status-pill">
            ● Human Intelligence Layer Active
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# WORKFLOW
# ============================================================
render_markdown(
    """
    <div class="workflow">

        <div class="workflow-step">
            <b>01</b> — Message
        </div>

        <div class="workflow-arrow">→</div>

        <div class="workflow-step">
            <b>02</b> — Recipient Context
        </div>

        <div class="workflow-arrow">→</div>

        <div class="workflow-step">
            <b>03</b> — Human-Touch Analysis
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MESSAGE CARD
# ============================================================
render_markdown(
    """
    <div class="saas-card">
        <div class="section-title">Your Message</div>
        <div class="section-description">
            Paste the communication you want to evaluate.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

content_type = st.radio(
    "Communication type",
    ["Email", "Customer Service", "Business", "Social Post", "Other"],
    index=0,
    horizontal=True,
)

message = st.text_area(
    "Message",
    placeholder="Paste the message you want to evaluate…",
    height=250,
    label_visibility="collapsed",
)

render_markdown(
    f'<div style="color:#737f98;font-size:13px;margin:-10px 0 22px;">'
    f'<b style="color:#a5b4fc;">{len(message or "")}</b> characters'
    f'</div>',
    unsafe_allow_html=True,
)


# ============================================================
# CONTEXT CARD
# ============================================================
render_markdown(
    """
    <div class="saas-card">
        <div class="section-title">
            Help us understand the human on the other side.
        </div>
        <div class="section-description">
            Context helps the quality layer judge whether the message
            actually fits the person and situation.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns(2)

with left:
    audience = st.text_input(
        "Audience / Recipient",
        placeholder="Existing customer",
    )

    purpose = st.text_input(
        "Purpose",
        placeholder="Apologize for a delayed delivery",
    )

with right:
    brand_voice = st.text_input(
        "Brand Voice",
        placeholder="Warm, professional, concise",
    )

    additional_context = st.text_input(
        "Additional Context",
        placeholder="The customer has already waited 10 days.",
    )


# ============================================================
# CTA
# ============================================================
analyze = st.button(
    "✦  Analyze Human Touch",
    type="primary",
    use_container_width=True,
)

render_markdown(
    """
    <div style="
        text-align:center;
        color:#737f98;
        font-size:13px;
        margin:8px 0 25px;
    ">
        Your message is analyzed for communication quality —
        not treated as a definitive AI detector.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# ANALYSIS
# ============================================================
if analyze:
    if not message.strip():
        st.error("Please paste a message before analyzing.")
        st.session_state.result = None
    else:
        with st.spinner("Understanding the recipient's perspective…"):
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

                if any(
                    token in error_text.lower()
                    for token in ["401", "authentication", "api key", "unauthorized"]
                ):
                    st.error(
                        "Groq authentication failed. Please check that your "
                        "GROQ_API_KEY in Streamlit Secrets is correct."
                    )
                elif "rate" in error_text.lower() and "limit" in error_text.lower():
                    st.error(
                        "Groq rate limit reached. Please wait a moment and try again."
                    )
                elif "model" in error_text.lower() and (
                    "not found" in error_text.lower()
                    or "does not exist" in error_text.lower()
                ):
                    st.error(
                        f"Groq model error. Current model: {get_model()}. "
                        "Check the model name in Streamlit Secrets."
                    )
                else:
                    # Temporary diagnostic message so deployment problems are visible.
                    st.error(f"Evaluation failed: {error_text}")


# ============================================================
# RESULTS
# ============================================================
if st.session_state.result:
    result = st.session_state.result

    render_markdown(
        '<div class="section-title" style="margin-top:35px;">Your Human-Touch Report</div>',
        unsafe_allow_html=True,
    )

    render_markdown(
        '<div class="saas-card">',
        unsafe_allow_html=True,
    )

    render_markdown(
        build_results(
            result,
            st.session_state.analyzed_message,
        ),
        unsafe_allow_html=True,
    )

    render_markdown("</div>", unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================
render_markdown(
    """
    <div style="
        text-align:center;
        margin-top:35px;
        padding-top:20px;
        border-top:1px solid rgba(148,163,184,.08);
        color:#64748b;
        font-size:12px;
        line-height:1.6;
    ">
        Human Touch Quality Layer evaluates communication quality.
        It does not determine whether content was written by a human or AI.
    </div>
    """,
    unsafe_allow_html=True,
)
