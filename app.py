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
# SIMPLE DARK UI
# ============================================================

st.html("""
<style>
#MainMenu, footer {visibility:hidden;}
header[data-testid="stHeader"] {background:transparent;}
[data-testid="stAppViewContainer"] {background:#0d0f14;color:#f4f4f5;}
[data-testid="stSidebar"] {background:#25262f;border-right:1px solid #363741;}
[data-testid="stSidebarContent"] {padding:28px 25px;}
[data-testid="stSidebar"] h2 {color:#f4f4f5;font-size:23px;}
.brand-title{font-size:28px;font-weight:800;letter-spacing:-1px;}
.brand-subtitle{color:#a7a8b1;font-size:14px;margin-top:2px;}
.author{color:#c8c9d0;font-size:12px;margin-top:14px;}
.line{height:1px;background:#41424c;margin:28px 0;}
.main-wrap{max-width:1050px;margin:0 auto;padding:70px 48px 70px;}
.intro{color:#9699a4;font-size:19px;line-height:1.65;margin-bottom:70px;}
.section-title{font-size:31px;font-weight:800;letter-spacing:-.7px;}
.small-note{color:#777a85;font-size:13px;margin:8px 0 14px;}
.result-line{height:1px;background:#292c34;margin:55px 0 42px;}
.score-card,.panel,.metric{background:#171920;border:1px solid #2b2e37;border-radius:14px;}
.score-card{padding:27px;min-height:150px;}
.score-number{font-size:55px;font-weight:850;line-height:1;}
.score-label{font-size:11px;letter-spacing:1.6px;color:#777a85;margin-top:8px;}
.verdict{color:#bfc1c9;font-size:15px;margin-top:13px;}
.panel{padding:23px;}
.panel-label{font-size:11px;letter-spacing:1.5px;color:#777a85;font-weight:700;margin-bottom:9px;}
.panel-text{font-size:14px;color:#b8bac4;line-height:1.7;}
.metric{padding:17px 18px;margin:10px 0;}
.metric-top{display:flex;justify-content:space-between;color:#d6d7dd;font-size:14px;margin-bottom:9px;}
.metric-value{font-weight:800;color:#9b8cff;}
.track{height:5px;background:#2d2f38;border-radius:10px;overflow:hidden;}
.fill{height:100%;background:#8d7cff;border-radius:10px;}
.concern{padding:18px 0;border-bottom:1px solid #2b2e37;}
.concern:last-child{border-bottom:0;}
.concern-title{font-weight:750;margin-bottom:6px;}
.concern-text{font-size:13px;color:#a9abb5;line-height:1.6;}
.badge{display:inline-block;background:#2a2c35;color:#c4c6ce;border-radius:20px;padding:4px 9px;font-size:10px;margin-top:8px;}
.rewrite{background:#111319;border:1px solid #2b2e37;border-radius:12px;padding:20px;color:#c9cad1;line-height:1.7;font-size:14px;white-space:pre-wrap;}
@media(max-width:800px){.main-wrap{padding:35px 18px 50px}.intro{font-size:16px;margin-bottom:42px}.section-title{font-size:25px}}
</style>
""")

with st.sidebar:
    st.html("""
    <div class="brand-title">Human Touch</div>
    <div class="brand-subtitle">Quality Layer</div>
    <div class="author">By Engr. Muhammad Mubashir Asim</div>
    <div class="line"></div>
    """)

    st.subheader("Evaluation Settings")
    st.selectbox("AI Model", [get_model()], disabled=True,
                 help="Configured through GROQ_MODEL in Streamlit Secrets.")

    st.markdown("<div class='line'></div>", unsafe_allow_html=True)
    st.subheader("Communication Context")
    content_type = st.selectbox(
        "Content Type",
        ["Email", "Customer-service response", "Business message", "Social-media post", "Other"],
    )
    audience = st.text_input("Audience / Recipient", placeholder="Customer, client, colleague...")
    purpose = st.text_input("Purpose", placeholder="What should this message achieve?")
    brand_voice = st.text_input("Brand Voice", placeholder="Warm, professional, direct...")
    additional_context = st.text_area(
        "Additional Context",
        placeholder="Relevant details the evaluator should know...",
        height=100,
    )

st.html("<div class='main-wrap'>")
st.html("<div class='intro'>An AI-powered communication quality-control layer that evaluates whether your message feels human, empathetic, natural, personalized, context-aware, and aligned with your intended voice.</div>")

st.markdown("## Communication to Evaluate")
st.html("<div class='small-note'>Paste your message below. Add context in the left panel if needed.</div>")
message = st.text_area(
    "Message",
    placeholder="Paste your email, social-media post, customer-service response, business message, or other communication here...",
    height=300,
    label_visibility="collapsed",
)

if st.button("Evaluate Message", type="primary", use_container_width=True):
    if not message.strip():
        st.warning("Please paste a message first.")
    else:
        with st.spinner("Evaluating human touch…"):
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

result = st.session_state.get("result")
original = st.session_state.get("original", "")

if result:
    st.html("<div class='result-line'></div>")
    st.markdown("## Human-Touch Result")

    overall = clamp_score(result.get("overall_human_touch_score"))
    left, right = st.columns([1, 2])
    with left:
        st.html(f"""
        <div class="score-card">
            <div class="score-number">{overall}<span style="font-size:22px;color:#747782;">/100</span></div>
            <div class="score-label">HUMAN TOUCH SCORE</div>
            <div class="verdict">{safe_text(result.get('verdict'))}</div>
        </div>
        """)
    with right:
        st.html(f"""
        <div class="panel">
            <div class="panel-label">SUMMARY</div>
            <div class="panel-text">{safe_text(result.get('summary'))}</div>
        </div>
        """)

    st.markdown("### Quality Dimensions")
    dimensions = [
        ("Empathy", "empathy"),
        ("Naturalness", "naturalness"),
        ("Personalization", "personalization"),
        ("Context Awareness", "context_awareness"),
        ("Brand Voice", "brand_voice"),
    ]
    for label, key in dimensions:
        value = clamp_score(result["scores"].get(key))
        st.html(f"""
        <div class="metric">
            <div class="metric-top"><span>{label}</span><span class="metric-value">{value}</span></div>
            <div class="track"><div class="fill" style="width:{value}%"></div></div>
        </div>
        """)

    st.markdown("### Recipient Perspective")
    st.html(f"<div class='panel'><div class='panel-text'>{safe_text(result.get('recipient_perspective'))}</div></div>")

    st.markdown("### Potential Concerns")
    concerns = result.get("potential_concerns") or []
    if not concerns:
        st.success("No major concerns identified.")
    else:
        for item in concerns:
            st.html(f"""
            <div class="concern">
                <div class="concern-title">{safe_text(item.get('issue', 'Potential concern'))}</div>
                <div class="concern-text"><b>Dimension:</b> {safe_text(item.get('dimension'))}</div>
                <div class="concern-text"><b>Evidence:</b> {safe_text(item.get('evidence'))}</div>
                <div class="concern-text"><b>Recipient impact:</b> {safe_text(item.get('impact_on_recipient'))}</div>
                <span class="badge">{safe_text(item.get('severity', 'Low'))}</span>
            </div>
            """)

    st.markdown("### Recommended Changes")
    changes = result.get("recommended_changes") or []
    if not changes:
        st.info("No major changes needed.")
    else:
        for item in changes:
            st.html(f"""
            <div class="panel" style="margin:10px 0;">
                <div style="font-weight:750;margin-bottom:7px;">{safe_text(item.get('title', 'Recommended improvement'))}</div>
                <div class="panel-text">{safe_text(item.get('explanation'))}</div>
            </div>
            """)

    st.markdown("### Recommended Rewrite")
    st.html(f"<div class='rewrite'>{safe_text(result.get('recommended_rewrite') or 'No rewrite was provided.')}</div>")

st.html("</div>")
