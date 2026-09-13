import json
import os
import html
from typing import Any, Dict, List

import streamlit as st
from groq import Groq


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Human Touch Quality Layer",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# -----------------------------
# Visual design
# -----------------------------
CUSTOM_CSS = """
<style>
:root {
    --htql-accent: #7c5cff;
    --htql-accent-2: #00c6ff;
    --htql-card: rgba(255,255,255,0.82);
    --htql-border: rgba(20, 24, 40, 0.10);
    --htql-text: #172033;
    --htql-muted: #65708a;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 8% 4%, rgba(124, 92, 255, 0.13), transparent 30%),
        radial-gradient(circle at 92% 8%, rgba(0, 198, 255, 0.11), transparent 27%),
        linear-gradient(180deg, #f8f9ff 0%, #f4f7fb 100%);
}

[data-testid="stHeader"] { background: transparent; }

.block-container {
    max-width: 1180px;
    padding-top: 1.5rem;
    padding-bottom: 4rem;
}

.htql-hero {
    padding: 2rem 2.15rem;
    border-radius: 28px;
    background: linear-gradient(135deg, #171936 0%, #3b267f 56%, #0d8fbe 100%);
    color: white;
    box-shadow: 0 22px 55px rgba(44, 32, 111, 0.22);
    margin-bottom: 1.35rem;
}

.htql-kicker {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    font-weight: 800;
    opacity: 0.78;
    margin-bottom: 0.55rem;
}

.htql-hero h1 {
    margin: 0;
    font-size: clamp(2.1rem, 4vw, 3.5rem);
    line-height: 0.98;
    letter-spacing: -0.045em;
}

.htql-hero p {
    max-width: 790px;
    margin: 0.95rem 0 0;
    font-size: 1.06rem;
    line-height: 1.6;
    color: rgba(255,255,255,0.88);
}

.htql-principle {
    margin-top: 1.2rem;
    padding: 0.88rem 1rem;
    border: 1px solid rgba(255,255,255,0.19);
    border-radius: 16px;
    background: rgba(255,255,255,0.08);
    font-size: 0.92rem;
}

.htql-section-title {
    margin-top: 1.5rem;
    margin-bottom: 0.65rem;
    color: var(--htql-text);
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: -0.02em;
}

.htql-card {
    background: var(--htql-card);
    border: 1px solid var(--htql-border);
    border-radius: 20px;
    padding: 1.05rem 1.1rem;
    box-shadow: 0 10px 30px rgba(31, 42, 70, 0.05);
    backdrop-filter: blur(10px);
}

.metric-card {
    min-height: 155px;
    background: rgba(255,255,255,0.84);
    border: 1px solid rgba(20,24,40,0.09);
    border-radius: 20px;
    padding: 1rem;
    box-shadow: 0 10px 28px rgba(28, 38, 68, 0.05);
}

.metric-label { color: #58637a; font-weight: 700; font-size: 0.86rem; }
.metric-score { color: #172033; font-weight: 900; font-size: 2.05rem; line-height: 1.05; margin-top: 0.28rem; }
.metric-reason { color: #5f6a82; font-size: 0.85rem; line-height: 1.45; margin-top: 0.55rem; }

.overall-card {
    padding: 1.3rem 1.35rem;
    border-radius: 24px;
    background: linear-gradient(145deg, rgba(124,92,255,0.12), rgba(0,198,255,0.11));
    border: 1px solid rgba(124,92,255,0.16);
    box-shadow: 0 12px 34px rgba(64, 61, 132, 0.07);
}

.overall-score { font-size: 3.2rem; font-weight: 950; letter-spacing: -0.05em; color: #1a2040; }
.overall-label { font-size: 0.88rem; text-transform: uppercase; letter-spacing: 0.12em; font-weight: 800; color: #5d6482; }

.concern-card {
    border-radius: 16px;
    padding: 0.95rem 1rem;
    margin-bottom: 0.7rem;
    background: rgba(255,255,255,0.86);
    border: 1px solid rgba(20,24,40,0.08);
}
.concern-title { font-weight: 850; color: #1e2438; }
.concern-body { color: #5f6a82; font-size: 0.9rem; line-height: 1.48; margin-top: 0.2rem; }

.badge {
    display: inline-block;
    border-radius: 999px;
    padding: 0.18rem 0.56rem;
    font-size: 0.72rem;
    font-weight: 850;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-right: 0.45rem;
}
.badge-low { background: #e8f7ee; color: #237a48; }
.badge-medium { background: #fff2d9; color: #9b6515; }
.badge-high { background: #ffe6e5; color: #b13b37; }

.rewrite-card {
    background: rgba(255,255,255,0.92);
    border-radius: 18px;
    padding: 1.1rem;
    border: 1px solid rgba(20,24,40,0.09);
    min-height: 245px;
}

.stButton > button {
    border-radius: 14px;
    min-height: 3rem;
    font-weight: 850;
    border: 0;
    background: linear-gradient(135deg, #7252ff 0%, #089bd2 100%);
    color: white;
    box-shadow: 0 12px 26px rgba(76, 71, 178, 0.22);
}

.stButton > button:hover { filter: brightness(1.03); transform: translateY(-1px); }

div[data-testid="stProgressBar"] > div > div { border-radius: 999px; }

.small-note {
    color: #6b7488;
    font-size: 0.78rem;
    line-height: 1.4;
}

footer { visibility: hidden; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------
# Session state
# -----------------------------
if "results" not in st.session_state:
    st.session_state["results"] = None
if "analyzed_message" not in st.session_state:
    st.session_state["analyzed_message"] = ""


# -----------------------------
# Configuration helpers
# -----------------------------
def get_groq_api_key() -> str:
    """Read the Groq API key from Streamlit Secrets or environment variables."""
    try:
        secret_value = st.secrets.get("GROQ_API_KEY")
        if secret_value:
            return str(secret_value).strip()
    except Exception:
        # Streamlit raises when secrets are not configured in some environments.
        pass

    return os.getenv("GROQ_API_KEY", "").strip()


def get_groq_model() -> str:
    """Read a configurable Groq model name, with a sensible default."""
    try:
        secret_value = st.secrets.get("GROQ_MODEL")
        if secret_value:
            return str(secret_value).strip()
    except Exception:
        pass

    return os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()


# -----------------------------
# Prompt architecture
# -----------------------------
def build_system_prompt() -> str:
    return """
You are the Communication Quality Evaluator for the product "Human Touch Quality Layer".

Your job is to evaluate a communication from the recipient's perspective. You are NOT an
AI-content detector, authorship detector, generic grammar checker, sentiment classifier,
or generic writing assistant.

The central question is:
"Would the recipient reasonably feel that this message was thoughtfully written for their
specific situation?"

Evaluate only what can be supported by the submitted communication and the supplied context.
Never invent or assume facts. Missing information is not a license to fabricate details.
Never infer a specific recipient emotion as a fact unless the user supplied evidence for it.
When context is missing, explicitly state the limitation in the relevant reason/evidence.

SCORING RULES
- Score each dimension from 0 to 100.
- Use evidence from the message, supplied context, missing information, or context mismatch.
- High scores require actual evidence.
- A missing context field does not automatically mean a low score, but it can reduce confidence.
- Do not reward generic politeness, a recipient name alone, or polished wording by themselves.
- Never say or imply that a message was written by AI.
- Do not create fake quotations. Any quoted evidence must be copied exactly from the message.
- Keep reasons concise and useful to a non-technical stakeholder.

DIMENSIONS
1. Empathy: whether the communication appropriately recognizes relevant difficulty, concern,
   inconvenience, needs, or emotions when relevant. Do not reward formulaic empathy without evidence.
2. Naturalness: whether the message sounds like something a real person would naturally write.
   Consider jargon, robotic phrasing, filler, excessive polish, awkward transitions, and over-formality.
   Unnaturalness is a quality issue, NOT evidence of AI authorship.
3. Personalization: whether the message is meaningfully tailored to the recipient/audience and
   situation. A name alone is not sufficient.
4. Context Awareness: whether the message respects the provided purpose, audience, relationship,
   channel, tone, sensitivity, facts, constraints, and sender expectations.
5. Brand Voice: whether the message aligns with the supplied sender/brand voice. If no brand voice
   is supplied, explicitly say the score is based on observable style and limited context rather than
   a verified brand standard.

CONCERNS
Only report supported concerns. Use zero concerns when no material concerns are present.
Allowed examples include: Generic language, Templated language, Weak empathy, Artificial empathy,
Missing personalization, Ignored context, Inappropriate tone, Unsupported assumption,
Overpromising, Brand voice mismatch, Missing important information, Missing call to action,
Ambiguous wording, Excessive verbosity, Unnecessary formality, Sensitive wording,
Factual inconsistency.

REWRITE
Produce a recommended rewrite that preserves the original intent, all supported facts, required
information, appropriate commitments, calls to action, and provided sender/brand voice.
Never invent names, dates, policies, discounts, delivery dates, results, commitments, feelings,
background, or other unsupported details. If required information is absent, do not manufacture it.
Improve the message only from evidence supplied by the user.

OUTPUT CONTRACT
Return valid JSON only. No markdown fences, no commentary outside JSON.
Use exactly these top-level fields:
{
  "empathy_score": 0,
  "empathy_reason": "",
  "empathy_evidence": "",
  "naturalness_score": 0,
  "naturalness_reason": "",
  "naturalness_evidence": "",
  "personalization_score": 0,
  "personalization_reason": "",
  "personalization_evidence": "",
  "context_awareness_score": 0,
  "context_awareness_reason": "",
  "context_awareness_evidence": "",
  "brand_voice_score": 0,
  "brand_voice_reason": "",
  "brand_voice_evidence": "",
  "overall_human_touch_score": 0,
  "potential_concerns": [
    {"title": "", "severity": "low", "explanation": ""}
  ],
  "recommended_rewrite": ""
}

Do not add unsupported facts to evidence, concerns, or the rewrite.
""".strip()


def build_user_prompt(context: Dict[str, str]) -> str:
    def block(label: str, value: str) -> str:
        cleaned = value.strip() if isinstance(value, str) else ""
        return f"### {label}\n{cleaned if cleaned else '[Not provided]'}"

    return "\n\n".join(
        [
            block("Communication", context.get("communication", "")),
            block("Communication Type", context.get("communication_type", "")),
            block("Purpose", context.get("purpose", "")),
            block("Audience / Recipient", context.get("audience", "")),
            block("Relationship With Recipient", context.get("relationship", "")),
            block("Channel", context.get("channel", "")),
            block("Desired Tone", context.get("tone", "")),
            block("Sensitivity Level", context.get("sensitivity", "")),
            block("Important Facts / Requirements", context.get("facts", "")),
            block("Brand / Sender Voice", context.get("brand_voice", "")),
            "### Evaluation instruction\nEvaluate the communication using only the evidence above. Explicitly identify meaningful context limitations.",
        ]
    )


# -----------------------------
# Groq API + response validation
# -----------------------------
def call_groq(system_prompt: str, user_prompt: str) -> str:
    api_key = get_groq_api_key()
    if not api_key:
        raise RuntimeError(
            "Groq API key is not configured. Add GROQ_API_KEY to your environment variables or Streamlit Secrets."
        )

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=get_groq_model(),
        temperature=0.2,
        max_tokens=2200,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response.choices[0].message.content if response.choices else None
    if not content or not content.strip():
        raise ValueError("The evaluator returned an empty response.")
    return content.strip()


REQUIRED_KEYS = [
    "empathy_score",
    "empathy_reason",
    "empathy_evidence",
    "naturalness_score",
    "naturalness_reason",
    "naturalness_evidence",
    "personalization_score",
    "personalization_reason",
    "personalization_evidence",
    "context_awareness_score",
    "context_awareness_reason",
    "context_awareness_evidence",
    "brand_voice_score",
    "brand_voice_reason",
    "brand_voice_evidence",
    "overall_human_touch_score",
    "potential_concerns",
    "recommended_rewrite",
]


def _extract_json_object(raw_text: str) -> str | None:
    """Safely extract the first balanced JSON object from a response."""
    text = raw_text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _score_is_valid(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= float(value) <= 100


def validate_result(raw_response: str) -> Dict[str, Any]:
    """Parse and validate the evaluator JSON. Raises ValueError on invalid output."""
    candidate = raw_response.strip()
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        extracted = _extract_json_object(candidate)
        if not extracted:
            raise ValueError("The evaluator returned invalid JSON.")
        try:
            data = json.loads(extracted)
        except json.JSONDecodeError as exc:
            raise ValueError("The evaluator returned invalid JSON.") from exc

    if not isinstance(data, dict):
        raise ValueError("The evaluator response is not a JSON object.")

    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        raise ValueError(f"The evaluator response is missing required fields: {', '.join(missing)}")

    score_keys = [
        "empathy_score",
        "naturalness_score",
        "personalization_score",
        "context_awareness_score",
        "brand_voice_score",
        "overall_human_touch_score",
    ]
    for key in score_keys:
        if not _score_is_valid(data[key]):
            raise ValueError(f"Invalid score for {key}.")

    for key in [
        "empathy_reason",
        "empathy_evidence",
        "naturalness_reason",
        "naturalness_evidence",
        "personalization_reason",
        "personalization_evidence",
        "context_awareness_reason",
        "context_awareness_evidence",
        "brand_voice_reason",
        "brand_voice_evidence",
        "recommended_rewrite",
    ]:
        if not isinstance(data[key], str):
            raise ValueError(f"Invalid text field: {key}.")

    concerns = data["potential_concerns"]
    if not isinstance(concerns, list):
        raise ValueError("Potential concerns must be a list.")

    normalized_concerns: List[Dict[str, str]] = []
    for concern in concerns:
        if not isinstance(concern, dict):
            raise ValueError("Each concern must be an object.")
        title = concern.get("title")
        severity = str(concern.get("severity", "")).lower().strip()
        explanation = concern.get("explanation")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("Concern title is missing.")
        if severity not in {"low", "medium", "high"}:
            raise ValueError("Concern severity must be low, medium, or high.")
        if not isinstance(explanation, str) or not explanation.strip():
            raise ValueError("Concern explanation is missing.")
        normalized_concerns.append(
            {"title": title.strip(), "severity": severity, "explanation": explanation.strip()}
        )

    data["potential_concerns"] = normalized_concerns
    for key in score_keys:
        data[key] = round(float(data[key]))

    return data


def calculate_human_touch_score(result: Dict[str, Any]) -> int:
    scores = [
        result["empathy_score"],
        result["naturalness_score"],
        result["personalization_score"],
        result["context_awareness_score"],
        result["brand_voice_score"],
    ]
    return round(sum(scores) / len(scores))


# -----------------------------
# UI helpers
# -----------------------------
def score_label(score: int) -> str:
    if score <= 39:
        return "Poor"
    if score <= 59:
        return "Needs Improvement"
    if score <= 74:
        return "Fair"
    if score <= 89:
        return "Strong"
    return "Excellent"


def render_metric_card(label: str, score: int, reason: str, evidence: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-score">{score}<span style="font-size:1rem;color:#7b8498;"> / 100</span></div>
            <div class="metric-reason">{reason}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("View evidence"):
        st.write(evidence or "No additional evidence was provided.")


def render_concerns(concerns: List[Dict[str, str]]) -> None:
    if not concerns:
        st.success("No material concerns were identified from the supplied communication and context.")
        return

    for concern in concerns:
        severity = concern["severity"]
        badge_class = f"badge badge-{severity}"
        title = safe_display_text(concern["title"])
        explanation = safe_display_text(concern["explanation"])
        severity_display = html.escape(severity)
        st.markdown(
            f"""
            <div class="concern-card">
                <div><span class="{badge_class}">{severity_display}</span><span class="concern-title">{title}</span></div>
                <div class="concern-body">{explanation}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def safe_display_text(text: str) -> str:
    """Normalize and HTML-escape model text used inside custom HTML blocks."""
    if not isinstance(text, str):
        return ""
    return html.escape(text.strip())


# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div class="htql-hero">
        <div class="htql-kicker">Recipient-centered communication QA</div>
        <h1>Human Touch<br/>Quality Layer</h1>
        <p>Evaluate whether your communication feels thoughtful, specific, natural, context-aware, and right for the recipient.</p>
        <div class="htql-principle"><strong>Core question:</strong> Would the recipient reasonably feel that this message was thoughtfully written for their specific situation?</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Input section
# -----------------------------
st.markdown('<div class="htql-section-title">1. Communication</div>', unsafe_allow_html=True)

communication = st.text_area(
    "Communication / Message",
    height=240,
    placeholder="Paste the email, customer response, post, or business message you want to evaluate...",
    help="The message is required. It is sent to Groq only when you click Analyze Communication.",
)

communication_type = st.selectbox(
    "Communication Type",
    [
        "Email",
        "Social Media Post",
        "Customer Service Response",
        "Business Message",
        "Internal Communication",
        "Other",
    ],
)


# -----------------------------
# Context section
# -----------------------------
st.markdown('<div class="htql-section-title">2. Context</div>', unsafe_allow_html=True)
st.caption("Provide what you know. Missing context is allowed; the evaluator will identify where confidence is limited.")

left, right = st.columns(2)
with left:
    purpose_options = [
        "Follow up",
        "Apologize",
        "Inform",
        "Request information",
        "Resolve a complaint",
        "Sell / promote",
        "Thank someone",
        "Announce something",
        "Invite someone",
        "Provide support",
        "Other",
    ]
    purpose_choice = st.selectbox("Purpose", purpose_options)
    purpose_other = ""
    if purpose_choice == "Other":
        purpose_other = st.text_input("Purpose details", placeholder="Describe the communication goal...")
    audience = st.text_input(
        "Audience / Recipient",
        placeholder="e.g. Existing customer who experienced a delayed order.",
    )
    relationship = st.selectbox(
        "Relationship With Recipient",
        [
            "Not provided",
            "New customer",
            "Existing customer",
            "Prospect",
            "Colleague",
            "Manager",
            "Business partner",
            "Friend",
            "Public audience",
            "Other",
        ],
    )
    relationship_other = ""
    if relationship == "Other":
        relationship_other = st.text_input("Relationship details", placeholder="Describe the relationship...")
    channel = st.selectbox(
        "Channel",
        [
            "Email",
            "LinkedIn",
            "Instagram",
            "WhatsApp",
            "Website",
            "Support ticket",
            "SMS",
            "Internal communication",
            "Other",
        ],
    )

with right:
    tone = st.selectbox(
        "Desired Tone",
        [
            "Professional",
            "Friendly",
            "Warm",
            "Concise",
            "Empathetic",
            "Persuasive",
            "Formal",
            "Casual",
            "Reassuring",
            "Apologetic",
            "Other",
        ],
    )
    tone_other = ""
    if tone == "Other":
        tone_other = st.text_input("Tone details", placeholder="Describe the intended tone...")
    sensitivity = st.select_slider("Sensitivity Level", options=["Low", "Medium", "High"], value="Low")
    facts = st.text_area(
        "Important Facts / Requirements",
        height=135,
        placeholder="e.g. Order was delayed by 3 days. Refund has been approved. Do not promise a delivery date. Include ticket #4821.",
    )
    brand_voice = st.text_area(
        "Brand / Sender Voice",
        height=135,
        placeholder="e.g. Professional but warm. Avoid corporate jargon. Use concise sentences. Never use exaggerated claims.",
    )

purpose = purpose_other.strip() if purpose_choice == "Other" and purpose_other.strip() else purpose_choice
resolved_relationship = (
    relationship_other.strip() if relationship == "Other" and relationship_other.strip() else relationship
)
resolved_tone = tone_other.strip() if tone == "Other" and tone_other.strip() else tone


# -----------------------------
# Analysis action
# -----------------------------
analyze = st.button("Analyze Communication", type="primary", use_container_width=True)

if analyze:
    if not communication.strip():
        st.session_state["results"] = None
        st.error("Please enter a communication before analyzing.")
    else:
        context = {
            "communication": communication.strip(),
            "communication_type": communication_type,
            "purpose": purpose,
            "audience": audience.strip(),
            "relationship": resolved_relationship,
            "channel": channel,
            "tone": resolved_tone,
            "sensitivity": sensitivity,
            "facts": facts.strip(),
            "brand_voice": brand_voice.strip(),
        }

        with st.spinner("Evaluating communication..."):
            try:
                system_prompt = build_system_prompt()
                user_prompt = build_user_prompt(context)
                raw_response = call_groq(system_prompt, user_prompt)
                result = validate_result(raw_response)
                # The local formula is authoritative and transparent.
                result["overall_human_touch_score"] = calculate_human_touch_score(result)
                st.session_state["results"] = result
                st.session_state["analyzed_message"] = communication.strip()
            except RuntimeError as exc:
                st.session_state["results"] = None
                st.error(str(exc))
            except ValueError:
                st.session_state["results"] = None
                st.error("The evaluator returned an invalid response. Please try again.")
            except Exception as exc:
                st.session_state["results"] = None
                message = str(exc).lower()
                if "rate" in message and "limit" in message:
                    st.warning("The AI service is temporarily rate-limited. Please wait a moment and try again.")
                elif any(token in message for token in ["timeout", "timed out", "deadline"]):
                    st.warning("The evaluation timed out. Please try again.")
                elif any(token in message for token in ["api key", "authentication", "unauthorized", "401"]):
                    st.error("Groq API authentication failed. Check your GROQ_API_KEY configuration.")
                else:
                    st.error("We could not complete the evaluation. Please check your API configuration and try again.")


# -----------------------------
# Results section
# -----------------------------
result = st.session_state.get("results")
if result:
    st.markdown('<div class="htql-section-title">3. Results</div>', unsafe_allow_html=True)

    overall = result["overall_human_touch_score"]
    st.markdown(
        f"""
        <div class="overall-card">
            <div class="overall-label">Human-Touch Score</div>
            <div class="overall-score">{overall} <span style="font-size:1.05rem;color:#717b90;">/ 100</span></div>
            <div style="font-size:1rem;font-weight:800;color:#4f5974;">{score_label(overall)}</div>
            <div class="small-note" style="margin-top:0.45rem;">Average of the five primary dimensions: Empathy, Naturalness, Personalization, Context Awareness, and Brand Voice.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(overall / 100)

    metric_items = [
        ("Empathy", "empathy_score", "empathy_reason", "empathy_evidence"),
        ("Naturalness", "naturalness_score", "naturalness_reason", "naturalness_evidence"),
        ("Personalization", "personalization_score", "personalization_reason", "personalization_evidence"),
        ("Context Awareness", "context_awareness_score", "context_awareness_reason", "context_awareness_evidence"),
        ("Brand Voice", "brand_voice_score", "brand_voice_reason", "brand_voice_evidence"),
    ]
    cols = st.columns(5)
    for column, (label, score_key, reason_key, evidence_key) in zip(cols, metric_items):
        with column:
            render_metric_card(label, int(result[score_key]), safe_display_text(result[reason_key]), safe_display_text(result[evidence_key]))

    st.markdown('<div class="htql-section-title">Potential Concerns</div>', unsafe_allow_html=True)
    render_concerns(result["potential_concerns"])

    st.markdown('<div class="htql-section-title">Original vs. Recommended Rewrite</div>', unsafe_allow_html=True)
    original_col, rewrite_col = st.columns(2)
    with original_col:
        st.markdown('<div class="rewrite-card"><strong>Original Communication</strong><br><br></div>', unsafe_allow_html=True)
        st.text(st.session_state.get("analyzed_message", ""))
    with rewrite_col:
        st.markdown('<div class="rewrite-card"><strong>Recommended Rewrite</strong><br><br></div>', unsafe_allow_html=True)
        st.text(result["recommended_rewrite"])

    st.caption("The rewrite is intended as a recommendation. Review it before sending, especially for sensitive communications.")


# -----------------------------
# Footer note
# -----------------------------
st.markdown(
    """
    <div style="margin-top:2rem;padding-top:1rem;border-top:1px solid rgba(20,24,40,0.08);color:#6b7488;font-size:0.8rem;line-height:1.5;">
        Human Touch Quality Layer evaluates communication quality. It does not determine whether content was written by a human or AI, and its scores are model-generated judgments rather than objective measurements of authenticity.
    </div>
    """,
    unsafe_allow_html=True,
)
