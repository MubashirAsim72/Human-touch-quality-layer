# Human Touch Quality Layer

A recipient-centered AI communication quality-control MVP built with **Python, Streamlit, and Groq**.

The product evaluates whether a communication feels thoughtful, specific, natural, context-aware, emotionally appropriate, and consistent with the intended sender or brand voice.

> **Core question:** Would the recipient reasonably feel that this message was thoughtfully written for their specific situation?

This application is **not an AI-content detector** and never attempts to determine whether a message was written by a human or AI.

## Features

- Five primary communication-quality dimensions:
  - Empathy
  - Naturalness
  - Personalization
  - Context Awareness
  - Brand Voice
- Optional overall **Human-Touch Score**
- Evidence-based explanations for each score
- Context limitations called out when information is missing
- Structured potential concerns with severity levels
- Recommended rewrite that preserves the original intent and supplied facts
- Secure Groq API integration using environment variables or Streamlit Secrets
- Session-state storage of the latest successful evaluation only
- Professional gradient-based Streamlit UI
- Input validation and graceful API/JSON error handling

## Project Structure

```text
human-touch-quality-layer/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Architecture

```text
Streamlit UI
      ↓
Context Builder
      ↓
Evaluation Prompt
      ↓
Groq API
      ↓
Structured JSON
      ↓
Validation
      ↓
Results UI
```

The application keeps prompt construction separate from the UI and API call flow through dedicated functions:

```python
build_system_prompt()
build_user_prompt(context)
call_groq(system_prompt, user_prompt)
validate_result(raw_response)
calculate_human_touch_score(result)
```

## Installation

### Local setup

Use Python 3.10+ and create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Groq Configuration

The application reads two configuration values:

```text
GROQ_API_KEY
GROQ_MODEL
```

`GROQ_API_KEY` is required.

`GROQ_MODEL` is optional and lets you change the model without modifying `app.py`.

The default model in this MVP is:

```text
llama-3.3-70b-versatile
```

Because model availability can change on the Groq platform, set `GROQ_MODEL` to a currently available Groq chat model when necessary.

### Local environment variables

Windows PowerShell example:

```powershell
$env:GROQ_API_KEY="your-key-here"
$env:GROQ_MODEL="llama-3.3-70b-versatile"
streamlit run app.py
```

macOS/Linux example:

```bash
export GROQ_API_KEY="your-key-here"
export GROQ_MODEL="llama-3.3-70b-versatile"
streamlit run app.py
```

Never commit a real API key to GitHub.

## Streamlit Secrets

For Streamlit deployment, create `.streamlit/secrets.toml` locally for testing or enter the same values in the deployment platform's Secrets interface.

Example:

```toml
GROQ_API_KEY = "your-key-here"
GROQ_MODEL = "llama-3.3-70b-versatile"
```

The real key should only be entered into the Streamlit Secrets interface or a secure local environment variable. It must never be committed to GitHub.

## Google Colab Testing

The app itself is normal Streamlit code and contains no mandatory Colab-specific logic.

In a new Google Colab notebook:

### 1. Install dependencies

```python
!pip install -r /content/human-touch-quality-layer/requirements.txt
```

Or install directly:

```python
!pip install "streamlit>=1.40,<2.0" "groq>=0.11,<1.0"
```

### 2. Set the Groq key securely for the current Colab runtime

```python
import os
from getpass import getpass

os.environ["GROQ_API_KEY"] = getpass("Enter your Groq API key: ")
os.environ["GROQ_MODEL"] = "llama-3.3-70b-versatile"
```

Do not hardcode the real key into notebook cells that you intend to publish.

### 3. Start Streamlit

```python
!streamlit run /content/human-touch-quality-layer/app.py &>/content/streamlit.log &
```

### 4. Expose Streamlit temporarily for browser testing

A tunneling service can expose the Colab port temporarily. One common pattern is to use a tunnel package supported by your environment, for example:

```python
!pip install pyngrok
```

Then configure a tunnel according to your tunnel provider's current authentication requirements and forward port `8501`.

For production, deploy the repository directly to Streamlit rather than relying on a Colab tunnel.

## GitHub Setup

From the project directory:

```bash
git init
git add .
git commit -m "Initial Human Touch Quality Layer MVP"
git branch -M main
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin main
```

No credentials belong in the repository. `.gitignore` already excludes common local secret/config files.

## Streamlit Deployment

1. Push this project to a GitHub repository.
2. Open Streamlit and create a new app.
3. Connect the GitHub repository.
4. Select `app.py` as the main file.
5. Configure the application's Secrets with:

```toml
GROQ_API_KEY = "your-key-here"
GROQ_MODEL = "llama-3.3-70b-versatile"
```

6. Deploy the application.
7. Keep the real API key only in Streamlit Secrets.

## Human-Touch Score Formula

The five dimension scores remain the primary outputs.

The optional overall score is calculated locally by the application using an equal-weight average:

```text
Human Touch Score =
(
    Empathy +
    Naturalness +
    Personalization +
    Context Awareness +
    Brand Voice
) / 5
```

This local calculation is intentionally authoritative so the displayed overall score is transparent and cannot drift from the five visible dimensions.

Score interpretation:

```text
0–39   = Poor
40–59  = Needs Improvement
60–74  = Fair
75–89  = Strong
90–100 = Excellent
```

## Security

- API keys are loaded from environment variables or Streamlit Secrets.
- No API key is embedded in Python, HTML, JavaScript, CSS, or the README.
- No API key is printed to the interface.
- `.streamlit/secrets.toml` and `.env` are excluded by `.gitignore`.
- Communication content is kept in Streamlit session state only for the latest successful result; this MVP does not implement external persistence.

## Response Validation

Before results are rendered, `validate_result()` checks:

- A response exists
- The response is valid JSON, with a safe extraction fallback when reasonable
- All required top-level fields exist
- All scores are numeric and between 0 and 100
- Text fields are actually strings
- `potential_concerns` is a list
- Every concern has a title, allowed severity, and explanation
- The rewrite exists and is text

Malformed model output is never silently converted into fabricated values.

## Prompt Architecture

The system prompt establishes evaluator behavior and safety rules. The user prompt separately supplies:

- Communication
- Communication type
- Purpose
- Audience / recipient
- Relationship
- Channel
- Desired tone
- Sensitivity
- Important facts / requirements
- Brand / sender voice

This separation makes the application easier to test and extend.

## Limitations

- Scores are model-generated evaluations, not objective measurements.
- Missing context can reduce evaluation confidence.
- The product does not determine whether content is AI-generated or human-written.
- Recipient experience is inherently contextual and cannot be measured with certainty.
- High-sensitivity communication should still receive human review before sending.
- The default Groq model may need to be changed when Groq retires or changes model availability.

## Future Extensions

Possible future directions include:

- More communication dimensions
- Historical benchmarking
- Team workflows
- Enterprise integrations
- Analytics

These are intentionally not implemented in this MVP.

## License

Choose the license that fits your repository and intended distribution model before publishing.
