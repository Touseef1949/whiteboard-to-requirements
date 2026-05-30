import json
import os
import re
from html import escape
from io import BytesIO
from typing import Any, Dict, List, Tuple

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from PIL import Image as PILImage, UnidentifiedImageError

from agno.agent import Agent
from agno.media import Image as AgnoImage
from agno.models.google import Gemini
from agno.models.openai import OpenAIChat

try:
    from agno.models.deepseek import DeepSeek

    NATIVE_DEEPSEEK_AVAILABLE = True
except Exception:
    DeepSeek = None  # type: ignore[assignment]
    NATIVE_DEEPSEEK_AVAILABLE = False


# Try loading .env from multiple locations
load_dotenv()  # current directory
load_dotenv(os.path.expanduser("~/.hermes/.env"))  # Hermes config

APP_TITLE = "Whiteboard to Requirements"
VISION_MODEL_ID = os.getenv("GEMINI_MODEL_ID", "gemini-3.5-flash")
TEXT_MODEL_ID = os.getenv("DEEPSEEK_MODEL_ID", "deepseek-v4-flash")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        :root {
            --card-border: rgba(49, 51, 63, 0.15);
            --soft-bg: rgba(250, 250, 252, 0.85);
        }
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .hero-card {
            border: 1px solid var(--card-border);
            border-radius: 20px;
            padding: 1.35rem 1.5rem;
            background: linear-gradient(135deg, rgba(246, 248, 255, 0.95), rgba(255, 255, 255, 0.95));
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.05);
            margin-bottom: 1.25rem;
        }
        .hero-title {
            font-size: 2.05rem;
            font-weight: 760;
            letter-spacing: -0.035em;
            margin-bottom: 0.25rem;
        }
        .hero-subtitle {
            color: #475569;
            font-size: 1.02rem;
            line-height: 1.55;
        }
        .step-card {
            border: 1px solid var(--card-border);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            background: var(--soft-bg);
            margin: 0.75rem 0 1rem 0;
        }
        .status-pill-ok,
        .status-pill-warn {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            border-radius: 999px;
            padding: 0.25rem 0.65rem;
            font-size: 0.82rem;
            font-weight: 650;
        }
        .status-pill-ok {
            color: #166534;
            background: #dcfce7;
            border: 1px solid #bbf7d0;
        }
        .status-pill-warn {
            color: #9a3412;
            background: #ffedd5;
            border: 1px solid #fed7aa;
        }
        .small-muted {
            color: #64748b;
            font-size: 0.9rem;
            line-height: 1.45;
        }
        .footer {
            color: #64748b;
            font-size: 0.88rem;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid rgba(100, 116, 139, 0.18);
            text-align: center;
        }
        .stButton > button {
            border-radius: 12px;
            font-weight: 650;
        }
        div[data-testid="stTextInput"] input {
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Utility functions
# -----------------------------


def get_secret(name: str) -> str:
    """Resolve a key from environment variables first, then Streamlit secrets."""
    value = os.getenv(name, "")
    if value:
        return value
    try:
        secret_value = st.secrets.get(name, "")
        return str(secret_value) if secret_value else ""
    except Exception:
        return ""


GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")
DEEPSEEK_API_KEY = get_secret("DEEPSEEK_API_KEY")


def initialize_session_state() -> None:
    defaults = {
        "image_bytes": None,
        "image_mime_type": None,
        "image_filename": None,
        "vision_analysis": "",
        "detected_elements": [],
        "questions": [],
        "answers": {},
        "final_output": "",
        "last_error": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_analysis_state() -> None:
    st.session_state.vision_analysis = ""
    st.session_state.detected_elements = []
    st.session_state.questions = []
    st.session_state.answers = {}
    st.session_state.final_output = ""
    st.session_state.last_error = ""


def reset_all_state() -> None:
    for key in [
        "image_bytes",
        "image_mime_type",
        "image_filename",
        "vision_analysis",
        "detected_elements",
        "questions",
        "answers",
        "final_output",
        "last_error",
    ]:
        if key in st.session_state:
            del st.session_state[key]
    initialize_session_state()


def validate_image(uploaded_file) -> Tuple[bool, str]:
    if uploaded_file is None:
        return False, "Upload a PNG, JPG, or JPEG image to continue."

    if uploaded_file.type not in {"image/png", "image/jpeg", "image/jpg"}:
        return False, "Invalid image type. Please upload a PNG, JPG, or JPEG file."

    try:
        image_bytes = uploaded_file.getvalue()
        image = PILImage.open(BytesIO(image_bytes))
        image.verify()
        return True, ""
    except UnidentifiedImageError:
        return False, "The uploaded file could not be read as an image. Try another whiteboard photo."
    except Exception as exc:
        return False, f"Image validation failed: {exc}"


def make_agno_image() -> AgnoImage:
    if not st.session_state.image_bytes or not st.session_state.image_mime_type:
        raise ValueError("No image is available in session state.")
    return AgnoImage(
        content=st.session_state.image_bytes,
        mime_type=st.session_state.image_mime_type,
    )


@st.cache_resource(show_spinner=False)
def get_vision_agent(api_key: str) -> Agent:
    return Agent(
        name="Whiteboard Vision Agent",
        role=(
            "Analyze whiteboard sketches, workflow drawings, system diagrams, "
            "process flows, and fintech product sketches."
        ),
        model=Gemini(id=VISION_MODEL_ID, api_key=api_key),
        markdown=True,
        instructions=[
            "You are a senior business analyst and visual systems analyst.",
            "Inspect the uploaded whiteboard photo and identify actors, systems, data stores, decisions, arrows, labels, missing context, and probable process flow.",
            "Generate clarifying questions that are grounded in visible elements of the sketch. Avoid generic questions unless the sketch is unreadable.",
            "Return valid JSON only. Do not wrap the JSON in Markdown.",
        ],
    )


@st.cache_resource(show_spinner=False)
def get_output_agent(api_key: str) -> Agent:
    if NATIVE_DEEPSEEK_AVAILABLE and DeepSeek is not None:
        text_model = DeepSeek(id=TEXT_MODEL_ID, api_key=api_key)
    else:
        text_model = OpenAIChat(
            id=TEXT_MODEL_ID,
            api_key=api_key,
            base_url=DEEPSEEK_BASE_URL,
        )

    return Agent(
        name="Requirements Generation Agent",
        role=(
            "Generate structured business analysis artifacts from whiteboard analysis, "
            "clarifying answers, and product context."
        ),
        model=text_model,
        markdown=True,
        instructions=[
            "You are a senior fintech business analyst, product owner, and agile requirements writer.",
            "Use INVEST principles for user stories and write testable acceptance criteria.",
            "Create concise but implementation-ready functional requirements with FR-XXX IDs grouped by module/domain.",
            "Generate Mermaid.js code blocks only for diagrams. Do not use a Python Mermaid library.",
            "Flag assumptions and unknowns explicitly in the gap analysis.",
        ],
    )


def extract_json_object(text: str) -> Dict[str, Any]:
    """Best-effort JSON extraction from an LLM response."""
    text = text.strip()

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            parsed = json.loads(fenced.group(1))
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            pass

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if 0 <= first_brace < last_brace:
        try:
            parsed = json.loads(text[first_brace : last_brace + 1])
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}

    return {}


def normalize_questions(raw_questions: Any) -> List[str]:
    questions: List[str] = []

    if isinstance(raw_questions, dict):
        raw_questions = list(raw_questions.values())

    if isinstance(raw_questions, list):
        for item in raw_questions:
            if isinstance(item, dict):
                text = item.get("question") or item.get("text") or item.get("q") or ""
            else:
                text = str(item)
            text = re.sub(r"^\s*(?:Q\s*)?\d+[\).:-]\s*", "", text).strip()
            if text:
                questions.append(text)

    return questions[:5]


def fallback_question_parse(text: str) -> List[str]:
    candidates = []
    for line in text.splitlines():
        line = re.sub(r"^[\s\-*•]*(?:Q\s*)?\d+[\).:-]?\s*", "", line).strip()
        if line.endswith("?") and len(line) > 15:
            candidates.append(line)
    return candidates[:5]


def parse_vision_response(content: str) -> Tuple[str, List[str], List[str]]:
    parsed = extract_json_object(content)
    if parsed:
        summary = str(
            parsed.get("analysis_summary")
            or parsed.get("summary")
            or parsed.get("whiteboard_summary")
            or ""
        ).strip()
        detected = parsed.get("detected_elements") or parsed.get("elements") or []
        if isinstance(detected, str):
            detected_elements = [item.strip() for item in re.split(r"[\n,;]+", detected) if item.strip()]
        elif isinstance(detected, list):
            detected_elements = [str(item).strip() for item in detected if str(item).strip()]
        else:
            detected_elements = []

        questions = normalize_questions(parsed.get("questions", []))
        return summary or content, detected_elements, questions

    return content, [], fallback_question_parse(content)


def fallback_questions() -> List[str]:
    return [
        "Which actor or role initiates the main workflow shown on the whiteboard?",
        "Which boxes represent internal systems versus third-party services or external partners?",
        "What event or business rule triggers the primary handoff between the first two components?",
        "Which data fields, documents, or records must be created, updated, or validated in this process?",
        "What exceptions, approvals, or failure paths should be included in the requirements?",
    ]


def analyze_whiteboard() -> None:
    if not GOOGLE_API_KEY:
        st.session_state.last_error = "GOOGLE_API_KEY is missing. Add it to Streamlit secrets or your environment."
        return

    prompt = """
Analyze this whiteboard diagram for a BA/Product Owner building fintech or workflow requirements.

Return valid JSON only with this exact shape:
{
  "analysis_summary": "A concise paragraph describing the visible flow, actors, systems, arrows, labels, decisions, and likely business purpose.",
  "detected_elements": ["element 1", "element 2", "element 3"],
  "questions": [
    "A specific clarifying question grounded in visible sketch content?",
    "A specific clarifying question grounded in visible sketch content?",
    "A specific clarifying question grounded in visible sketch content?"
  ]
}

Rules:
- Generate 3 to 5 questions.
- Make each question specific to visible boxes, actors, arrows, labels, data stores, decisions, or unclear handoffs.
- Prefer fintech/product language where relevant, such as customer, KYC, payment gateway, ledger, settlement, risk, approval, reconciliation, notification, API, webhook, and audit trail.
- If handwriting is unclear, state that uncertainty in the analysis_summary and ask targeted questions about the unclear labels.
""".strip()

    agent = get_vision_agent(GOOGLE_API_KEY)
    response = agent.run(prompt, stream=False, images=[make_agno_image()])
    content = getattr(response, "content", "") or str(response)
    summary, detected_elements, questions = parse_vision_response(content)

    if not questions:
        questions = fallback_questions()
        st.session_state.last_error = (
            "The vision response did not include parseable questions, so fallback questions were generated. "
            "You can retry analysis for more sketch-specific questions."
        )
    else:
        st.session_state.last_error = ""

    st.session_state.vision_analysis = summary
    st.session_state.detected_elements = detected_elements
    st.session_state.questions = questions
    st.session_state.answers = {str(idx): st.session_state.answers.get(str(idx), "") for idx in range(len(questions))}
    st.session_state.final_output = ""


def build_qa_transcript(questions: List[str], answers: Dict[str, str]) -> str:
    lines = []
    for idx, question in enumerate(questions):
        answer = answers.get(str(idx), "").strip()
        lines.append(f"Q{idx + 1}: {question}\nA{idx + 1}: {answer}")
    return "\n\n".join(lines)


def generate_requirements() -> None:
    if not DEEPSEEK_API_KEY:
        st.session_state.last_error = "DEEPSEEK_API_KEY is missing. Add it to Streamlit secrets or your environment."
        return

    questions = st.session_state.questions
    answers = st.session_state.answers
    missing = [idx + 1 for idx, question in enumerate(questions) if question and not answers.get(str(idx), "").strip()]
    if missing:
        st.session_state.last_error = f"Please answer all clarifying questions before generating requirements. Missing: {missing}."
        return

    qa_transcript = build_qa_transcript(questions, answers)
    detected_elements = "\n".join(f"- {item}" for item in st.session_state.detected_elements) or "- Not separately provided."

    prompt = f"""
Create structured BA artifacts from the following whiteboard analysis and user clarifications.

IMAGE ANALYSIS SUMMARY:
{st.session_state.vision_analysis}

DETECTED ELEMENTS:
{detected_elements}

Q&A TRANSCRIPT:
{qa_transcript}

Generate the final answer in Markdown with exactly these sections and no extra wrapper text:

## Epic Summary
Write a 2-3 sentence overview of the product/process capability.

## Functional Requirements
Group requirements by module/domain. Each requirement must have a stable FR-XXX ID, a concise title, and a testable requirement statement.

## INVEST User Stories
Write user stories in this exact style: "As a [role], I want [action] so that [benefit]." Include acceptance criteria under each story using Given/When/Then where practical.

## Mermaid Sequence Diagram
Provide one Mermaid.js sequence diagram in a fenced mermaid code block. Model the primary end-to-end process flow.

## Mermaid Flowchart
Provide one Mermaid.js flowchart in a fenced mermaid code block. Model the system architecture/component relationship. Use flowchart TD or flowchart LR.

## Gap Analysis
List ambiguous, missing, risky, or assumption-based areas that should be clarified before delivery.

Quality bar:
- Be specific and implementation-ready.
- Use fintech-friendly terminology where applicable.
- Do not invent regulatory claims; identify them as assumptions or gaps.
- Keep Mermaid syntax valid and avoid unsupported characters in node IDs.
""".strip()

    agent = get_output_agent(DEEPSEEK_API_KEY)
    response = agent.run(prompt, stream=False)
    st.session_state.final_output = getattr(response, "content", "") or str(response)
    st.session_state.last_error = ""


def extract_mermaid_blocks(markdown_text: str) -> Tuple[str, List[str]]:
    pattern = re.compile(r"```mermaid\s*(.*?)```", re.DOTALL | re.IGNORECASE)
    blocks = [match.group(1).strip() for match in pattern.finditer(markdown_text)]
    cleaned = pattern.sub("", markdown_text).strip()
    return cleaned, blocks


def classify_mermaid_blocks(blocks: List[str]) -> Tuple[str, str, List[str]]:
    sequence = ""
    flowchart = ""
    other: List[str] = []

    for block in blocks:
        normalized = block.strip().lower()
        if not sequence and normalized.startswith("sequencediagram"):
            sequence = block
        elif not flowchart and (
            normalized.startswith("flowchart")
            or normalized.startswith("graph")
            or normalized.startswith("statediagram")
        ):
            flowchart = block
        else:
            other.append(block)

    return sequence, flowchart, other


def render_mermaid(mermaid_code: str, height: int = 520) -> None:
    safe_code = escape(mermaid_code)
    html_doc = f"""
    <html>
      <head>
        <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
        <style>
          body {{ margin: 0; padding: 8px; background: transparent; }}
          .mermaid {{
            display: flex;
            justify-content: center;
            align-items: flex-start;
            min-height: 120px;
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          }}
        </style>
      </head>
      <body>
        <div class="mermaid">{safe_code}</div>
        <script>
          mermaid.initialize({{ startOnLoad: true, securityLevel: "loose", theme: "default" }});
        </script>
      </body>
    </html>
    """
    try:
        st.components.v1.html(html_doc, height=height, scrolling=True)
    except AttributeError:
        components.html(html_doc, height=height, scrolling=True)


def render_final_output(markdown_text: str) -> None:
    cleaned_markdown, mermaid_blocks = extract_mermaid_blocks(markdown_text)
    sequence_code, flowchart_code, other_blocks = classify_mermaid_blocks(mermaid_blocks)

    st.markdown(cleaned_markdown)

    if sequence_code:
        st.markdown("### 🔄 Process Flow")
        render_mermaid(sequence_code, height=520)

    if flowchart_code:
        st.markdown("### 🧱 System Architecture / Component Flow")
        render_mermaid(flowchart_code, height=520)

    for idx, block in enumerate(other_blocks, start=1):
        st.markdown(f"### Mermaid Diagram {idx}")
        render_mermaid(block, height=520)

    with st.expander("Raw generated Markdown"):
        st.code(markdown_text, language="markdown")

    st.download_button(
        "Download requirements as Markdown",
        data=markdown_text,
        file_name="whiteboard_requirements.md",
        mime="text/markdown",
        use_container_width=True,
    )


def render_sidebar() -> None:
    st.sidebar.title("🧩 Whiteboard BA Generator")
    st.sidebar.markdown("Convert a whiteboard sketch into BA-ready requirements, user stories, and Mermaid diagrams.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("API key status")
    st.sidebar.markdown(
        f"""
        <div class="{'status-pill-ok' if GOOGLE_API_KEY else 'status-pill-warn'}">
            {'✅' if GOOGLE_API_KEY else '⚠️'} GOOGLE_API_KEY
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        f"""
        <div class="{'status-pill-ok' if DEEPSEEK_API_KEY else 'status-pill-warn'}">
            {'✅' if DEEPSEEK_API_KEY else '⚠️'} DEEPSEEK_API_KEY
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar.expander("Model configuration", expanded=False):
        st.write(f"Vision: `{VISION_MODEL_ID}`")
        st.write(f"Text: `{TEXT_MODEL_ID}`")
        if NATIVE_DEEPSEEK_AVAILABLE:
            st.caption("Using native Agno DeepSeek model class.")
        else:
            st.caption(f"Using OpenAI-compatible fallback: `{DEEPSEEK_BASE_URL}`")

    st.sidebar.markdown("---")
    st.sidebar.subheader("How it works")
    st.sidebar.markdown(
        """
        1. Upload a whiteboard image.
        2. Gemini analyzes the sketch and asks 3-5 clarifying questions.
        3. You answer the questions.
        4. DeepSeek generates requirements, user stories, diagrams, and gaps.
        """
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠 More AI tools for BAs")
    st.sidebar.markdown(
        """
        <div style="background: linear-gradient(135deg, #eff6ff, #fafbff); border: 1px solid #bfdbfe; border-radius: 14px; padding: 0.9rem 1rem; margin-bottom: 0.75rem;">
            <strong style="font-size:0.95rem;">🧩 BA Assistant</strong><br>
            <span style="font-size:0.82rem; color:#475569;">AI-powered requirements analysis — structured specs, user stories, NFRs, Mermaid diagrams in 60s.</span><br>
            <a href="https://touseefshaik.com/tools/ba-assistant" target="_blank" style="font-size:0.82rem;">Try it →</a>
        </div>
        <div style="background: linear-gradient(135deg, #fff7ed, #fffbf7); border: 1px solid #fed7aa; border-radius: 14px; padding: 0.9rem 1rem; margin-bottom: 0.75rem;">
            <strong style="font-size:0.95rem;">🎙️ Sarvam Voice AI</strong><br>
            <span style="font-size:0.82rem; color:#475569;">Text-to-speech in 11 Indian languages.</span><br>
            <a href="https://touseefshaik.com/tools/sarvam-voice-ai" target="_blank" style="font-size:0.82rem;">Try it →</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    if st.sidebar.button("Reset app", use_container_width=True):
        reset_all_state()
        st.rerun()


def render_progress() -> None:
    has_image = bool(st.session_state.image_bytes)
    has_questions = bool(st.session_state.questions)
    has_answers = has_questions and all(
        st.session_state.answers.get(str(idx), "").strip()
        for idx in range(len(st.session_state.questions))
    )
    has_output = bool(st.session_state.final_output)

    progress_value = 0
    if has_image:
        progress_value = 25
    if has_questions:
        progress_value = 50
    if has_answers:
        progress_value = 75
    if has_output:
        progress_value = 100

    st.progress(progress_value, text=f"Progress: {progress_value}%")


def main() -> None:
    initialize_session_state()
    render_sidebar()

    st.markdown(
        """
        <div class="hero-card">
          <div class="hero-title">Whiteboard to Requirements Generator</div>
          <div class="hero-subtitle">
            Upload a process flow, system diagram, or workflow sketch. The app analyzes the image,
            asks targeted clarification questions, and generates BA artifacts for delivery teams.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_progress()

    if st.session_state.last_error:
        st.warning(st.session_state.last_error)

    left_col, right_col = st.columns([0.9, 1.1], gap="large")

    with left_col:
        st.markdown("### 1️⃣ Upload Image")
        uploaded_file = st.file_uploader(
            "Upload a whiteboard sketch",
            type=["png", "jpg", "jpeg"],
            help="Use a clear image with readable labels and arrows where possible.",
        )

        if uploaded_file is not None:
            is_valid, validation_message = validate_image(uploaded_file)
            if not is_valid:
                st.warning(validation_message)
            else:
                image_bytes = uploaded_file.getvalue()
                if st.session_state.image_filename != uploaded_file.name:
                    reset_analysis_state()
                    st.session_state.image_bytes = image_bytes
                    st.session_state.image_mime_type = uploaded_file.type
                    st.session_state.image_filename = uploaded_file.name

                preview = PILImage.open(BytesIO(image_bytes))
                st.image(preview, caption=uploaded_file.name, use_container_width=True)

                st.markdown(
                    "<div class='small-muted'>Tip: Better image contrast and readable labels improve the quality of the generated questions and requirements.</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("Upload an image to start the analysis flow.")

    with right_col:
        st.markdown("### 2️⃣ Analyze Whiteboard")
        st.markdown(
            "<div class='step-card'>Gemini reads the sketch and produces a concise analysis plus targeted clarifying questions.</div>",
            unsafe_allow_html=True,
        )

        can_analyze = bool(st.session_state.image_bytes) and bool(GOOGLE_API_KEY)
        analyze_label = "Retry Analyze Whiteboard" if st.session_state.vision_analysis else "Analyze Whiteboard"
        if st.button(analyze_label, disabled=not can_analyze, use_container_width=True):
            with st.spinner("Analyzing the whiteboard and generating clarification questions..."):
                try:
                    analyze_whiteboard()
                    st.rerun()
                except Exception as exc:
                    st.session_state.last_error = f"Vision agent failed: {exc}"
                    st.error(st.session_state.last_error)

        if not GOOGLE_API_KEY:
            st.error("Add GOOGLE_API_KEY to Streamlit secrets or your environment to enable image analysis.")

        if st.session_state.vision_analysis:
            st.markdown("#### Image Analysis Summary")
            st.markdown(st.session_state.vision_analysis)

            if st.session_state.detected_elements:
                with st.expander("Detected elements", expanded=False):
                    for element in st.session_state.detected_elements:
                        st.markdown(f"- {element}")

        if st.session_state.questions:
            st.markdown("### 3️⃣ Answer Clarifying Questions")
            st.markdown(
                "<div class='step-card'>Answer each question so the requirements agent can reduce assumptions and generate stronger BA artifacts.</div>",
                unsafe_allow_html=True,
            )

            for idx, question in enumerate(st.session_state.questions):
                key = str(idx)
                st.session_state.answers[key] = st.text_input(
                    f"Q{idx + 1}. {question}",
                    value=st.session_state.answers.get(key, ""),
                    key=f"answer_{idx}",
                    placeholder="Type your answer here...",
                )

            st.markdown("### 4️⃣ Generate Requirements")
            if not DEEPSEEK_API_KEY:
                st.error("Add DEEPSEEK_API_KEY to Streamlit secrets or your environment to enable requirements generation.")

            if st.button("Generate Requirements", disabled=not DEEPSEEK_API_KEY, use_container_width=True):
                with st.spinner("Generating BA artifacts, requirements, user stories, and Mermaid diagrams..."):
                    try:
                        generate_requirements()
                        st.rerun()
                    except Exception as exc:
                        st.session_state.last_error = f"Requirements generation failed: {exc}"
                        st.error(st.session_state.last_error)

        if st.session_state.final_output:
            st.markdown("---")
            st.markdown("## Generated BA Artifacts")
            render_final_output(st.session_state.final_output)
            st.markdown("---")
            st.markdown(
                """
                <div style="background: linear-gradient(135deg, #eff6ff, #f0f7ff); border: 1px solid #93c5fd; border-radius: 16px; padding: 1.5rem 2rem; margin-top: 1rem;">
                    <h3 style="margin:0 0 0.4rem 0; font-size:1.15rem;">Want deeper analysis?</h3>
                    <p style="margin:0 0 1rem 0; color:#475569; font-size:0.92rem;">
                        BA Assistant goes further — NFRs, risk assessment, architecture notes, and full BRD processing. Free to start.
                    </p>
                    <a href="https://touseefshaik.com/tools/ba-assistant" target="_blank" style="display:inline-block; background:#2563eb; color:#fff; padding:0.6rem 1.4rem; border-radius:100px; font-weight:600; text-decoration:none; font-size:0.9rem;">Try BA Assistant →</a>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        "<div class='footer'>Built with Agno + Gemini 3.5 Flash + DeepSeek  ·  By <a href='https://touseefshaik.com' target='_blank'>Touseef Shaik</a></div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
