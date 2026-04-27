import os
import sys
import shutil
import glob
import streamlit as st
from dotenv import load_dotenv

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Educational Document Insight Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #e8e8f0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(255,255,255,0.04);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* Cards */
    .glass-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
        transition: box-shadow 0.3s ease;
    }
    .glass-card:hover {
        box-shadow: 0 8px 32px rgba(99,102,241,0.25);
    }

    /* PDF badge */
    .pdf-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(99,102,241,0.18);
        border: 1px solid rgba(99,102,241,0.4);
        border-radius: 8px;
        padding: 6px 12px;
        margin: 4px 4px 4px 0;
        font-size: 0.85rem;
        color: #a5b4fc;
        font-weight: 500;
    }

    /* Result box */
    .result-box {
        background: rgba(16,185,129,0.08);
        border: 1px solid rgba(16,185,129,0.3);
        border-radius: 12px;
        padding: 20px;
        white-space: pre-wrap;
        font-size: 0.92rem;
        line-height: 1.7;
        color: #d1fae5;
    }

    .error-box {
        background: rgba(239,68,68,0.10);
        border: 1px solid rgba(239,68,68,0.35);
        border-radius: 12px;
        padding: 16px;
        color: #fca5a5;
        font-size: 0.9rem;
    }

    /* Section headings */
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #c7d2fe;
        margin-bottom: 10px;
        letter-spacing: 0.5px;
    }

    /* Streamlit button overrides */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.2s ease !important;
        border: none !important;
    }

    /* Primary action buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(99,102,241,0.4) !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(99,102,241,0.55) !important;
    }

    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 10px !important;
        color: #000000 !important;
    }

    /* Spinner text */
    .stSpinner > div > div {
        color: #a5b4fc !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.04) !important;
        border-radius: 10px !important;
        color: #c7d2fe !important;
    }

    /* Divider */
    hr { border-color: rgba(255,255,255,0.1); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Constants ────────────────────────────────────────────────────────────────
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ── Session state defaults ───────────────────────────────────────────────────
for key, default in {
    "result_summary": None,
    "result_insights": None,
    "result_mcq": None,
    "rag_initialized": False,
    "rag_retriever": None,
    "rag_file_set": set(),
    "error": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_uploaded_pdfs():
    """Return list of PDF file names currently in the data directory."""
    return sorted([os.path.basename(p) for p in glob.glob(os.path.join(DATA_DIR, "*.pdf"))])


def save_uploaded_file(uploaded_file):
    dest = os.path.join(DATA_DIR, uploaded_file.name)
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest


def clear_results():
    """Clear all stored results from session state."""
    st.session_state.result_summary = None
    st.session_state.result_insights = None
    st.session_state.result_mcq = None
    st.session_state.error = None


@st.cache_resource(show_spinner=False)
def init_rag(file_tuple):
    """Build/rebuild the RAG pipeline. Cached by the set of PDF filenames."""
    from src.rag_pipeline import setup_rag_pipeline
    return setup_rag_pipeline(data_dir=DATA_DIR)


def get_crew():
    from src.agents import EducationCrew
    return EducationCrew()


def run_agent(operation: str, topic: str = "", question: str = ""):
    """Run a single-operation crew task and return the text result."""
    from crewai import Agent, Task, Crew, Process
    from src.config import get_crewai_llm
    from src.tools import query_documents

    llm = get_crewai_llm()

    if operation == "summarize":
        agent = Agent(
            role="Senior Document Summarizer",
            goal="Provide clear, concise, and comprehensive summaries of the entire document contents.",
            backstory=(
                "You are an expert educator who extracts core concepts from complex textbooks. "
                "You never invent information; you only use what is in the source text. "
                "You summarize ALL the content you find, covering every major section and key point."
            ),
            verbose=False,
            allow_delegation=False,
            tools=[query_documents],
            llm=llm,
        )
        task = Task(
            description=(
                "Search and retrieve ALL content from the uploaded documents. "
                "Provide a comprehensive summary of the entire document, covering all major sections, "
                "key concepts, definitions, main ideas, conclusions, and critical examples. "
                "Do not focus on a single topic — summarize everything."
            ),
            expected_output="A well-structured, comprehensive summary (4-6 paragraphs) of the entire document content.",
            agent=agent,
        )

    elif operation == "insights":
        question_text = question or f"What are the key insights about {topic}?"
        agent = Agent(
            role="Question Answering Specialist",
            goal="Answer user questions accurately based ONLY on the retrieved reference documents.",
            backstory=(
                "You are a meticulous teaching assistant who finds exact answers from textbooks. "
                "If the answer is not in the text, you clearly state it is missing."
            ),
            verbose=False,
            allow_delegation=False,
            tools=[query_documents],
            llm=llm,
        )
        task = Task(
            description=(
                f'Based on documents about "{topic}", answer: "{question_text}". '
                "Search documents for keywords in the question. "
                'If not found, state: "I could not find the answer in the provided materials."'
            ),
            expected_output="A clear, direct answer citing concepts from the text, or a statement that the answer is not found.",
            agent=agent,
        )

    elif operation == "mcq":
        agent = Agent(
            role="Assessment Creator",
            goal="Generate challenging MCQs with correct answers and explanations based on the document text.",
            backstory=(
                "You are a seasoned instructional designer who creates MCQs testing true comprehension. "
                "Every question includes the correct answer and a brief explanation referencing the source."
            ),
            verbose=False,
            allow_delegation=False,
            tools=[query_documents],
            llm=llm,
        )
        task = Task(
            description=(
                f'Based on information about "{topic}", generate 5 Multiple Choice Questions (MCQs).\n'
                "Requirements for each MCQ:\n"
                "1. A clear question testing understanding.\n"
                "2. 4 options labeled A, B, C, D.\n"
                "3. Clearly indicate the correct option.\n"
                "4. Provide a 1-2 sentence explanation of WHY the option is correct based on the text."
            ),
            expected_output="A formatted list of 5 MCQs with options, the correct answer, and an explanation for each.",
            agent=agent,
        )
    else:
        raise ValueError(f"Unknown operation: {operation}")

    crew = Crew(agents=[agent], tasks=[task], verbose=False, process=Process.sequential)
    result = crew.kickoff()
    # CrewAI may return a CrewOutput object; convert to string
    return str(result)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📚 Doc Insight Assistant")
    st.markdown("---")

    # --- Upload section ---
    st.markdown('<p class="section-title">📂 Upload PDF</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        label_visibility="collapsed",
    )
    if uploaded is not None:
        save_uploaded_file(uploaded)
        st.success(f"✅ **{uploaded.name}** uploaded!")
        # Invalidate cache so RAG re-initialises with new docs
        init_rag.clear()

    st.markdown("---")

    # --- PDF list ---
    st.markdown('<p class="section-title">📄 Uploaded Documents</p>', unsafe_allow_html=True)
    pdfs = get_uploaded_pdfs()
    if pdfs:
        for pdf in pdfs:
            st.markdown(f'<div class="pdf-badge">📄 {pdf}</div>', unsafe_allow_html=True)
    else:
        st.caption("No PDFs uploaded yet.")

    st.markdown("---")

    # --- Model info ---
    st.markdown('<p class="section-title">⚙️ Model Config</p>', unsafe_allow_html=True)
    st.caption("LLM: llama3 (Ollama)")
    st.caption("Embeddings: nomic-embed-text")
    st.caption("Vector store: FAISS")


# ── Main area ────────────────────────────────────────────────────────────────
st.markdown("# 🎓 Educational Document Insight Assistant")
st.markdown(
    "Upload PDFs and let the multi-agent crew analyse your documents."
)
st.markdown("---")

# ── Inputs ───────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">🔍 Topic <span style="font-size:0.8rem;color:#818cf8;">(for Insights / MCQ)</span></p>', unsafe_allow_html=True)
    topic = st.text_input(
        "Topic",
        placeholder="e.g. Machine Learning, Neural Networks …",
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">❓ Specific Question <span style="font-size:0.8rem;color:#818cf8;">(for Insights)</span></p>', unsafe_allow_html=True)
    question = st.text_input(
        "Question",
        placeholder="e.g. What is gradient descent?",
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ── Action buttons ────────────────────────────────────────────────────────────
st.markdown("### ⚡ Agent Operations")
b1, b2, b3, b4 = st.columns(4, gap="medium")

run_summary = b1.button("📝 Summarize", use_container_width=True, type="primary")
run_insights = b2.button("💡 Insights / Q&A", use_container_width=True, type="primary")
run_mcq = b3.button("📋 Generate MCQs", use_container_width=True, type="primary")
clear_btn = b4.button("🗑️ Clear Results", use_container_width=True)

if clear_btn:
    clear_results()

# ── Validation helper ─────────────────────────────────────────────────────────
def validate_inputs(require_topic=False, require_question=False):
    if not get_uploaded_pdfs():
        st.warning("⚠️ Please upload at least one PDF document first.")
        return False
    if require_topic and not topic.strip():
        st.warning("⚠️ Please enter a **topic** before running this agent.")
        return False
    if require_question and not question.strip():
        st.warning("⚠️ Please enter a **specific question** for Insights / Q&A.")
        return False
    return True

# ── Summarize ─────────────────────────────────────────────────────────────────
if run_summary:
    if validate_inputs():
        clear_results()
        with st.spinner("🤖 Summarizer agent is working…"):
            try:
                st.session_state.result_summary = run_agent("summarize")
                st.session_state.error = None
            except Exception as e:
                st.session_state.error = str(e)

# ── Insights / Q&A ────────────────────────────────────────────────────────────
if run_insights:
    if validate_inputs(require_topic=True, require_question=True):
        clear_results()
        with st.spinner("🤖 Q&A agent is working…"):
            try:
                st.session_state.result_insights = run_agent("insights", topic, question)
                st.session_state.error = None
            except Exception as e:
                st.session_state.error = str(e)

# ── MCQ ───────────────────────────────────────────────────────────────────────
if run_mcq:
    if validate_inputs(require_topic=True):
        clear_results()
        with st.spinner("🤖 MCQ generator agent is working…"):
            try:
                st.session_state.result_mcq = run_agent("mcq", topic)
                st.session_state.error = None
            except Exception as e:
                st.session_state.error = str(e)

# ── Error display ─────────────────────────────────────────────────────────────
if st.session_state.error:
    st.markdown(
        f'<div class="error-box">❌ <strong>Error:</strong> {st.session_state.error}</div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Results ───────────────────────────────────────────────────────────────────
has_result = any([
    st.session_state.result_summary,
    st.session_state.result_insights,
    st.session_state.result_mcq,
])

if has_result:
    st.markdown("## 📊 Results")

    if st.session_state.result_summary:
        with st.expander("📝 Summary", expanded=True):
            st.markdown(
                f'<div class="result-box">{st.session_state.result_summary}</div>',
                unsafe_allow_html=True,
            )
            st.download_button(
                "⬇️ Download Summary",
                data=st.session_state.result_summary,
                file_name="summary.txt",
                mime="text/plain",
            )

    if st.session_state.result_insights:
        with st.expander("💡 Insights / Q&A", expanded=True):
            st.markdown(
                f'<div class="result-box">{st.session_state.result_insights}</div>',
                unsafe_allow_html=True,
            )
            st.download_button(
                "⬇️ Download Insights",
                data=st.session_state.result_insights,
                file_name=f"insights_{topic.replace(' ','_')}.txt",
                mime="text/plain",
            )

    if st.session_state.result_mcq:
        with st.expander("📋 Multiple Choice Questions", expanded=True):
            st.markdown(
                f'<div class="result-box">{st.session_state.result_mcq}</div>',
                unsafe_allow_html=True,
            )
            st.download_button(
                "⬇️ Download MCQs",
                data=st.session_state.result_mcq,
                file_name=f"mcqs_{topic.replace(' ','_')}.txt",
                mime="text/plain",
            )
else:
    st.markdown(
        """
        <div class="glass-card" style="text-align:center; padding: 48px;">
            <div style="font-size:3rem;">🤖</div>
            <p style="color:#a5b4fc; font-size:1.1rem; margin-top:12px;">
                Upload a PDF and hit <strong>Summarize</strong> to get a full overview, or enter a topic for Insights / MCQ.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
