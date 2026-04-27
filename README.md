# Educational Document Insight Assistant

A multi-agent document analysis tool powered by **CrewAI**, **LangChain**, and a **FAISS RAG pipeline**. It features a clean **Streamlit web UI** for uploading PDFs and running specialized AI agents directly from your browser.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📂 **PDF Upload** | Upload PDFs via the sidebar; files are saved to `data/` automatically |
| 📄 **Document List** | Sidebar shows all currently loaded PDFs |
| 📝 **Summarize** | Summarizes the entire document — no topic required |
| 💡 **Insights / Q&A** | Answers a specific question on a given topic |
| 📋 **Generate MCQs** | Creates 5 MCQs with options, correct answers, and explanations |
| ⬇️ **Download Results** | Export any result as a `.txt` file |

---

## 🗂️ Project Structure

```
poc_pdfcheck/
├── app.py               # Streamlit UI (main entry point)
├── main.py              # Original CLI entry point
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
├── data/                # Place PDF/CSV files here (auto-created)
└── src/
    ├── config.py        # LLM & embedding model configuration (Ollama)
    ├── rag_pipeline.py  # Document ingestion, chunking, FAISS vector store
    ├── tools.py         # LangChain tool wrapping the RAG retriever
    └── agents.py        # CrewAI agents (Summarizer, QA, MCQ Generator)
```

---

## 🚀 Setup & Running

### 1. Create a Virtual Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Setup Ollama (Local LLM)

This project runs **entirely locally** — no API keys required.

1. Install [Ollama](https://ollama.com/) for your OS.
2. Start the Ollama server.
3. Pull the required models:
   ```bash
   ollama pull llama3
   ollama pull nomic-embed-text
   ```

### 4. Run the Streamlit UI

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501**.

> **Alternative (CLI):** `python main.py`

---

## 🖥️ Using the UI

1. **Upload a PDF** using the sidebar upload button.
2. The PDF will appear in the **Uploaded Documents** list.
3. Click **📝 Summarize** — summarizes the entire document (no topic needed).
4. For **💡 Insights / Q&A** — enter a **Topic** and a **Specific Question**, then click the button.
5. For **📋 Generate MCQs** — enter a **Topic**, then click the button.
6. Results appear below with a **Download** button for each.

---

## ⚙️ How It Works

1. **RAG Pipeline** (`src/rag_pipeline.py`): PDFs are loaded with `PyPDFLoader`, chunked with `RecursiveCharacterTextSplitter`, embedded via `nomic-embed-text`, and stored in a local **FAISS** vector store.
2. **Prompt Engineering** (`src/agents.py`): Each agent has a crafted role, goal, and backstory to ensure it stays grounded in document content.
3. **Agentic Workflow** (CrewAI): Each button triggers a single-agent `Crew` running sequentially, keeping operations independent and fast.

---

## 🔧 Configuration

Edit `src/config.py` to switch LLM or embedding models:

```python
# Example: switch to a different Ollama model
return ChatOllama(model="mistral", temperature=0.2)
```
