# Educational Document Insight Assistant

This is a comprehensive training project designed to derive deep insights from educational documents (PDFs, CSVs) using Large Language Models (LLMs). It features a multi-agent workflow powered by CrewAI and a Retrieval-Augmented Generation (RAG) pipeline built with Langchain.

## Objective
To process textbook or reference book materials and automatically generate:
1.  **Document Summaries:** Clear and concise overviews of specified topics.
2.  **Question Answering:** Precise answers to specific user questions based purely on the provided text.
3.  **MCQ Generation:** Multiple Choice Questions with correct answers and explanations for self-assessment.

## Project Structure
*   `main.py`: The entry point for the application.
*   `requirements.txt`: Project dependencies.
*   `.env.example`: Template for environment variables.
*   `data/`: Directory where you should place your `.pdf` and `.csv` files.
*   `src/`:
    *   `config.py`: Configuration for the local LLM (`llama3` via Ollama) and Embeddings (`nomic-embed-text`).
    *   `rag_pipeline.py`: Code to ingest documents (Python pre-processing), chunk them, and build the FAISS vector database.
    *   `tools.py`: Custom Langchain tools that allow CrewAI agents to query the RAG pipeline.
    *   `agents.py`: Definition of the Multi-Agent workflow using CrewAI, including role definitions and prompt engineering.

## Setup Instructions

### 1. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Ollama (Local LLM)
This project uses local models via Ollama. No API keys are required!

1. Install [Ollama](https://ollama.com/) for your operating system.
2. Start the Ollama server (or ensure it's running in the background).
3. Pull the required models for generation and embeddings:
   ```bash
   ollama pull llama3
   ollama pull nomic-embed-text
   ```

### 4. Add Your Data
Place any educational PDF documents (textbooks, reference materials) or CSV data files into the `data/` directory.

### 5. Run the Application
```bash
python main.py
```

## How It Works (The Technologies Used)

1.  **Python for Data Pre-Processing**: Langchain document loaders (`PyPDFLoader`, `CSVLoader`) are used in `src/rag_pipeline.py` to ingest the raw files.
2.  **RAG Pipeline**: The text is split into chunks using `RecursiveCharacterTextSplitter`. These chunks are converted into embeddings and stored in a local `FAISS` vector database. This allows for semantic search across the documents.
3.  **Prompt Engineering**: Located in `src/agents.py`, specific system prompts (backstories, goals) and task descriptions are crafted to ensure the agents:
    *   Rely *only* on the provided documents.
    *   Format output correctly (e.g., specific MCQ formats).
    *   Understand their specific educational roles (Summarizer vs. QA Specialist vs. Assessor).
4.  **Agentic Workflow**: CrewAI coordinates three specialized agents:
    *   **Document Summarizer**: Extracts the core concepts.
    *   **QA Specialist**: Answers the specific user question.
    *   **Assessment Creator**: Generates 3 MCQs.
    These agents work sequentially (`Process.sequential`), sharing information to produce the final comprehensive report.
