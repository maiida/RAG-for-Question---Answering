# RAG for Question Answering

A Retrieval-Augmented Generation (RAG) system that detects contradictions and ambiguities in retrieved context before answering. When conflicting information is found, the system asks the user for clarification to produce a more precise answer.

## How it works

Two modes are available:

- **naive** — standard RAG: retrieve chunks → generate answer
- **judge** — retrieves chunks, runs an LLM judge to detect contradictions, asks for clarification if needed, then generates the final answer

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/maiida/RAG-for-Question-Answering.git
cd RAG-for-Question-Answering
```

**2. Create and activate a virtual environment**
```bash
# Create
python -m venv .venv

# Activate — Linux/macOS
source .venv/bin/activate

# Activate — Windows
.venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

No HuggingFace token required. Models and the vector store index are built automatically on first run.

## Usage

```bash
# Basic RAG
python main.py "What happens if the QRC fails during boot?"

# With contradiction detection
python main.py "Explain the ambiguity around Safe Mode" --mode judge

# Different chunking strategy
python main.py "Who can override network restrictions?" --mode judge --chunking markdown

# Rebuild vector store
python main.py "Your question" --rebuild

# Different model
python main.py "Your question" --model "Qwen/Qwen3.5-2B"
```

## Project structure

```
├── main.py                    # CLI entry point
├── engine.py                  # RAGEngine class (retrieval + generation)
├── ingest.py                  # Build vector store
├── config.py                  # Config
├── evaluate.py                # evaluate_doc_level(), plot_metrics()
├── docs/                      # Source documents
├── evaluation/
│   └── evaluation.json        # benchmark questions + expected sources
├── analysis/
│   └── token_analysis.py      # token count analysis per document
└── requirements.txt
```
