# 🧠 Advanced RAG-Based MCQ Generator

An AI-powered multiple-choice question generator built with **Python, LangChain, RAG, FAISS, OpenAI-compatible LLM APIs, Pydantic, and Streamlit**.

The application accepts educational content as **PDF, DOCX, or plain text**, retrieves relevant content using a vector store, and generates context-aware MCQs based on a selected difficulty level and **Bloom's Taxonomy** level. Users can then take an interactive quiz and receive their score and explanations.

## ✨ Features

- Upload **PDF** or **DOCX** documents
- Enter raw text directly
- Generate **1–20 MCQs**
- Select Easy, Medium, or Hard difficulty
- Select a Bloom's Taxonomy level: Remember, Understand, Apply, Analyze, Evaluate, or Create
- Retrieval-Augmented Generation (RAG)
- Recursive text chunking with overlap
- Semantic embeddings using `text-embedding-3-small`
- FAISS vector database for similarity retrieval
- LLM-based MCQ generation
- Pydantic-based output validation
- Semantic deduplication of similar questions
- Interactive quiz mode
- Automatic scoring and answer explanations
- Streamlit user interface

## 🏗️ Architecture

```text
PDF / DOCX / Text
       │
       ▼
Document Loading & Validation
       │
       ▼
Text Chunking
(chunk size 1000, overlap 150)
       │
       ▼
Embeddings
(text-embedding-3-small)
       │
       ▼
FAISS Vector Store
       │
       ▼
Top-K Relevant Chunks (K=5)
       │
       ▼
Prompt Builder
(Difficulty + Bloom's Taxonomy)
       │
       ▼
LLM Generation
       │
       ▼
JSON Parsing + Pydantic Validation
       │
       ▼
Semantic MCQ Deduplication
       │
       ▼
Streamlit MCQ / Quiz Interface
```

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Language | Python |
| LLM Framework | LangChain |
| Generative AI | OpenAI-compatible Chat API |
| RAG | LangChain Retrieval + FAISS |
| Embeddings | `text-embedding-3-small` |
| Vector Database | FAISS |
| Document Processing | PyPDF, python-docx |
| Validation | Pydantic |
| Similarity | Cosine similarity, scikit-learn |
| Frontend | Streamlit |
| Configuration | python-dotenv |
| Version Control | Git / GitHub |

## 📁 Project Structure

```text
advanced_rag_mcq_generator/
│
├── app.py
├── requirements.txt
├── .gitignore
│
└── core/
    ├── __init__.py
    ├── loaders.py
    ├── vectorstore.py
    ├── prompt.py
    ├── generator.py
    ├── schema.py
    ├── deduplicator.py
    └── evaluator.py
```

### Module Overview

- **`app.py`** — Streamlit UI and orchestration of the complete pipeline.
- **`core/loaders.py`** — PDF/DOCX text extraction and raw-text validation.
- **`core/vectorstore.py`** — Text splitting, embeddings, FAISS indexing, and top-K retrieval.
- **`core/prompt.py`** — Structured MCQ prompt generation using difficulty and Bloom's Taxonomy.
- **`core/generator.py`** — LLM invocation, JSON parsing, and Pydantic validation.
- **`core/schema.py`** — Pydantic models for structured MCQ output.
- **`core/deduplicator.py`** — Embedding-based semantic similarity filtering.
- **`core/evaluator.py`** — Reserved module; currently contains no implementation.

## 🔄 How the RAG Pipeline Works

1. The user uploads a PDF/DOCX file or enters text.
2. The application extracts and validates the content.
3. Content is split into chunks using `RecursiveCharacterTextSplitter`.
4. Each chunk is converted into an embedding.
5. Embeddings are stored in a FAISS vector store.
6. The application retrieves the top 5 relevant chunks.
7. Retrieved content is inserted into a structured MCQ-generation prompt.
8. The LLM generates MCQs in JSON format.
9. Pydantic validates the generated structure.
10. Semantically similar questions are filtered using cosine similarity.
11. The final MCQs are displayed in Streamlit.
12. The user can take an interactive quiz and receive immediate feedback and a final score.

## 🔐 Environment Setup

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
```

**Never commit your API key or `.env` file to GitHub.**

The application uses an OpenAI-compatible API endpoint configured in the source code.

## 🚀 Installation

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd advanced_rag_mcq_generator-main
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your API key

Create `.env`:

```env
OPENAI_API_KEY=your_api_key_here
```

### 5. Run

```bash
streamlit run app.py
```

## 🎮 Usage

1. Enter text or upload a PDF/DOCX document.
2. Select the number of questions.
3. Select difficulty.
4. Select a Bloom's Taxonomy level.
5. Click **Generate MCQs**.
6. Review the generated questions, answers, and explanations.
7. Click **Start Quiz** to attempt the questions.
8. Submit answers to receive feedback.
9. View the final score after completing the quiz.

## 🔍 Semantic Deduplication

Generated questions are checked for semantic similarity using:

- `text-embedding-3-small`
- Cosine similarity
- Default similarity threshold: **0.85**

Questions that are too similar to an already accepted question are filtered from the final result.

## 🧩 Error Handling

The application defines custom exceptions for major pipeline stages:

- `DocumentLoaderError`
- `VectorStoreError`
- `PromptBuilderError`
- `LLMGenerationError`
- `DeduplicationError`

This helps the application return meaningful error messages for common failures.

## 📌 Current Limitations

- The FAISS vector store is created in memory for each generation request.
- The current retrieval query is fixed rather than dynamically derived from the user's topic.
- The application requires an external LLM/embedding API.
- Scanned/image-only PDFs without extractable text are not supported by the current PDF loader.
- The current DOCX loader extracts paragraph text but does not process tables.
- Automated tests are not currently included.
- `core/evaluator.py` is currently reserved and has no implementation.

## 🔮 Possible Future Improvements

- Add automated unit and integration tests
- Persist FAISS indexes between sessions
- Add support for additional document formats
- Add table/image extraction
- Improve retrieval with metadata filtering and configurable strategies
- Add automated MCQ quality evaluation
- Make LLM and embedding models configurable
- Add deployment configuration
- Include source references for generated questions

## 👨‍💻 Author

**Rakesh Kumar Pandeeti**

Computer Science | Generative AI | RAG | Python
