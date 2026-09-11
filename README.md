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
├── .env.example
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
- **`core/generator.py`** — Provider-aware LLM invocation, JSON parsing, and Pydantic validation.
- **`core/provider.py`** — Shared OpenAI/OpenRouter configuration for chat and embedding models.
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

🔐 Environment Setup

This project uses environment variables for API configuration. Never
hard-code an API key in Python files, README.md, .env.example, or
any file that will be uploaded to GitHub.

.env.example

The repository should contain a .env.example file as a safe
configuration template:

# Choose: openrouter or openai
AI_PROVIDER=openrouter

# Add your own API key
OPENAI_API_KEY=your_api_key_here

# Optional model overrides
# CHAT_MODEL=
# EMBEDDING_MODEL=

Create your local .env

After cloning the repository, create a file named .env in the
project root, in the same directory as app.py.

Copy the values from .env.example and replace the placeholder with
your own API key.

Using OpenRouter

AI_PROVIDER=openrouter
OPENAI_API_KEY=your_openrouter_api_key_here

Using OpenAI directly

AI_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here

The application supports both providers without requiring changes to the
Python source code. The provider is selected through AI_PROVIDER.

Optional model overrides:

CHAT_MODEL=your_chat_model
EMBEDDING_MODEL=text-embedding-3-small

If model overrides are not provided, the application uses its configured
provider defaults.

.env vs .env.example

File Purpose Upload to GitHub?

.env.example Safe configuration      ✅ Yes
template with
placeholder values

Your .gitignore should contain:

.env
venv/
__pycache__/

Never commit, share, or paste your real API key into GitHub, README
files, screenshots, or public discussions.

If an API key is accidentally exposed, revoke/rotate it with the
provider immediately and create a new key.

🚀 Installation

Python Version

Recommended: Python 3.12.x

The project was developed/tested with Python 3.12. Using a supported
64-bit Python 3.12 installation is recommended for the best
compatibility with packages that contain native components, such as
FAISS and lxml.

Check your Python version:

python --version

On Windows, if multiple Python versions are installed:

py -0p
py -3.12 --version

1. Clone the repository

git clone <your-github-repository-url>
cd advanced_rag_mcq_generator-main

2. Create a virtual environment

Using Python 3.12:

Windows:

py -3.12 -m venv venv

Activate it:

.env\Scripts\Activate.ps1

If PowerShell blocks local script execution, you can allow locally
created scripts for your current Windows user:

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

Then activate the environment again:

.env\Scripts\Activate.ps1

Linux/macOS:

python3.12 -m venv venv
source venv/bin/activate

After activation, verify that the virtual environment is being used:

python --version
python -c "import sys; print(sys.executable)"

3. Install dependencies

python -m pip install --upgrade pip
pip install -r requirements.txt

4. Configure the AI provider

Create .env in the project root, next to app.py.

For OpenRouter:

AI_PROVIDER=openrouter
OPENAI_API_KEY=your_openrouter_api_key_here

For OpenAI directly:

AI_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here

No Python source-code changes are required when switching between these
two supported providers.

5. Run the application

python -m streamlit run app.py

Then open the local URL shown by Streamlit, usually:

http://localhost:8501


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
- The application requires an external LLM/embedding API from OpenAI or OpenRouter.
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
