# 🧠 RAGForge
## 🚀 Live Demo

👉 [Launch RAGForge](https://ragforge1.streamlit.app/)

### Adaptive Hybrid Retrieval-Augmented Generation

RAGForge is a modular Retrieval-Augmented Generation (RAG) application that allows users to upload documents and ask questions about their content.

The system combines semantic query routing, dense vector retrieval, BM25 keyword retrieval, Reciprocal Rank Fusion (RRF), CrossEncoder reranking, and LLM-based generation to produce grounded answers with source references.

---

## 🚀 Features

- 📄 Upload PDF, DOCX, TXT, and Markdown documents
- 🧭 Semantic query routing
- 🔎 Dense semantic retrieval
- 🔤 BM25 keyword retrieval
- 🔀 Hybrid retrieval using Reciprocal Rank Fusion (RRF)
- 🏆 CrossEncoder reranking
- 🤖 LLM-powered answer generation
- 💬 Conversation-aware follow-up questions
- 📚 Source/chunk display for retrieved context
- 🖥️ Streamlit web interface
- 🔑 Environment-based API key management
- 🧩 Modular retrieval and generation architecture

---

## 🏗️ Architecture

```text
                         ┌──────────────────┐
                         │    User Query    │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Semantic Router  │
                         └────────┬─────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
                    ▼             ▼             ▼
                Greeting       Casual       Document
                    │             │             │
                    ▼             ▼             ▼
              Direct Reply       LLM      Hybrid Retrieval
                                                │
                                  ┌─────────────┴─────────────┐
                                  │                           │
                                  ▼                           ▼
                          Dense Retrieval                 BM25
                                  │                           │
                                  └─────────────┬─────────────┘
                                                │
                                                ▼
                                      Reciprocal Rank Fusion
                                                │
                                                ▼
                                         CrossEncoder
                                          Reranking
                                                │
                                                ▼
                                         Top Documents
                                                │
                                                ▼
                                          LLM Generator
                                                │
                                                ▼
                                      Grounded Answer
                                                │
                                                ▼
                                             Sources
```

---

## 🔍 How It Works

### 1. Document Ingestion

Users upload documents through the Streamlit interface.

Supported formats:

- PDF
- DOCX
- TXT
- Markdown

The documents are loaded and divided into smaller chunks before being indexed.

### 2. Semantic Query Routing

The incoming query is first classified into one of three categories:

- `greeting`
- `casual`
- `document`

Document-related queries continue through the retrieval pipeline, while greetings and casual conversations can bypass retrieval.

The router also receives conversation history so contextual follow-up queries can be interpreted correctly.

For example:

```text
User: Tell me an interesting fact from this paper.

Assistant: The paper discusses...

User: Another one.
```

The second query can be interpreted using the previous conversation instead of being treated as an unrelated query.

### 3. Dense Retrieval

Documents are converted into vector embeddings.

The query is also converted into an embedding, and semantically similar document chunks are retrieved from the vector store.

### 4. BM25 Retrieval

BM25 provides lexical/keyword-based retrieval.

This is useful when exact terminology, names, technical terms, or keywords are important.

### 5. Hybrid Retrieval

Dense retrieval and BM25 retrieval are combined using Reciprocal Rank Fusion (RRF).

This allows RAGForge to benefit from both:

- Semantic similarity
- Keyword matching

### 6. Reranking

The retrieved candidates are passed through a CrossEncoder reranker.

The reranker evaluates the relationship between the query and each candidate document and produces a ranking.

The highest-ranked documents are then passed to the generator.

> Note: CrossEncoder scores are model-specific relevance scores and are not probabilities. Negative values can be valid.

### 7. Generation

The final retrieved documents are provided as context to the LLM.

The generator is instructed to:

- Answer using the retrieved context
- Avoid inventing unsupported information
- Use conversation history when resolving follow-up questions
- Indicate when the available context is insufficient
- Identify the sources used

---

## 🛠️ Tech Stack

### Programming

- Python

### Interface

- Streamlit

### LLM

- OpenAI API

### Retrieval

- Qdrant
- Sentence Transformers
- BM25
- Reciprocal Rank Fusion

### Reranking

- CrossEncoder
- `cross-encoder/ms-marco-MiniLM-L-6-v2`

### Frameworks / Libraries

- LangChain
- Hugging Face
- PyTorch
- NumPy
- Pandas

---

## 📁 Project Structure

```text
RAGForge/
│
├── app/
│   └── streamlit_app.py
│
├── src/
│   │
│   ├── evaluation/
│   │
│   ├── generation/
│   │   ├── generator.py
│   │   └── test_generator.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── chunker.py
│   │   └── loaders.py
│   │
│   ├── retrieval/
│   │   ├── basic_rag.py
│   │   ├── hybrid.py
│   │   ├── reranker.py
│   │   ├── router.py
│   │   └── test_reranker.py
│   │
│   └── pipeline.py
│
├── data/
│   └── uploads/
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/randomaban-beep/RAGForge.git
```

Move into the project directory:

```bash
cd RAGForge
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root.

```env
OPENAI_API_KEY=your_openai_api_key
```

Never commit your `.env` file or API keys to GitHub.

---

## ▶️ Run Locally

Start the Streamlit application:

```bash
streamlit run app/streamlit_app.py
```

Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

Open the URL in your browser.

---

## 💬 Example Usage

Upload a document and ask:

```text
What is this document about?
```

Then ask a follow-up:

```text
Tell me an interesting fact.
```

Then:

```text
Another one.
```

Then:

```text
What about its limitations?
```

RAGForge uses the previous conversation to help resolve contextual follow-up questions.

---

## 🔎 Retrieval Pipeline

RAGForge uses a multi-stage retrieval architecture.

### Dense Retrieval

Dense retrieval represents documents and queries as vectors and retrieves semantically similar chunks.

### BM25

BM25 performs lexical retrieval based on term matching.

### Reciprocal Rank Fusion

RRF combines the rankings produced by dense retrieval and BM25.

This creates a hybrid retrieval stage that can handle both semantic and keyword-oriented queries.

### CrossEncoder Reranking

The hybrid retrieval candidates are passed to a CrossEncoder that scores query-document relevance.

The highest-ranked documents are passed to the generation stage.

---

## 🧭 Query Routing

RAGForge uses semantic routing to determine how a query should be handled.

```text
Query
 │
 ├── Greeting
 │      └── Direct response
 │
 ├── Casual
 │      └── LLM response
 │
 └── Document
        └── Retrieval pipeline
```

This prevents every query from unnecessarily going through the complete retrieval pipeline.

---

## 💬 Conversation Context

RAGForge maintains conversation history within the Streamlit session.

This allows the system to interpret follow-up queries such as:

```text
"Tell me about the architecture."

"What are its advantages?"

"Tell me another one."

"What about the limitations?"
```

without requiring the user to repeat the full context every time.

---

## 📚 Source Transparency

For document-related questions, RAGForge displays the retrieved source chunks used to generate the answer.

Each source can be expanded in the Streamlit interface to inspect its content.

This makes the retrieval process more transparent and helps users understand where the generated answer came from.

---

## 🔐 Security

API keys are stored through environment variables and should never be committed to source control.

The following files and directories are excluded from Git:

```text
.env
.venv/
data/uploads/
data/qdrant/
__pycache__/
```

---

## 🎯 Project Goal

RAGForge was built to explore and implement a practical, modular Retrieval-Augmented Generation architecture.

The project focuses on combining multiple retrieval techniques rather than relying on a single vector search approach.

The architecture separates:

```text
Routing
   ↓
Retrieval
   ↓
Reranking
   ↓
Generation
```

This makes individual components easier to test, replace, and extend.

---

## 🚧 Current Status

RAGForge currently supports:

- Document ingestion
- Document chunking
- Dense retrieval
- BM25 retrieval
- Hybrid retrieval
- CrossEncoder reranking
- Semantic query routing
- Conversation-aware queries
- LLM generation
- Source display
- Streamlit interface

The project is actively being developed.

---

## 📌 Future Improvements

Potential future improvements include:

- Persistent production vector storage
- Improved retrieval evaluation
- Query rewriting
- Retrieval caching
- Streaming generation
- Better document metadata handling
- Authentication
- Observability and tracing
- Production deployment
- More advanced retrieval strategies

---

## 👨‍💻 Author

**Md Aban Zeya**

GitHub:

https://github.com/randomaban-beep

---

## ⭐ Acknowledgements

This project uses open-source libraries and models from the Python, LangChain, Hugging Face, Sentence Transformers, and Qdrant ecosystems.
