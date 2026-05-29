# Research Paper Brain

### Offline Research Paper Q&A using RAG on Edge Devices

Research Paper Brain is an offline AI-powered Research Paper Question-Answering system built using Retrieval-Augmented Generation (RAG). The system allows users to upload research paper PDFs and ask questions in natural language. It retrieves relevant sections from the document and generates context-aware answers with citations — completely offline.

Built for edge AI deployment using NVIDIA Jetson + Ollama.

---

## 🚀 Features

* 📄 Upload and process research paper PDFs
* 💬 Ask questions in natural language
* 🔍 Retrieval-Augmented Generation (RAG) pipeline
* 📌 Citation-based responses with page references
* 🧠 Beginner-friendly explanation of complex concepts
* 📚 Section-wise summaries (Abstract, Methodology, Results, Conclusion)
* 🔒 Fully offline execution (zero external API calls)
* ⚡ Lightweight deployment on NVIDIA Jetson devices

---

## 🧠 System Architecture

```text
PDF Upload
     ↓
PDF Text Extraction
     ↓
Chunking
     ↓
Embeddings Generation
     ↓
ChromaDB Vector Storage
     ↓
Semantic Retrieval
     ↓
LLM via Ollama
     ↓
Answer + Citation
```

---

## 🛠️ Tech Stack

| Component            | Technology                      |
| -------------------- | ------------------------------- |
| Programming Language | Python                          |
| Framework            | LangChain                       |
| LLM                  | Qwen2.5:1.5B / DeepSeek-R1:1.5B |
| Vector Database      | ChromaDB                        |
| PDF Processing       | PyMuPDF                         |
| Frontend             | Streamlit                       |
| Local LLM Runtime    | Ollama                          |
| Deployment           | NVIDIA Jetson                   |

---

## 📂 Project Structure

```bash
research-paper-rag/
│
├── app.py
├── requirements.txt
│
├── data/
│   └── sample_papers/
│
├── rag/
│   ├── pdf_parser.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── retriever.py
│   └── generator.py
│
├── ui/
│   └── streamlit_ui.py
│
├── docs/
│   ├── architecture.png
│   └── screenshots/
│
└── README.md
```

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-username/research-paper-rag.git
cd research-paper-rag
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🤖 Install Ollama

Download and install Ollama:

https://ollama.com

Pull required models:

```bash
ollama pull qwen2.5:1.5b
```




---

## ▶️ Run the Application

```bash
streamlit run app.py
```

Open browser:

```text
http://localhost:8501
```

---

## 🧩 How It Works

### 1. PDF Parsing

Research papers are uploaded and converted into raw text using PyMuPDF.

### 2. Chunking

Large text is split into smaller semantic chunks for efficient retrieval.

### 3. Embeddings

Text chunks are converted into vector embeddings using embedding models.

### 4. Vector Database

Embeddings are stored in ChromaDB for semantic similarity search.

### 5. Retrieval

Relevant chunks are retrieved based on the user query.

### 6. Answer Generation

Retrieved context is passed to a local LLM through Ollama to generate answers with citations.

---

## 📸 Screenshots

Add screenshots inside:

```text
docs/screenshots/
```

Example:

* Upload Interface
* Chat Interface
* Citation-based Answers
* System Architecture

---

## 🎯 Expected Outcome

This project helps students and researchers:

* understand research papers faster
* get accurate citation-based answers
* simplify complex technical concepts
* use AI completely offline without cloud dependency

---

## 🔮 Future Improvements

* Multi-PDF querying
* Research paper summarization
* Voice-based interaction
* Hybrid search (keyword + semantic)
* GPU optimization
* Web deployment
* Chat history and memory

---

## 👨‍💻 Team

### Neural Ninjas

* Shreyash Omar 
* Aniket Sahu 
* Priyaansh Pandey
* Devansh Shukla

---

## 📜 License

This project is licensed under the MIT License.

---

## ⭐ Acknowledgements

* Ollama
* LangChain
* ChromaDB
* Streamlit
* Hugging Face
* NVIDIA Jetson
