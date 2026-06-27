# RAG Pipeline from Scratch

A document Q&A system built from scratch using Python, Groq, and sentence-transformers.
Ask questions in natural language and get answers retrieved from your own documents.

## How it works
- Splits document into overlapping paragraph chunks
- Embeds all chunks using sentence-transformers (all-MiniLM-L6-v2)
- When user asks a question, embeds the query and finds most similar chunks using cosine similarity
- Sends only the relevant chunks to Llama 3.3 70b via Groq API
- Filters out irrelevant results using a similarity threshold

## Live Demo

Built a RAG pipeline with LangChain + FAISS, exposed via FastAPI, containerized with Docker, and deployed on Railway.

**Live API:** [https://groqsetup-production.up.railway.app/docs](https://groqsetup-production.up.railway.app/docs)

## Setup
```bash
pip install groq sentence-transformers torch python-dotenv
```

Create a `.env` file with your Groq API key:
GROQ_API_KEY=your_key_here

## Run
```bash
python chat_loop.py
```

## Example queries
- "how does the federal reserve control inflation?"
- "how do grapplers train?"
- "what is retrieval augmented generation?"