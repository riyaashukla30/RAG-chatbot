# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions from PDF documents using context-aware responses.

## Tech
Python, LangChain, Gemini API, ChromaDB, Streamlit  

## How it works
- Loads PDFs from `data/`  
- Splits text into chunks  
- Creates embeddings and stores in ChromaDB  
- Retrieves relevant data on query  
- Generates answer using LLM  

## Setup & Run

Create `.env`:

Run:
python ingest.py
streamlit run app.py


Open: http://localhost:8501  

## Notes
- Do not upload `.chroma_db/`  
- Add your PDFs in `data/`  

## Author
Riya Shukla 
23BCE11293
https://github.com/riyaashukla30
