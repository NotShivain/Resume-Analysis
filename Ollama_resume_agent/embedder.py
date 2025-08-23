import pandas as pd
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import torch 
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print('Device set to ', device)
df = pd.read_csv(r'C:\Users\shiva\Resume-Analysis\Ollama_resume_agent\resume_cleaned.csv', engine='python', encoding='latin1', on_bad_lines='warn')
df['text'] = df['Category'] + ': ' + df['Cleaned_text'].fillna('')
documents = [
    Document(page_content=row['text'], metadata={"category": row['Category'], "skills": row.get('Skills', 'N/A')})
    for _, row in df.iterrows()
]
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device})

db = FAISS.from_documents(documents, embedding=embedder)
db.save_local("faiss_index/resume_data")
print("FAISS index saved to faiss_index/resume_data")
