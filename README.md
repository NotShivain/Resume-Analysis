# SkillScan: AI-Powered Resume Analyzer

Welcome to **SkillScan**, an intelligent resume analysis tool powered by **LLMs** and a **fine-tuned BERT model**! Upload your resume, select a job category, and get a personalized relevance score along with smart suggestions to enhance your resume – all with the power of **Artificial Intelligence**.


## Tech Stack

- `Python`
- `Streamlit` – for building the interactive web UI
- `HuggingFace Transformers` – fine-tuned `distilbert-base-uncased` for category prediction
- `Ollama` – for lightweight, locally-run LLM inference
- `PyPDF2`, `NLTK` – for text extraction and cleaning
- `Torch` – for model inference

---

## Behind the Scenes

### Fine-Tuned BERT Model

Trained on a labeled resume dataset across 24 job roles:
- HR, Engineering, Finance, Designer, Teacher, IT, Healthcare, and more.
- Achieves high accuracy in predicting job relevance from resume content.

### LLM-Powered Feedback

Used a local **LLM** (via `Groq`) to:
- Suggest missing keywords/skills
- Identify mismatches in role alignment
- Provide natural language resume improvement tips

---

## Installation And How to run
- First get your api keys from groq and langchain (imp)
- git clone https://github.com/NotShivain/Resume-Analysis
- start a virtual environment in your project
- run pip install -r requirements.txt
- add your api keys in .env folder
- download the faiss index from the url: https://drive.google.com/drive/folders/1s8XjJoOKWoKe5o21421ufjgfEUAppLLa?usp=sharing
- streamlit run App.py
 
# Please make sure all the file paths are correctly set before running the project
