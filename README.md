# 🧠 SkillScan: AI-Powered Resume Analyzer

Welcome to **SkillScan**, an intelligent resume analysis tool powered by **LLMs** and a **fine-tuned BERT model**! Upload your resume, select a job category, and get a personalized relevance score along with smart suggestions to enhance your resume – all with the power of **Artificial Intelligence**.


---

## 🚀 Features

- 🔍 **Category Prediction** – Fine-tuned BERT model classifies resumes into 24 job categories.
- 🤖 **LLM Feedback** – Real-time feedback on resume alignment using a local LLM via **Ollama**.
- 📄 **PDF Parsing** – Upload resumes in `.pdf` format and extract clean, readable text.
- 🧼 **Smart Preprocessing** – Advanced lemmatization, stopword removal, and noise cleaning.
- 💬 **Resume Assistant Chat** – Ask questions and get AI-generated career insights.

---

## 🛠️ Tech Stack

- `Python`
- `Streamlit` – for building the interactive web UI
- `HuggingFace Transformers` – fine-tuned `distilbert-base-uncased` for category prediction
- `Ollama` – for lightweight, locally-run LLM inference
- `PyPDF2`, `NLTK` – for text extraction and cleaning
- `Torch` – for model inference

---

## 🧠 Behind the Scenes

### Fine-Tuned BERT Model

Trained on a labeled resume dataset across 24 job roles:
- HR, Engineering, Finance, Designer, Teacher, IT, Healthcare, and more.
- Achieves high accuracy in predicting job relevance from resume content.

### LLM-Powered Feedback

Used a local **LLM** (via `Ollama`) to:
- Suggest missing keywords/skills
- Identify mismatches in role alignment
- Provide natural language resume improvement tips

---

## 📦 Installation
```bash
git clone https://github.com/your-username/skillscan-resume-analyzer.git
cd skillscan-resume-analyzer
pip install -r requirements.txt
ollama run assistant
streamlit run app.py
