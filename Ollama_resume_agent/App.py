import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import PyPDF2
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
load_dotenv()
os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNINGS"] = "true"
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
api_key = os.getenv("GROQ_API_KEY")
device = "cuda" if torch.cuda.is_available() else "cpu"
st.toast(f"Torch device set to: {device}")

embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device}
)

db = FAISS.load_local(
    "faiss_index/resume_data",
    embeddings=embedder,
    allow_dangerous_deserialization=True
)
template = """
You are a helpful assistant designed to analyze job category and resumes to identify skill gaps.
Keep the tone friendly, here are some relevant resume data {context}
Compare the skills required in the job category with the skills presented in the resume.
Focus on technical skills and experience. Provide a concise summary of potential skill gaps.
Also check the resume for any missing content like projects, education, experience etc and by doing all of the above list some 
helpful tips for the user and then display a message like "If you have any queries feel free to ask me!"
If no resume is provided, analyze the job category and list key skills required.
"""


def cleanse(concat_text):
    concat_text = concat_text.lower()
    concat_text = re.sub(r'[^a-zA-Z\s]', ' ', concat_text)
    concat_text = [word for word in concat_text.split() if word not in stopwords.words('english')]
    sentence = []
    lemmatizer = WordNetLemmatizer()
    for word in concat_text:
        sentence.append(lemmatizer.lemmatize(word, 'v'))
    return ' '.join(sentence)

def load_bert_model():
    model_name = "notshivain1/distilbert-base-uncased-resume-category-pred"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_bert = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model_bert
with st.spinner("Loading Bert model..."):
    tokenizer, model_bert = load_bert_model()

id2label = {
    0: 'HR',
    1: 'DESIGNER',
    2: 'INFORMATION-TECHNOLOGY',
    3: 'TEACHER',
    4: 'ADVOCATE',
    5: 'BUSINESS-DEVELOPMENT',
    6: 'HEALTHCARE',
    7: 'FITNESS',
    8: 'AGRICULTURE',
    9: 'BPO',
    10: 'SALES',
    11: 'CONSULTANT',
    12: 'DIGITAL-MEDIA',
    13: 'AUTOMOBILE',
    14: 'CHEF',
    15: 'FINANCE',
    16: 'APPAREL',
    17: 'ENGINEERING',
    18: 'ACCOUNTANT',
    19: 'CONSTRUCTION',
    20: 'PUBLIC-RELATIONS',
    21: 'BANKING',
    22: 'ARTS',
    23: 'AVIATION'
}

label2id = {label: idx for idx, label in id2label.items()}

def predict_category_score(text, selected_category):
    cleaned_text = cleanse(text)
    inputs = tokenizer(cleaned_text, truncation=True, padding="max_length", return_tensors="pt")
    with torch.no_grad():
        outputs = model_bert(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    category_id = label2id[selected_category]
    confidence_score = probabilities[0][category_id].item()
    return round(confidence_score * 100, 2)


prompt = PromptTemplate(
    input_variables=["question", "context"],
    template=template
)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

st.title('SkillScan Resume Analyzer')
job_category_prompt = st.selectbox(
    "Select a Job Category",
    tuple(label2id.keys()),
    index=None,
    placeholder="Select category...",
    key="job_category"
)

uploaded_resume = st.file_uploader("Upload your Resume in PDF format", type=["pdf"])

if st.button("Generate Resume Analysis"):
    if job_category_prompt:
        resume_text = ""
        if uploaded_resume is not None:
            try:
                pdf_reader = PyPDF2.PdfReader(uploaded_resume)
                for page in pdf_reader.pages:
                    resume_text += page.extract_text()
            except Exception as e:
                st.error(f"Error processing resume: {e}")
        
        if resume_text:
            with st.spinner("Analyzing resume..."):
                try:
                    score = predict_category_score(resume_text, job_category_prompt)
                    st.subheader("Predicted Category Confidence Score")
                    st.write(f"{job_category_prompt}: **{score:.2f}**")
                except Exception as e:
                    st.error(f"Error scoring resume: {e}")
        else:
            st.warning("Couldn't extract resume text.")

        with st.spinner("Generating LLM analysis..."):
            try:
                response = rag_chain.invoke({"query": f"Job Category: {job_category_prompt}\nResume Content: {resume_text}"})
                content = response['result']
                st.subheader("LLM Feedback")
                st.write(content)
            except Exception as e:
                st.error(f"Error from LLM: {e}")
    else:
        st.warning("Please select a job category.")

# Chat with LLM
st.divider()
st.header("Chat with Resume Assistant 💬")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

user_input = st.chat_input("Ask me anything")
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        response = rag_chain.invoke({"query": user_input})
        reply = response['result']
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)
    except Exception as e:
        st.error(f"Error talking to assistant: {e}")