import streamlit as st
import pickle
import re
import PyPDF2
from PIL import Image
import pytesseract

# OCR path (IMPORTANT for image)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# load model
model = pickle.load(open("resume_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# PDF extract
def extract_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# Image extract
def extract_image(file):
    image = Image.open(file)
    text = pytesseract.image_to_string(image)
    return text

st.title("AI Resume Screening System 🤖")

option = st.radio("Choose Input Type:", ["Text", "PDF", "Image"])

resume_text = ""

if option == "Text":
    resume_text = st.text_area("Paste Resume Here")

elif option == "PDF":
    uploaded_file = st.file_uploader("Upload Resume PDF", type=["pdf"])
    if uploaded_file:
        resume_text = extract_pdf(uploaded_file)

elif option == "Image":
    st.warning("image OCR temporarily disabled on cloud. Please use PDF or Text.")
    resume_text = ""

if st.button("Predict Category"):
    if resume_text.strip() == "":
        st.warning("Please provide resume input")
    else:
        cleaned = clean_text(resume_text)
        vector = tfidf.transform([cleaned])
        prediction = model.predict(vector)
        st.success(f"Predicted Category: {prediction[0]}")