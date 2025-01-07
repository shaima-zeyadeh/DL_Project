import streamlit as st
from transformers import pipeline
import PyPDF2
import nltk
import torch
nltk.download("punkt")
import tensorflow as tf
from tensorflow import keras
from PyPDF2 import PdfReader

# Load the summarization pipeline
st.subheader("Generating PDF Summary")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=0 if torch.cuda.is_available() else -1)

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text()
    return text

# Streamlit file uploader for PDF
pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])

if pdf_file:
    try:
        # Extract text from the uploaded PDF
        text = extract_text_from_pdf(pdf_file)
        
        if len(text) > 0:
            # Split the extracted text into smaller chunks if necessary
            def split_text(text, max_chunk_size=512):
                words = text.split()
                for i in range(0, len(words), max_chunk_size):
                    yield " ".join(words[i:i + max_chunk_size])

            chunks = list(split_text(text))
            summaries = []
            for chunk in chunks:
                result = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
                summaries.append(result[0]['summary_text'])
            
            summary = " ".join(summaries)

            # Display the summary
            st.subheader("Summary")
            st.write(summary)
        else:
            st.warning("No text could be extracted from the PDF.")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

'''
# Streamlit app
def main():
    st.title("Optimized PDF Summarizer")

    # File upload
    uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

    if uploaded_file is not None:
        try:
            # Read the PDF file
            with pdfplumber.open(uploaded_file) as pdf:
                text = ''
                for page in pdf.pages:
                    text += page.extract_text()

            if not text.strip():
                st.error("No text found in the uploaded PDF.")
                return

            # Display extracted text
            st.subheader("Extracted Text")
            st.text_area("Text from PDF", text[:5000], height=300)  # Show up to 5000 chars

            # Load the summarization pipeline
            st.subheader("Generating Summary")
            summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=0 if torch.cuda.is_available() else -1)

            # Split and summarize text
            def split_text(text, max_chunk_size=512):
                words = text.split()
                for i in range(0, len(words), max_chunk_size):
                    yield " ".join(words[i:i + max_chunk_size])

            chunks = list(split_text(text))
            summaries = []
            for chunk in chunks:
                summaries.append(summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]['summary_text'])
            
            summary = " ".join(summaries)

            # Display the summary
            st.subheader("Summary")
            st.write(summary)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
'''
