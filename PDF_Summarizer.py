import streamlit as st
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from PyPDF2 import PdfReader

# Load the summarization pipeline (Hugging Face model)
st.subheader("Generating PDF Summary")

# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1

# Use a more general model loading approach for better error handling
try:
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=device)
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    summarizer = None

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

if pdf_file and summarizer:
    try:
        # Extract text from the uploaded PDF
        text = extract_text_from_pdf(pdf_file)
        
        if len(text) > 0:
            # Split the extracted text into smaller chunks if necessary
            def split_text(text, max_chunk_size=512):
                words = text.split()
                for i in range(0, len(words), max_chunk_size):
                    yield " ".join(words[i:i + max_chunk_size])

            # Split the PDF text into chunks
            chunks = list(split_text(text))
            summaries = []
            for chunk in chunks:
                result = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
                summaries.append(result[0]['summary_text'])
            
            # Combine summaries from all chunks
            summary = " ".join(summaries)

            # Display the summary
            st.subheader("Summary")
            st.write(summary)
        else:
            st.warning("No text could be extracted from the PDF.")
    
    except Exception as e:
        st.error(f"An error occurred during summarization: {str(e)}")
        
'''
import streamlit as st
import pdfplumber
from transformers import pipeline
import nltk
nltk.download("punkt_tab")

# Streamlit app
def main():
    st.title("Cutting-Edge PDF Summarizer")

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

            # Display extracted text (optional)
            st.subheader("Extracted Text")
            st.text_area("Text from PDF", text, height=300)

            # Load the summarization pipeline
            st.subheader("Processing the Summary")
            summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

            # Customize the summary length
            max_length = st.slider("Maximum length of the summary (words)", 50, 500, 150)
            min_length = st.slider("Minimum length of the summary (words)", 20, 100, 50)

            # Split and summarize text
            def split_text(text, max_chunk_size=1024):
                words = text.split()
                for i in range(0, len(words), max_chunk_size):
                    yield " ".join(words[i:i + max_chunk_size])

            chunks = list(split_text(text))
            summaries = [summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text'] for chunk in chunks]
            summary = " ".join(summaries)

            # Display the summary
            st.subheader("Summary")
            st.write(summary)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
'''
