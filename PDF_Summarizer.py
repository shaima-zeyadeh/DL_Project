import streamlit as st
import pdfplumber
from transformers import pipeline
import nltk
nltk.download("punkt")

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
