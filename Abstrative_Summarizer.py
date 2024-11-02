import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer
from newspaper import Article
import pdfplumber
from docx import Document
import re
from pydub import AudioSegment
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np


# Function to check if the input is a valid URL
def check_url(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return bool(url_pattern.match(text))


# Function to extract text from a web article
def extract_text_from_article(url):
    article = Article(url, language='en')
    article.download()
    article.parse()
    return article.text


# Function to extract text from a PDF
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ''.join([page.extract_text() or '' for page in pdf.pages])
    return text


# Function to extract text from a DOCX file
def extract_text_from_docx(file):
    doc = Document(file)
    return '\n'.join([paragraph.text for paragraph in doc.paragraphs])


# Function to extract text from an audio file
def Audio_to_Text(uploaded_file):
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    
    audio = AudioSegment.from_file(uploaded_file)

    audio_samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    sample_rate = audio.frame_rate  

    if audio.channels > 1:  
        audio = audio.set_channels(1)
        audio_samples = np.array(audio.get_array_of_samples(), dtype=np.float32)

    audio_samples = audio_samples / np.max(np.abs(audio_samples))

    input_features = processor(audio_samples, sampling_rate=sample_rate, return_tensors="pt", return_attention_mask=True)

    predicted_ids = model.generate(
        input_features['input_features'],
        attention_mask=input_features['attention_mask'],
        language='en'  
    )

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
    return transcription


# Function to split text into chunks for processing
def split_text_into_chunks(text, chunk_size=1024):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield ' '.join(words[i:i + chunk_size])


# Function to summarize text using BART model
def summarize_text(text, max_length=200, min_length=150, num_beams=4):
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)

    if inputs["input_ids"].size(1) == 0:
        return "Error: Input is too short or not properly tokenized."
    
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        num_beams=num_beams,
        early_stopping=True
    )

    if summary_ids is None or summary_ids.size(0) == 0:
        return "Error: Failed to generate summary. Output was empty."

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return summary


# UI functionality
st.markdown(
    """
    <style>
    body {
        background-image: url('https://wallpapers.com/images/hd/black-shiny-background-d4tg43kuk0yw8s9m.jpg');  
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: white;  
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("Summarizer using BART")
st.write("Summarize articles, documents, audio, or text.")

input_type = st.radio("Choose input type:", ("URL", "Text", "File", "Audio"))


if input_type == "URL":
    input_data = st.text_input("Enter the URL:")
elif input_type == "Text":
    input_data = st.text_area("Enter the text:")
elif input_type == 'Audio':
    input_data = st.file_uploader("Upload the audio file:", type=["mp3", "wav", "m4a"])
else:
    input_data = st.file_uploader("Upload a file (PDF or DOCX):", type=["pdf", "docx"])


if st.button("Summarize"):
    with st.spinner("Generating Summaries in a bit..."):
        article_text = ""

        if input_type == "URL" and input_data:
            if check_url(input_data):
                article_text = extract_text_from_article(input_data)
            else:
                st.error("Please enter a valid URL.")

        elif input_type == "Text" and input_data:
            article_text = input_data

        elif input_type == "File" and input_data is not None:
            if input_data.type == "application/pdf":
                article_text = extract_text_from_pdf(input_data)
            elif input_data.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                article_text = extract_text_from_docx(input_data)
            else:
                st.error("Unsupported file type. Please upload a PDF or DOCX file.")

        elif input_type == "Audio" and input_data:
            article_text = Audio_to_Text(input_data)

        else:
            st.error("Please enter a valid URL, text, or upload a file to summarize.")

        if article_text:
            chunks = list(split_text_into_chunks(article_text))
            summaries = [summarize_text(chunk) for chunk in chunks]
            generated_summary = ' '.join(summaries)
        
            st.subheader("Summary")
            st.write(generated_summary)

