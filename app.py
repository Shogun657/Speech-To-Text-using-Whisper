import streamlit as st
import tempfile
import os
from transcribe import transcribe_audio

st.set_page_config(page_title="Whisper Transcription App", layout="centered")

st.title("🎤 Whisper Transcriber")
st.write("Upload an audio or video file and transcribe it using OpenAI's Whisper model.")

uploaded_file = st.file_uploader("Choose an audio or video file", type=["mp3", "wav", "m4a", "mp4", "mov", "mkv", "avi"])

model_size = st.selectbox("Model Size", options=["tiny", "base", "small", "medium", "large"], index=1)
device = st.selectbox("Device", options=["cpu", "mps"], index=0)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    if st.button("Transcribe"):
        with st.spinner("Transcribing..."):
            try:
                transcription = transcribe_audio(temp_path, model_size=model_size, device=device)
                st.success("Transcription completed!")
                st.text_area("Transcription Result", transcription, height=300)
            except Exception as e:
                st.error(f"Error: {str(e)}")

        os.unlink(temp_path)