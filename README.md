# 🎙️ Whisper Transcriber on macOS (Python 3.11 + Streamlit UI)

This project is a command-line tool and web-based interface (via Streamlit) that uses [OpenAI's Whisper](https://github.com/openai/whisper) to transcribe audio and video files into text.

---

## 📋 Features

- ✅ Transcribes `.mp4`, `.mov`, `.mkv`, `.avi`, `.mp3`, `.wav`, etc.
- ✅ Supports all Whisper model sizes (`tiny`, `base`, `small`, `medium`, `large`)
- ✅ Compatible with macOS (Apple Silicon or Intel)
- ✅ CLI and Web UI via Streamlit
- ✅ Optional use of `mps` (Apple GPU) for acceleration (use `cpu` if issues)

---

## 📦 Requirements

- macOS (Apple Silicon or Intel)
- Python 3.11
- Homebrew
- FFmpeg
- GitHub account (for uploading)

---

## 🛠️ Setup

### 1. Install Python 3.11 and Homebrew (if not installed)

```bash
brew install python@3.11
```

### 2. Create and activate a virtual environment

```bash
python3.11 -m venv whisper-env
source whisper-env/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install openai-whisper streamlit
```

### 4. Install FFmpeg

```bash
brew install ffmpeg
```

---

## 📄 `transcribe.py` Script

```python
#!/usr/bin/env python3
import whisper
import os
import subprocess
import tempfile
import argparse

def extract_audio(video_path):
    temp_audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    temp_audio.close()
    subprocess.run([
        "ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a",
        temp_audio.name, "-y"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return temp_audio.name

def transcribe_file(file_path, model_size="base", device="cpu"):
    model = whisper.load_model(model_size).to(device)
    if file_path.lower().endswith(('.mp4', '.mov', '.mkv', '.avi')):
        audio_path = extract_audio(file_path)
    else:
        audio_path = file_path

    result = model.transcribe(audio_path)
    if audio_path != file_path:
        os.remove(audio_path)

    return result["text"]
```

---

## 💻 Streamlit Web Interface

### `app.py`

```python
import streamlit as st
import whisper
import tempfile
import os
from transcribe import transcribe_file

st.title("🎙️ Whisper Speech-to-Text Transcriber")

uploaded_file = st.file_uploader("Upload an audio or video file", type=["mp3", "mp4", "wav", "mkv", "mov", "avi"])
model_size = st.selectbox("Choose Whisper model size", ["tiny", "base", "small", "medium", "large"], index=1)
device = st.selectbox("Choose device", ["cpu", "mps", "cuda"], index=0)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    if st.button("Transcribe"):
        with st.spinner("Transcribing..."):
            result = transcribe_file(temp_file_path, model_size=model_size, device=device)
            st.success("Transcription complete!")
            st.text_area("📝 Transcribed Text", value=result, height=300)
            os.remove(temp_file_path)
```

### Run the app

```bash
streamlit run app.py
```

---

## 🖼️ Screenshots (UI Preview)

> 📌 _Add screenshots of your Streamlit UI here for visual reference_

<img width="823" alt="image" src="https://github.com/user-attachments/assets/dfaa58ec-29eb-4bc7-abe8-0e10b3b7a917" />

<img width="785" alt="image" src="https://github.com/user-attachments/assets/9a6fe017-9a69-4f86-9efd-1e163740403e" />


---

## ▶️ CLI Usage

```bash
./transcribe.py testvideo.mp4 --model small --device cpu
```

You can also redirect output to a file:

```bash
./transcribe.py testvideo.mp4 > transcript.txt
```

---

## 🐛 Common Errors & Fixes

### ❌ Running `.py` as Shell Script

```bash
./transcribe.py: line 1: import: command not found
```

➡️ **Fix:** Ensure first line is `#!/usr/bin/env python3` or run with `python transcribe.py`.

---

### ❌ MPS Backend Sparse Tensor Error

```bash
NotImplementedError: Could not run 'aten::_sparse_coo_tensor_with_dims_and_tensors'
```

➡️ **Fix:** Use `--device cpu`

---

### ❌ No module named 'whisper'

➡️ **Fix:**

```bash
pip install openai-whisper
```

---

### ❌ FFmpeg Not Found

```bash
FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'
```

➡️ **Fix:**

```bash
brew install ffmpeg
```

---

## 🌐 Upload to GitHub

### 1. Initialize Git

```bash
git init
echo "whisper-env/" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
git add .
git commit -m "Initial commit"
```

### 2. Create GitHub Repo

Go to [https://github.com/new](https://github.com/new), name your repo, and copy the remote URL.

### 3. Push to GitHub

```bash
git remote add origin https://github.com/YOUR_USERNAME/whisper-transcriber.git
git branch -M main
git push -u origin main
```

---

## 🔗 Resources

- [Whisper GitHub](https://github.com/openai/whisper)
- [OpenAI Whisper Docs](https://platform.openai.com/docs/guides/speech-to-text)
- [Streamlit](https://streamlit.io)
- [PyTorch MPS Info](https://pytorch.org/docs/stable/notes/mps.html)

---

## 🙌 Credits

Created using OpenAI Whisper and Streamlit on macOS.
