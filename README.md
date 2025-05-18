# 🎙️ Whisper Transcriber on macOS (Python 3.11)

This project is a command-line tool that uses [OpenAI's Whisper](https://github.com/openai/whisper) to transcribe audio and video files into text. It includes setup instructions, usage examples, and fixes for common macOS-related errors.

---

## 📋 Features

- ✅ Transcribes `.mp4`, `.mov`, `.mkv`, `.avi`, `.mp3`, `.wav`, etc.
- ✅ Supports all Whisper model sizes (`tiny`, `base`, `small`, `medium`, `large`)
- ✅ Compatible with macOS (Apple Silicon or Intel)
- ✅ Optionally uses `mps` (Apple GPU), but `cpu` is recommended for stability

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
pip install openai-whisper
```

### 4. Install FFmpeg

```bash
brew install ffmpeg
```

---

## 📄 `transcribe.py` Script

Save the following to `transcribe.py`:

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
    print("\n📝 Transcription:\n")
    print(result["text"])

    if audio_path != file_path:
        os.remove(audio_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to video or audio file")
    parser.add_argument("--model", default="base", help="Model size (tiny, base, small, medium, large)")
    parser.add_argument("--device", default="cpu", help="Device to run on (cpu, mps, cuda)")
    args = parser.parse_args()

    transcribe_file(args.file, model_size=args.model, device=args.device)
```

Make it executable:

```bash
chmod +x transcribe.py
```

---

## ▶️ Running the Script

Run with:

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

**Error:**
```bash
./transcribe.py: line 1: import: command not found
```

**Fix:** Ensure the first line is:
```python
#!/usr/bin/env python3
```
Or run it with Python:

```bash
python transcribe.py ...
```

---

### ❌ MPS Backend Sparse Tensor Error

**Error:**
```
NotImplementedError: Could not run 'aten::_sparse_coo_tensor_with_dims_and_tensors' with arguments from the 'SparseMPS' backend
```

**Fix:** Use the CPU instead of MPS:

```bash
--device cpu
```

---

### ❌ No module named 'whisper'

**Fix:**

```bash
pip install openai-whisper
```

Make sure you're inside your virtual environment.

---

### ❌ FFmpeg Not Found

**Error:**

```bash
FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'
```

**Fix:**

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
- [PyTorch MPS Info](https://pytorch.org/docs/stable/notes/mps.html)
- [Youtube](https://www.youtube.com/watch?v=iQqCAQ8hG_Y)

---

## 🙌 Credits

Created using OpenAI Whisper and PyTorch on macOS.

