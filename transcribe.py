#!/usr/bin/env python3
import whisper
import sys
import os
import subprocess
import tempfile
import argparse

def extract_audio(video_path):
    temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        temp_audio,
        "-y"
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return temp_audio

def transcribe_file(file_path, model_size="base", device="cpu"):
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return

    if file_path.lower().endswith(('.mp4', '.mov', '.mkv', '.avi')):
        print("Extracting audio from video...")
        file_path = extract_audio(file_path)

    print(f"Loading model '{model_size}' on device '{device}'...")
    model = whisper.load_model(model_size).to(device)

    print("Transcribing...")
    result = model.transcribe(file_path)
    print("\n--- Transcription ---\n")
    print(result["text"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio or video using OpenAI Whisper.")
    parser.add_argument("file", help="Path to audio or video file")
    parser.add_argument("--model", default="base", help="Model size: tiny, base, small, medium, large")
    parser.add_argument("--device", default="cpu", help="Device to run model on: cpu, cuda, mps")

    args = parser.parse_args()
    transcribe_file(args.file, model_size=args.model, device=args.device)