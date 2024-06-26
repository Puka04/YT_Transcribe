import os
import subprocess
import sys
import json
from pytube import YouTube
from pydub import AudioSegment
import whisper
import gradio as gr
import soundfile as sf

# Initialize Whisper model
model = whisper.load_model("base")

# Function to download video from YouTube
def download_video(video_url, output_folder="/tmp/youtube_video"):
    os.makedirs(output_folder, exist_ok=True)
    try:
        yt = YouTube(video_url)

        # Check if the video has a progressive stream (i.e., directly downloadable)
        if yt.streams.filter(progressive=True, file_extension='mp4').first():
            video_stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
        else:
            # If no progressive stream found, attempt to get the highest quality stream
            video_stream = yt.streams.get_highest_resolution()

        if video_stream is None:
            raise ValueError("No suitable video stream found.")

        video_path = video_stream.download(output_path=output_folder, filename='video_tempfile.mp4')
        return video_path
    except Exception as e:
        print(f"Error downloading video: {str(e)}")
        raise

# Function to extract audio from video
def extract_audio_from_video(video_path, audio_path="/tmp/youtube_audio/audio_tempfile.wav"):
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    try:
        audio = AudioSegment.from_file(video_path)
        audio.export(audio_path, format="wav")
        return audio_path
    except Exception as e:
        print(f"Error extracting audio: {str(e)}")
        raise

# Function to transcribe audio using Whisper model
def transcribe_audio(audio_path):
    try:
        result = model.transcribe(audio_path)
        transcription = result['text']
        return transcription
    except Exception as e:
        print(f"Error transcribing audio: {str(e)}")
        raise

# Function to split audio and transcript into semantic chunks
def semantic_chunking(transcription, audio_path, chunk_size=14500):
    try:
        audio = AudioSegment.from_file(audio_path)
        chunks = []
        chunk_id = 1
        words = transcription.split()
        num_words = len(words)
        chunk_length = num_words // (len(audio) // chunk_size)

        for start in range(0, len(audio), chunk_size):
            end = min(start + chunk_size, len(audio))
            text_chunk = ' '.join(words[(chunk_id-1)*chunk_length:chunk_id*chunk_length])
            chunks.append({
                "chunk_id": chunk_id,
                "chunk_length": (end - start) / 1000,
                "text": text_chunk,
                "start_time": start / 1000,
                "end_time": end / 1000,
            })
            chunk_id += 1

        return chunks
    except Exception as e:
        print(f"Error during semantic chunking: {str(e)}")
        raise

# Function to process YouTube video and output transcript and chunks
def process_video(video_url):
    try:
        # Step 1: Download video
        print("Downloading video...")
        video_path = download_video(video_url)
        print(f"Video downloaded at {video_path}")

        # Step 2: Extract audio
        print("Extracting audio from video...")
        audio_path = extract_audio_from_video(video_path)
        print(f"Audio extracted at {audio_path}")

        # Step 3: Transcribe audio
        print("Transcribing audio...")
        transcription = transcribe_audio(audio_path)
        print("Transcription completed.")
        print(f"Full Transcription: {transcription}")

        # Step 4: Semantic chunking
        print("Performing semantic chunking...")
        chunks = semantic_chunking(transcription, audio_path)
        print("Semantic chunking completed.")

        return transcription, chunks, video_path, audio_path

    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return str(e), None, None, None

# Function to generate downloadable text file
def save_transcript(transcription):
    transcript_path = "/tmp/youtube_audio/transcript.txt"
    with open(transcript_path, "w") as f:
        f.write(transcription)
    return transcript_path

# Define a new Gradio interface with an enhanced layout
def gradio_interface(video_url):
    transcription, chunks, video_path, audio_path = process_video(video_url)
    transcript_path = save_transcript(transcription)
    return transcription, json.dumps(chunks, indent=4), video_path, audio_path, transcript_path

# Create a new Gradio interface
iface = gr.Blocks()

with iface:
    gr.Markdown("# 🎥 YouTube Video Transcription and Semantic Chunking")
    gr.Markdown("### Upload a YouTube video URL to transcribe and chunk it into meaningful segments.")

    with gr.Row():
        with gr.Column():
            video_url_input = gr.Textbox(label="YouTube Video URL", placeholder="Enter YouTube video URL here...")
            submit_btn = gr.Button("Transcribe", elem_id="transcribe-btn")

    with gr.Row():
        transcription_output = gr.Textbox(label="Transcription", lines=10, interactive=False)
        chunks_output = gr.JSON(label="Semantic Chunks")

    with gr.Row():
        download_video_btn = gr.File(label="Download Video")
        download_audio_btn = gr.File(label="Download Audio")
        download_transcript_btn = gr.File(label="Download Transcript")

    submit_btn.click(gradio_interface, inputs=video_url_input, outputs=[transcription_output, chunks_output, download_video_btn, download_audio_btn, download_transcript_btn])

# Add custom CSS for the transcribe button
iface.css = """
#transcribe-btn {
    background-color: #4CAF50;
    color: white;
    border: none;
    padding: 10px 24px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    transition-duration: 0.4s;
    cursor: pointer;
}

#transcribe-btn:hover {
    background-color: white;
    color: black;
    border: 2px solid #4CAF50;
}
"""

# Launch the Gradio app
iface.launch()
