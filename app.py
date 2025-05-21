import os
import json
import subprocess
import whisper
import gradio as gr
from pydub import AudioSegment

# Initialize Whisper model
model = whisper.load_model("base")

# Download video using yt-dlp
def download_video(video_url, output_folder="/tmp/youtube_video"):
    os.makedirs(output_folder, exist_ok=True)
    video_path = os.path.join(output_folder, 'video_tempfile.mp4')
    try:
        subprocess.run([
            'yt-dlp',
            '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
            '-o', video_path,
            video_url
        ], check=True)
        return video_path
    except subprocess.CalledProcessError as e:
        print(f"Error downloading video: {e}")
        raise

# Extract audio from video
def extract_audio_from_video(video_path, audio_path="/tmp/youtube_audio/audio_tempfile.wav"):
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    try:
        audio = AudioSegment.from_file(video_path, format="mp4")
        audio.export(audio_path, format="wav")
        return audio_path
    except Exception as e:
        print(f"Error extracting audio: {e}")
        raise

# Transcribe audio using Whisper model
def transcribe_audio(audio_path):
    try:
        result = model.transcribe(audio_path)
        return result
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        raise

# Save transcription to file
def save_transcription_to_file(transcription_text, output_path="/tmp/transcription.txt"):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(transcription_text)
    return output_path

# Fallback chunking
def semantic_chunking_fallback(transcription_text, audio_length, chunk_duration=14.5):
    words = transcription_text.split()
    chunks = []
    chunk_id = 1
    total_chunks = int(audio_length // chunk_duration) + 1
    words_per_chunk = len(words) // total_chunks if total_chunks > 0 else len(words)

    for i in range(total_chunks):
        start_time = i * chunk_duration
        end_time = min((i + 1) * chunk_duration, audio_length)
        chunk_words = words[i * words_per_chunk : (i + 1) * words_per_chunk]
        chunks.append({
            "chunk_id": chunk_id,
            "chunk_length": round(end_time - start_time, 2),
            "text": " ".join(chunk_words),
            "start_time": round(start_time, 2),
            "end_time": round(end_time, 2),
        })
        chunk_id += 1

    return chunks

# Semantic chunking
def semantic_chunking(transcription_data, audio_path, chunk_duration_ms=14500):
    try:
        audio = AudioSegment.from_file(audio_path)
        audio_length_s = len(audio) / 1000
        chunks = []

        segments = transcription_data.get("segments", [])
        if not segments:
            print("No segments found in transcription, using fallback chunking.")
            transcription_text = transcription_data.get("text", "")
            return semantic_chunking_fallback(transcription_text, audio_length_s, chunk_duration=chunk_duration_ms/1000)

        chunk_id = 1
        current_chunk = {
            "chunk_id": chunk_id,
            "start_time": segments[0]['start'],
            "end_time": 0,
            "text": ""
        }

        for segment in segments:
            seg_start = segment['start']
            seg_end = segment['end']
            seg_text = segment['text'].strip()

            if seg_end - current_chunk['start_time'] > chunk_duration_ms / 1000:
                current_chunk['end_time'] = segment['start']
                chunks.append(current_chunk)
                chunk_id += 1
                current_chunk = {
                    "chunk_id": chunk_id,
                    "start_time": segment['start'],
                    "end_time": 0,
                    "text": seg_text
                }
            else:
                if current_chunk['text']:
                    current_chunk['text'] += " " + seg_text
                else:
                    current_chunk['text'] = seg_text

        current_chunk['end_time'] = segments[-1]['end']
        chunks.append(current_chunk)

        for c in chunks:
            c['chunk_length'] = round(c['end_time'] - c['start_time'], 2)

        return chunks

    except Exception as e:
        print(f"Error during semantic chunking: {e}")
        raise

# Main processing
def process_video(video_url):
    try:
        if "youtube.com" not in video_url and "youtu.be" not in video_url:
            raise ValueError("Invalid YouTube URL. Please provide a valid YouTube link.")

        print("Downloading video...")
        video_path = download_video(video_url)
        print(f"Video downloaded at {video_path}")

        print("Extracting audio from video...")
        audio_path = extract_audio_from_video(video_path)
        print(f"Audio extracted at {audio_path}")

        print("Transcribing audio...")
        transcription_data = transcribe_audio(audio_path)
        print("Transcription completed.")
        transcription_text = transcription_data.get('text', '')
        print(f"Full Transcription: {transcription_text}")

        print("Performing semantic chunking...")
        chunks = semantic_chunking(transcription_data, audio_path)
        print("Semantic chunking completed.")

        file_path = save_transcription_to_file(transcription_text)

        return transcription_text, chunks, file_path

    except Exception as e:
        print(f"Error processing video: {e}")
        return f"An error occurred: {str(e)}", None, None

# Gradio interface
def gradio_interface(video_url):
    transcription, chunks, file_path = process_video(video_url)
    return transcription, json.dumps(chunks, indent=4), file_path

# Gradio UI
iface = gr.Blocks()

with iface:
    gr.Markdown("# 🎥 YouTube Video Transcription and Semantic Chunking")
    gr.Markdown("### Upload a YouTube video URL to transcribe and chunk it into meaningful segments.")

    with gr.Row():
        video_url_input = gr.Textbox(label="YouTube Video URL", placeholder="Enter YouTube video URL here...")
        submit_btn = gr.Button("Transcribe", elem_id="transcribe-btn")

    with gr.Row():
        transcription_output = gr.Textbox(label="Transcription", lines=10, interactive=False)
        chunks_output = gr.JSON(label="Semantic Chunks")
        download_file_output = gr.File(label="Download Transcription")

    submit_btn.click(gradio_interface, inputs=video_url_input,
                     outputs=[transcription_output, chunks_output, download_file_output])
    download_file_output = gr.File(label="Download Transcription")


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

iface.launch()
