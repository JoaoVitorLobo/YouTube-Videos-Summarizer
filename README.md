# YouTube-Videos-Summarizer
# YouTube Video Summarizer Powered by AI. Whisper-1 and GPT-4o-Mini implementation through OpenAI API

from openai import OpenAI
from yt_dlp import YoutubeDL
import sys

# OpenAI API Key
key = "ENTER YOUR OPENAI API KEY HERE"

def download_wav(url: str, audio_file: str = "audio.wav") -> None:
    # Set URL download settings
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": audio_file,
        "overwrites": True # Overwrite audio_file if existing
    }
    
    # Download
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def speech_to_text(client: OpenAI, audio_file: str = "audio.wav") -> str:
    # Open audio_file
    with open(audio_file, "rb") as audio:
        # Transcribe using OpenAI's Whisper-1 model
        transcription = client.audio.transcriptions.create(model = "whisper-1", file = audio, response_format = "text")

    return transcription


def summarizer(transcription: str, client: OpenAI) -> str:
    # Summarize the transcription through chat with OpenAI's GPT-4o-mini model
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", 
             # ChatGPT role
             "content": "You are a YouTube Videos Summarizer who will describe pragmatically a summary of the video's audio in its primary language."
             },

            {"role": "user",
             # Command to ChatGPT
             "content": f"Summarize this video transcript: {transcription}. Keep the summary concise, focusing only on key points and main takeaways."
            }   
            ])
    
    return completion


def main():
    # Download audio from URL
    download_wav(sys.argv[1])

    # Set OpenAI API client
    client = OpenAI(api_key= key)

    # Transcribe audio to text
    transcription = speech_to_text(client)

    # Summarize the transcription
    summary = summarizer(transcription, client)

    # Summary
    print(f"→ {summary.choices[0].message.content}")

main()
