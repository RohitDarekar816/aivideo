import whisper
import os

def transcribe_audio_with_timestamps(audio_path, output_txt_path):
    # Load the Whisper model (choose "base", "small", "medium", or "large")
    model = whisper.load_model("base")

    # Transcribe the audio file
    result = model.transcribe(audio_path)

    # Open the output file for writing
    with open(output_txt_path, "w", encoding="utf-8") as f:

        for segment in result['segments']:
            start = segment['start']
            end = segment['end']
            text = segment['text'].strip()
            timestamp_line = f"[{format_timestamp(start)} - {format_timestamp(end)}] {text}"
            print(timestamp_line)  # Optional: print to console
            f.write(timestamp_line + "\n")

    print(f"\nTranscript saved to: {output_txt_path}")

def format_timestamp(seconds):
    # Format seconds to hh:mm:ss
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

# Example usage
if __name__ == "__main__":
    audio_file = "slowed_audio_ffmpeg.wav"  # Change this to your actual file
    output_file = "slowed_output.txt"

    if os.path.exists(audio_file):
        transcribe_audio_with_timestamps(audio_file, output_file)
    else:
        print(f"Audio file '{audio_file}' not found.")

