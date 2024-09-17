"""Use Whisper to transcribe audio files to text."""

import subprocess
import ssl
import pathlib
import typer
import whisper

from splitter import split_text

ssl._create_default_https_context = ssl._create_unverified_context
app = typer.Typer()

model = whisper.load_model("base")


def convert_video_audio_file(
    video_file: pathlib.Path, 
    audio_folder: pathlib.Path,
    ) -> pathlib.Path:
    dest_audio_path = f"{audio_folder}/{video_file.with_suffix('.mp3').name}"
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            video_file.absolute(),
            "-vn",
            "-ab",
            "192k",
            "-ar",
            "44100",
            "-f",
            "mp3",
            dest_audio_path,
        ]
    )
    return pathlib.Path(dest_audio_path)


def transcribe_audio_file(audio_file: pathlib.Path) -> str:
    """Transcribe an audio file to text"""

    transcription = model.transcribe(audio=str(audio_file), verbose=False)
    return "\n".join(split_text(transcription["text"]))

@app.command()
def transcribe_videos_folder(
    videos_folder: pathlib.Path,
    audio_folder: pathlib.Path,
    transcription_folder: pathlib.Path,
):
    for path in pathlib.Path(videos_folder).iterdir():
        audio_file = convert_video_audio_file(path, audio_folder=audio_folder)
        transcription = transcribe_audio_file(audio_file)
        transcription_folder.joinpath(audio_file.with_suffix(".txt").name).write_text(transcription)


if __name__ == "__main__":
    app()