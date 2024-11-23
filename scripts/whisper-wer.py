"""
This assumes you have a directory structure like this:

/root
    /<ISRC>
        /vocals.wav  # Demucs processed audio
        /audio.mp3  # Original audio
        /lyrics.json

The lyrics.json file should have the following structure:
{
    "unsynced": {
        "data": "lyrics"
    }
}
"""

import argparse
import json
import os

import whisperx
from whisperx.asr import FasterWhisperPipeline
from whisperx.types import TranscriptionResult


def transcribe_audio(
    model: FasterWhisperPipeline, audio_path: str
) -> TranscriptionResult:
    """
    Transcribes an audio file using WhisperX.

    :param model: The WhisperX model to use for transcription.
    :param audio_path: The path to the audio file to transcribe.
    :return: The transcription result.
    """
    audio = whisperx.load_audio(audio_path)
    return model.transcribe(audio, batch_size=16)


def process_files(model: FasterWhisperPipeline, args: argparse.Namespace, vad_enabled: bool):
    """
    Processes all audio files in the given directory and saves transcriptions.

    :param model: The WhisperX model to use for transcription.
    :param args: The parsed arguments from argparse.
    :param vad_enabled: Whether VAD is currently enabled
    """
    file_name = "vocals.wav" if args.use_demucs else "audio.mp3"

    for root, dirs, files in os.walk(args.directory):
        if root == args.directory:
            continue

        isrc = os.path.basename(root)

        if file_name in files:
            try:
                # Transcribe audio
                audio_path = os.path.join(root, file_name)
                result = transcribe_audio(model, audio_path)

                model_fs = args.model.split("/")[-1]
                source_type = "demucs" if args.use_demucs else "orig"
                vad_status = "vad" if vad_enabled else "novad"
                result_filename = f"{model_fs}_{source_type}_{vad_status}_results.json"

                # Convert result to a serializable dictionary
                serializable_result = {
                    "segments": [
                        {
                            "start": segment["start"],
                            "end": segment["end"],
                            "text": segment["text"],
                        }
                        for segment in result["segments"]
                    ],
                    "language": result["language"],
                }

                # Save hypothesis and language in a single JSON file
                with open(os.path.join(root, result_filename), "w") as f:
                    json.dump(serializable_result, f, indent=2)

                print(f"Processed {isrc}")

            except Exception as e:
                print(f"Error processing {isrc}: {str(e)}")
                continue


def disable_vad(model: FasterWhisperPipeline):
    """
    Disables VAD by setting minimal onset/offset thresholds.

    :param model: The WhisperX model to modify
    """
    model._vad_params["vad_onset"] = 0.00001
    model._vad_params["vad_offset"] = 0.00001


def process_both_modes(model: FasterWhisperPipeline, args: argparse.Namespace):
    """
    Processes all audio files with and without VAD.

    :param model: The WhisperX model to use for transcription.
    :param args: The parsed arguments from argparse.
    """
    print("\nProcessing with VAD enabled...")
    print("-----------------------------------------------------")
    process_files(model, args, vad_enabled=True)

    print("\nProcessing with VAD disabled...")
    print("-----------------------------------------------------")
    disable_vad(model)
    process_files(model, args, vad_enabled=False)


def main():
    """
    Process audio files using WhisperX and save their transcriptions.
    """
    parser = argparse.ArgumentParser(description="Transcribe audio using WhisperX")
    parser.add_argument(
        "--directory", type=str, required=True, help="Directory containing audio files"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Whisper model name (e.g., 'large-v2')"
    )
    parser.add_argument(
        "--use_demucs",
        action="store_true",
        help="Use demucs processed files instead of original MP3s",
    )

    args = parser.parse_args()

    # Initialize the WhisperX model
    print(f"Initializing WhisperX model: {args.model}")
    model = whisperx.load_model(args.model, device="cuda", compute_type="float16")

    # Process files in both VAD modes
    print(f"\nProcessing {'demucs' if args.use_demucs else 'original'} files...")
    process_both_modes(model, args)


if __name__ == "__main__":
    main()
