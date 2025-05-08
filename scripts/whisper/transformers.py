"""
This assumes you have a directory structure like this:

/root
    /<ISRC>
        /audio.mp3  # Original audio
        ... # Other processed versions
"""

import argparse
import glob
import json
import time
import os

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import librosa


def main():
    """
    Transcribe audio files using HuggingFace's Transformers Implementation of Whisper.
    """
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using HuggingFace Transformers Whisper"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--directory", type=str, required=True, help="Directory containing ISRC folders"
    )

    args = parser.parse_args()

    # Automatically use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model {args.model} on {device}...")
    
    processor = WhisperProcessor.from_pretrained(args.model)
    
    # Try to load with Flash Attention 2 if on CUDA
    if device == "cuda":
        try:
            model = WhisperForConditionalGeneration.from_pretrained(
                args.model,
                device_map=device,
                torch_dtype=torch.float16,
                attn_implementation="flash_attention_2"
            )
            print("Successfully enabled Flash Attention 2")
        except Exception as e:
            print(f"Flash Attention 2 not available, falling back to standard attention: {e}")
            model = WhisperForConditionalGeneration.from_pretrained(args.model).to(device)
    else:
        model = WhisperForConditionalGeneration.from_pretrained(args.model).to(device)

    # Find all ISRC folders in the directory
    isrc_folders = [
        f
        for f in os.listdir(args.directory)
        if os.path.isdir(os.path.join(args.directory, f))
    ]

    for isrc_folder in isrc_folders:
        isrc_path = os.path.join(args.directory, isrc_folder)

        # Find all versions of the audio file in the ISRC folder (either .wav or .mp3)
        audio_files = glob.glob(os.path.join(isrc_path, "*.wav"))
        audio_files.extend(glob.glob(os.path.join(isrc_path, "*.mp3")))

        if not audio_files:
            print(f"No audio files found in {isrc_path}")
            continue

        for audio_file in audio_files:
            print(f"Transcribing {audio_file}...")
            start_time = time.time()

            # Load and preprocess audio
            audio_array, sampling_rate = librosa.load(audio_file, sr=16000)
            input_features = processor(
                audio_array, sampling_rate=16000, return_tensors="pt"
            ).input_features.to(device)

            # Generate tokens
            with torch.no_grad():
                predicted_ids = model.generate(input_features)

            # Decode the tokens to text
            transcription = processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]

            end_time = time.time()
            print(f"Transcribed {audio_file} in {end_time - start_time:.2f}s")

            # Save results to a file in the ISRC folder
            results_file = os.path.join(isrc_path, "transcription_results.jsonl")
            with open(results_file, "a") as f:
                data = {
                    "file": audio_file,
                    "model": args.model,
                    "whisper_implementation": "transformers",
                    "transcription_time": end_time - start_time,
                    "transcription": transcription,
                }
                json.dump(data, f)
                f.write("\n")

            print(f"Transcribed {audio_file} and saved results to {results_file}")


if __name__ == "__main__":
    main()
