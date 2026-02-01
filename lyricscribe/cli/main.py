import argparse
import contextlib
import os
import platform
import sys

import torch

from .separators.demucs import DemucsSeparator
from .separators.spleeter import SpleeterSeparator
from .whisper.fasterwhisper import (
    FasterWhisperChunkedTranscriber,
    FasterWhisperSequentialTranscriber,
)
from .whisper.openai import OpenAITranscriber
from .whisper.transformers import (
    TransformersChunkedTranscriber,
    TransformersSequentialTranscriber,
)
from .whisper.whisperx import WhisperXTranscriber


def log_system_resources():
    """
    Logs general system resource information.
    """
    print("LyricScribe System Information:")
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  PyTorch: CUDA available (GPU: {gpu_name})")
        except Exception as e:
            print(f"  PyTorch: CUDA available, but failed to get device name: {e}")
    else:
        print("  PyTorch: CUDA not available or not detected by PyTorch.")

    cpu_info = "Unknown"
    cpu_cores = "Unknown"
    cpu_freq = "Unknown"
    ram_info = "Unknown"
    with contextlib.suppress(Exception):
        if platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        cpu_info = line.split(":")[1].strip()
                        break

            cpu_cores = os.cpu_count()
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "cpu MHz" in line:
                        cpu_freq = f"{float(line.split(':')[1].strip())/1000:.2f} GHz"
                        break

            with open("/proc/meminfo", "r") as f:
                total_ram = None
                for line in f:
                    if "MemTotal" in line:
                        total_ram = int(line.split()[1])  # Value is in kB
                        ram_info = f"{total_ram // 1024 // 1024} GB"
                        break
    print(f"  CPU Model: {cpu_info}")
    print(f"  CPU Cores: {cpu_cores}")
    print(f"  CPU Frequency: {cpu_freq}")
    print(f"  Total RAM: {ram_info}")
    print("-----------------------------------------------------")


# Handler for the whisper command
def handle_whisper_command(args: argparse.Namespace):
    directory = None
    if args.directory:
        directory = args.directory
    elif args.file:
        directory = os.path.dirname(args.file)

    transcriber = None
    if args.backend == "openai":
        transcriber = OpenAITranscriber(model_name=args.model, directory=directory)
    elif args.backend == "hf-sequential":
        transcriber = TransformersSequentialTranscriber(
            model_name=args.model, directory=directory
        )
    elif args.backend == "hf-chunked":
        transcriber = TransformersChunkedTranscriber(
            model_name=args.model,
            directory=directory,
            batch_size=args.batch_size,
        )
    elif args.backend == "faster-whisper-sequential":
        transcriber = FasterWhisperSequentialTranscriber(
            model_name=args.model,
            directory=directory,
            beam_size=args.beam_size,
            vad_filter=args.vad,
        )
    elif args.backend == "faster-whisper-chunked":
        transcriber = FasterWhisperChunkedTranscriber(
            model_name=args.model,
            directory=directory,
            beam_size=args.beam_size,
            vad_filter=args.vad,
            batch_size=args.batch_size,
        )
    elif args.backend == "whisperx":
        transcriber = WhisperXTranscriber(
            model_name=args.model,
            directory=directory,
            batch_size=args.batch_size,
            beam_size=args.beam_size,
        )

    if transcriber:
        if args.file:
            transcriber.process_single_file(args.file)
        elif args.directory:
            transcriber.process_directory()
    else:
        print(
            f"Error: Could not initialize transcriber for backend {args.backend}",
            file=sys.stderr,
        )


# Handler for the demucs separator command
def handle_demucs_command(args: argparse.Namespace):
    separator = DemucsSeparator(
        model_name=args.model,
        output_filename_suffix=args.output,
        directory=args.directory,
    )
    separator.process_directory()


# Handler for the spleeter separator command
def handle_spleeter_command(args: argparse.Namespace):
    separator = SpleeterSeparator(
        model_name=args.model,
        output_filename_suffix=args.output,
        directory=args.directory,
    )
    separator.process_directory()


def main():
    log_system_resources()

    parser = argparse.ArgumentParser(
        description="LyricScribe: A tool for audio transcription and source separation."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.required = True

    # --- Whisper subcommand ---
    whisper_parser = subparsers.add_parser(
        "whisper", help="Transcribe audio files using Whisper."
    )
    whisper_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Whisper model to use (e.g., 'tiny', 'base' for OpenAI; 'openai/whisper-large-v3' for Transformers).",
    )
    input_group = whisper_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--directory",
        type=str,
        help="Directory containing ISRC folders, each with audio files.",
    )
    input_group.add_argument(
        "--file",
        type=str,
        help="Single audio file to transcribe.",
    )
    whisper_parser.add_argument(
        "--backend",
        type=str,
        required=True,
        choices=[
            "openai",
            "hf-sequential",
            "hf-chunked",
            "faster-whisper-sequential",
            "faster-whisper-chunked",
            "whisperx",
        ],
        help="The transcription backend to use.",
    )
    whisper_parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for beam search decoding (default: 5). Not supported for all backends.",
    )
    whisper_parser.add_argument(
        "--vad",
        action="store_true",
        help="Enable Voice Activity Detection (VAD) filtering. Not supported for all backends.",
    )
    whisper_parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for chunked processing (default: 16). Not supported for all backends.",
    )
    whisper_parser.set_defaults(func=handle_whisper_command)

    # --- Separator subcommand ---
    separator_parser = subparsers.add_parser(
        "separator", help="Separate audio sources."
    )
    separator_subparsers = separator_parser.add_subparsers(
        dest="separator_command", help="Available separation tools"
    )
    separator_subparsers.required = True

    # Demucs subcommand for separator
    demucs_parser = separator_subparsers.add_parser(
        "demucs", help="Separate audio using Demucs."
    )
    demucs_parser.add_argument(
        "--directory", type=str, required=True, help="Directory containing ISRC folders"
    )
    demucs_parser.add_argument(
        "--model",
        type=str,
        default="htdemucs",
        help="Demucs model to use (default: htdemucs)",
    )
    demucs_parser.add_argument(
        "--output",
        type=str,
        default="demucs_vocals.wav",
        help="Output filename for separated vocals (e.g., demucs_vocals.wav)",
    )
    demucs_parser.set_defaults(func=handle_demucs_command)

    # Spleeter subcommand for separator
    spleeter_parser = separator_subparsers.add_parser(
        "spleeter", help="Separate audio using Spleeter."
    )
    spleeter_parser.add_argument(
        "--directory", type=str, required=True, help="Directory containing ISRC folders"
    )
    spleeter_parser.add_argument(
        "--model",
        type=str,
        default="spleeter:2stems",
        help="Spleeter model to use (default: spleeter:2stems)",
    )
    spleeter_parser.add_argument(
        "--output",
        type=str,
        default="spleeter_vocals.wav",
        help="Output filename for separated vocals (e.g., spleeter_vocals.wav)",
    )
    spleeter_parser.set_defaults(func=handle_spleeter_command)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        if args.command == "separator" and not hasattr(args, "separator_command"):
            separator_parser.print_help()
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
