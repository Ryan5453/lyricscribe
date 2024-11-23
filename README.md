# STT Research Project

> [!IMPORTANT]  
> This project is currently in progress. Updates will be posted over time as results come back.

## Introduction
TODO

## Dataset

The dataset consists of 39,886 audio files, with 831 files reserved for testing (used in the benchmarks below) and 39,055 files for training. The dataset is multilingual, comprising approximately 58.2% English, 28.5% Spanish, and the remaining 13.3% distributed across 18 other languages.

## Benchmarks

| Model                  | WER Type  | Average   | English   | Spanish   |
| ---------------------- | --------- | --------- | --------- | --------- |
| Whisper Large v1       | Raw       | x.x       | x.x       | x.x       |
|                        | Filtered‡ | x.x       | x.x       | x.x       |
| └─ with VAD            | Raw       | x.x       | x.x       | x.x       |
|                        | Filtered‡ | x.x       | x.x       | x.x       |
| └─ with Demucs         | Raw       | x.x       | x.x       | x.x       |
|                        | Filtered‡ | x.x       | x.x       | x.x       |
| └─ with Demucs + VAD   | Raw       | x.x       | x.x       | x.x       |
|                        | Filtered‡ | x.x       | x.x       | x.x       |
| Whisper Large v2       | Raw       | x.x       | x.x       | x.x       |
|                        | Filtered‡ | x.x       | x.x       | x.x       |
| └─ with VAD            | Raw       | x.x       | x.x       | x.x       |
|                        | Filtered‡ | x.x       | x.x       | x.x       |
| └─ with Demucs         | Raw       | x.x       | x.x       | x.x       |
|                        | Filtered‡ | x.x       | x.x       | x.x       |
| └─ with Demucs + VAD   | Raw       | x.x       | x.x       | x.x       |
|                        | Filtered‡ | x.x       | x.x       | x.x       |
| Whisper Large v3       | Raw       | x.x       | x.x       | x.x       |
|                        | Filtered‡ | x.x       | x.x       | x.x       |
| └─ with VAD            | Raw       | x.x       | x.x       | x.x       |
|                        | Filtered‡ | x.x       | x.x       | x.x       |
| └─ with Demucs         | Raw       | x.x       | x.x       | x.x       |
|                        | Filtered‡ | x.x       | x.x       | x.x       |
| └─ with Demucs + VAD   | Raw       | x.x       | x.x       | x.x       |
|                        | Filtered‡ | x.x       | x.x       | x.x       |
| Whisper Large v3 Turbo | Raw       | x.x       | x.x       | x.x       |
|                        | Filtered‡ | x.x       | x.x       | x.x       |
| └─ with VAD            | Raw       | x.x       | x.x       | x.x       |
|                        | Filtered‡ | x.x       | x.x       | x.x       |
| └─ with Demucs         | Raw       | x.x       | x.x       | x.x       |
|                        | Filtered‡ | x.x       | x.x       | x.x       |
| └─ with Demucs + VAD   | Raw       | x.x       | x.x       | x.x       |
|                        | Filtered‡ | x.x       | x.x       | x.x       |
| NVIDIA Canary-1B       | Raw       | x.x       | x.x       | x.x       |
|                        | Filtered‡ | x.x       | x.x       | x.x       |
| └─ with VAD            | Raw       | x.x       | x.x       | x.x       |
|                        | Filtered‡ | x.x       | x.x       | x.x       |
| └─ with Demucs         | Raw       | x.x       | x.x       | x.x       |
|                        | Filtered‡ | x.x       | x.x       | x.x       |
| └─ with Demucs + VAD   | Raw       | x.x       | x.x       | x.x       |
|                        | Filtered‡ | x.x       | x.x       | x.x       |

> [!NOTE]
> - [Demucs](https://github.com/adefossez/demucs) refers to the `htdemucs` model being applied to the source audio file before transcription
> - Filtered‡ refers to WER scores with outliers removed using the IQR method (`Q1 - 1.5*IQR, Q3 + 1.5*IQR`)
