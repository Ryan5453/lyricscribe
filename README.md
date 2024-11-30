# STT Research Project

> [!IMPORTANT]  
> This project is currently in progress. Updates will be posted over time as results come back.

## Introduction
TODO

## Dataset

The dataset consists of 39,886 audio files, with 831 files reserved for testing (used in the benchmarks below) and 39,055 files for training. The dataset is multilingual, comprising approximately 58.2% English, 28.5% Spanish, and the remaining 13.3% distributed across 18 other languages.

## Benchmarks
| Model                                       | WER Type  | Average | English | Spanish |
| ------------------------------------------- | --------- | ------- | ------- | ------- |
| Whisper Large v1                            | Raw       | 0.56    | 0.55    | 0.57    |
|                                             | Filtered‡ | 0.51    | 0.50    | 0.52    |
| └─ with VAD                                 | Raw       | 0.69    | 0.69    | 0.68    |
|                                             | Filtered‡ | 0.67    | 0.68    | 0.66    |
| └─ with Demucs                              | Raw       | 0.59    | 0.59    | 0.58    |
|                                             | Filtered‡ | 0.51    | 0.50    | 0.53    |
| └─ with Demucs + VAD                        | Raw       | 0.57    | 0.57    | 0.57    |
|                                             | Filtered‡ | 0.50    | 0.50    | 0.52    |
| Whisper Large v2                            | Raw       | 0.55    | 0.54    | 0.58    |
|                                             | Filtered‡ | 0.50    | 0.50    | 0.52    |
| └─ with VAD                                 | Raw       | 0.68    | 0.68    | 0.67    |
|                                             | Filtered‡ | 0.67    | 0.67    | 0.65    |
| └─ with Demucs                              | Raw       | 0.60    | 0.59    | 0.60    |
|                                             | Filtered‡ | 0.52    | 0.51    | 0.53    |
| └─ with Demucs + VAD                        | Raw       | 0.59    | 0.59    | 0.57    |
|                                             | Filtered‡ | 0.51    | 0.50    | 0.52    |
| Whisper Large v3                            | Raw       | 0.55    | 0.54    | 0.56    |
|                                             | Filtered‡ | 0.51    | 0.51    | 0.51    |
| └─ with VAD                                 | Raw       | 0.68    | 0.69    | 0.67    |
|                                             | Filtered‡ | 0.67    | 0.69    | 0.65    |
| └─ with Demucs                              | Raw       | 0.55    | 0.55    | 0.55    |
|                                             | Filtered‡ | 0.51    | 0.51    | 0.51    |
| └─ with Demucs + VAD                        | Raw       | 0.56    | 0.56    | 0.55    |
|                                             | Filtered‡ | **0.50**| **0.50**| **0.50**|
| Whisper Large v3 Turbo                      | Raw       | 0.61    | 0.63    | 0.58    |
|                                             | Filtered‡ | 0.53    | 0.53    | 0.52    |
| └─ with VAD                                 | Raw       | 0.69    | 0.71    | 0.67    |
|                                             | Filtered‡ | 0.68    | 0.70    | 0.65    |
| └─ with Demucs                              | Raw       | 0.57    | 0.58    | 0.57    |
|                                             | Filtered‡ | 0.53    | 0.54    | 0.51    |
| └─ with Demucs + VAD                        | Raw       | 0.57    | 0.58    | 0.55    |
|                                             | Filtered‡ | 0.52    | 0.52    | 0.51    |

> [!NOTE]
> - [Demucs](https://github.com/adefossez/demucs) refers to the `htdemucs` model being applied to the source audio file before transcription
> - Filtered‡ refers to WER scores with outliers removed using the IQR method (`Q1 - 1.5*IQR, Q3 + 1.5*IQR`)
