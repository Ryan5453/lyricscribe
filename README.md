# STT Research Project

> [!IMPORTANT]  
> This project is currently in progress. Updates will be posted over time as results come back.

## Dataset
The dataset used for this project contains a total of 39,886 audio files. 831 files are used exclusively for testing (the benchmarks below), while the other 39,055 are used for training. The dataset is multilingual - the test subset was randomly chosen so it should be approximately the same makeup as the training set. 

## Benchmarks

| Model                      | WER Type  | Average   | English | Language 2 | Language 3 | Language 4 |
| -------------------------- | --------- | --------- | ------- | ---------- | ---------- | ---------- |
| Whisper Large v1          | Raw       | 64.63     | x.x     | x.x        | x.x        | x.x        |
|                           | Filtered‡ | 63.77     | x.x     | x.x        | x.x        | x.x        |
| └─ with Demucs            | Raw       | 49.85     | x.x     | x.x        | x.x        | x.x        |
|                           | Filtered‡ | 43.18     | x.x     | x.x        | x.x        | x.x        |
| Whisper Large v2          | Raw       | 64.47     | x.x     | x.x        | x.x        | x.x        |
|                           | Filtered‡ | 63.27     | x.x     | x.x        | x.x        | x.x        |
| └─ with Demucs            | Raw       | 50.51     | x.x     | x.x        | x.x        | x.x        |
|                           | Filtered‡ | 43.29     | x.x     | x.x        | x.x        | x.x        |
| Whisper Large v3          | Raw       | 63.94     | x.x     | x.x        | x.x        | x.x        |
|                           | Filtered‡ | 63.16     | x.x     | x.x        | x.x        | x.x        |
| └─ with Demucs            | Raw       | 48.11     | x.x     | x.x        | x.x        | x.x        |
|                           | Filtered‡ | **42.26** | x.x     | x.x        | x.x        | x.x        |
| Whisper Large v3 Turbo    | Raw       | 65.06     | x.x     | x.x        | x.x        | x.x        |
|                           | Filtered‡ | 64.18     | x.x     | x.x        | x.x        | x.x        |
| └─ with Demucs            | Raw       | 48.45     | x.x     | x.x        | x.x        | x.x        |
|                           | Filtered‡ | 44.32     | x.x     | x.x        | x.x        | x.x        |
| NVIDIA Canary-1B          | Raw       | x.x       | x.x     | x.x        | x.x        | x.x        |
|                           | Filtered‡ | x.x       | x.x     | x.x        | x.x        | x.x        |
| └─ with Demucs            | Raw       | x.x       | x.x     | x.x        | x.x        | x.x        |
|                           | Filtered‡ | x.x       | x.x     | x.x        | x.x        | x.x        |

> [!NOTE]  
> [Demucs](https://github.com/adefossez/demucs) refers to the `htdemucs` model being applied to the source audio file before transcription
