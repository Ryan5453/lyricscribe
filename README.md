# STT Research Project

> [!IMPORTANT]  
> This project is currently in progress. Updates will be posted over time as results come back.

## Dataset

The dataset used for this project contains a total of 39,886 audio files. 831 files are used exclusively for testing (the benchmarks below), while the other 39,055 are used for training. The dataset is multilingual - approximately 58.2% English and 28.5% Spanish, with the remaining 13.3% is made up of 18 other languages.

## Benchmarks

| Model                  | WER Type  | Average   | English   | Spanish   |
| ---------------------- | --------- | --------- | --------- | --------- |
| Whisper Large v1       | Raw       | 51.55     | 44.44     | 46.40     |
|                        | Filtered‡ | 45.00     | 40.69     | 42.73     |
| └─ with VAD            | Raw       | 64.63     | 63.17     | 60.74     |
|                        | Filtered‡ | 63.77     | 62.37     | 59.91     |
| └─ with Demucs         | Raw       | 52.80     | 48.32     | 48.87     |
|                        | Filtered‡ | 45.49     | 40.37     | 43.87     |
| └─ with Demucs + VAD   | Raw       | 49.85     | 46.22     | 46.48     |
|                        | Filtered‡ | 43.18     | 39.16     | 42.29     |
| Whisper Large v2       | Raw       | 51.53     | 43.41     | 46.71     |
|                        | Filtered‡ | 46.24     | 40.32     | 41.94     |
| └─ with VAD            | Raw       | 64.47     | 62.10     | 59.99     |
|                        | Filtered‡ | 63.27     | 61.69     | 58.85     |
| └─ with Demucs         | Raw       | 52.43     | 49.03     | 50.92     |
|                        | Filtered‡ | 45.20     | 40.47     | 44.80     |
| └─ with Demucs + VAD   | Raw       | 50.51     | 48.41     | 47.20     |
|                        | Filtered‡ | 43.29     | 40.34     | 42.54     |
| Whisper Large v3       | Raw       | 50.95     | 44.88     | 46.09     |
|                        | Filtered‡ | 44.25     | 41.80     | 41.70     |
| └─ with VAD            | Raw       | 63.94     | 64.04     | 59.64     |
|                        | Filtered‡ | 63.16     | 63.40     | 58.80     |
| └─ with Demucs         | Raw       | 48.70     | 45.32     | 46.27     |
|                        | Filtered‡ | 44.47     | 42.01     | 42.14     |
| └─ with Demucs + VAD   | Raw       | 48.11     | 45.74     | 44.40     |
|                        | Filtered‡ | **42.26** | **39.50** | **39.91** |
| Whisper Large v3 Turbo | Raw       | 53.76     | 53.39     | 47.58     |
|                        | Filtered‡ | 45.36     | 44.13     | 43.12     |
| └─ with VAD            | Raw       | 65.06     | 65.38     | 60.11     |
|                        | Filtered‡ | 64.18     | 64.77     | 59.01     |
| └─ with Demucs         | Raw       | 49.96     | 47.81     | 47.59     |
|                        | Filtered‡ | 46.45     | 45.27     | 43.79     |
| └─ with Demucs + VAD   | Raw       | 48.45     | 46.59     | 45.04     |
|                        | Filtered‡ | 44.32     | 42.04     | 42.36     |
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
