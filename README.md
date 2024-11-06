# STT Research Project

> [!IMPORTANT]  
> This project is currently in progress. Updates will be posted over time as results come back.

## Dataset
The dataset used for this project contains a total of 39,886 audio files. 831 files are used exclusively for testing (the benchmarks below), while the other 39,055 are used for training. The dataset is multilingual - the test subset was randomly chosen so it should be approximately the same makeup as the training set. 

## Benchmarks

| Model                            | Average | English | Language 2 | Language 3 | Language 4 |
| -------------------------------- | ------- | ------- | ---------- | ---------- | ---------- |
|                                  | WER     | WER     | WER        | WER        | WER        |
| Whisper Large v1                 | x.x     | x.x     | x.x        | x.x        | x.x        |
| Whisper Large v1 w/ Demucs       | x.x     | x.x     | x.x        | x.x        | x.x        |
| Whisper Large v2                 | x.x     | x.x     | x.x        | x.x        | x.x        |
| Whisper Large v2 w/ Demucs       | x.x     | x.x     | x.x        | x.x        | x.x        |
| Whisper Large v3                 | x.x     | x.x     | x.x        | x.x        | x.x        |
| Whisper Large v3 w/ Demucs       | x.x     | x.x     | x.x        | x.x        | x.x        |
| Whisper Large v3 Turbo           | x.x     | x.x     | x.x        | x.x        | x.x        |
| Whisper Large v3 Turbo w/ Demucs | x.x     | x.x     | x.x        | x.x        | x.x        |
| NVIDIA Canary-1B                 | x.x     | x.x     | x.x        | x.x        | x.x        |
| NVIDIA Canary-1B w/ Demucs       | x.x     | x.x     | x.x        | x.x        | x.x        |

> [!NOTE]  
> [Demucs](https://github.com/adefossez/demucs) refers to the `htdemucs` model being applied to the source audio file before transcription
