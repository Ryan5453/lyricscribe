# STT Research

> [!WARNING]  
> This project is currently a draft, take everything with a grain of salt.

## Introduction
Most public libraries of music lyrics are generally high quality, but the size of the libraries are quite small. For example, MusixMatch claims to have ~11 million human lyric transcriptions in their library, but Spotify has ~100 million songs in their library, leading to around 89% of songs having no lyrics!

In 2022, while I was working on a project that involved music lyrics, I had the idea of using speech-to-text models to transcribe songs. After a few tests, I found that most speech-to-text models at the time were not able to transcribe music well. However, upon the release of OpenAI's Whisper series of models later that year, I decided to revisit my original idea. I found that Whisper was *much* better at transcribing music than other STT models but the output of the best model at the time (`large`, now `large-v1`) still did not compare to human-written transcriptions.

Transcribing music presents unique challenges compared to transcribing regular speech, as there is significant background noise that is not easily predictable, and the vocal delivery or lyrical expression often differs from natural speaking patterns. Machine learning models exist that extract the vocals (or acapella) from the audio. I found that applying such preprocessing to the audio before attempting to transcribe the music resulted in a significant increase in quality. 

This research project is to study the effect of such preprocessing, the impacts that different Whisper pipelines have on the word error rate (WER) along with ways to counteract it.

## Dataset

This project uses a private dataset that consists of 39,886 audio files, with 831 files reserved for testing and 39,055 files for training. The dataset is multilingual, comprising approximately 58.2% English, 28.5% Spanish, and the remaining 13.3% distributed across 18 other languages[^1]. It will not be released for copyright reasons.

[^1]: Language distribution was calculated synthetically based on Whisper `large-v3` outputs

## Audio Source Separators
Audio files: 181,254 seconds

Two of the most common open-source audio source separators are [Spleeter](https://github.com/deezer/spleeter) and [Demucs](https://github.com/facebookresearch/demucs). Generally, the quality of Demucs is much higher than the quality of Spleeter. Both audio source separators were run on a 40GB SMX4 A100 GPU on a system with 6 AMD EPYC 7543 cores and 24GB of RAM.

### Spleeter
Spleeter has three model varients - `2stems`, `4stems`, and `5stems`. By default, the Spleeter models cut all output above 11kHz but there are additional model configuraitions (using the same weights) that allow separation up to 16kHz. Spleeter can be *very* fast. When running on the test set, the 2stem 11kHz configuration took approximately 0.19 seconds per song (~1148x realtime), while the 16kHz configuration took approximately 0.21 seconds per song (~1039x realtime) 

### Demucs
Demucs has multiple model generations, but I chose to focus on the most recent (Hybrid Transformer Demucs.) The latest generation has three models, `htdemucs` (the base model), `htdemucs_ft` (finetuned version), and `htdemucs_6s` (adds piano and guitar sources.) The author claims that the finetuned version has higher quality than the base model but takes 4x longer to inference. When running the test set, the base model took approxamatly 4.43 seconds per song (~49x realtime), while the finetuned version took 20.14 seconds per song (~11x realtime.)

## Whisper Implementations

All implementations of Whisper are not created equal. 

- OpenAI's Implementation
- Hugging Face Transformers
- FasterWhisper
- WhisperX
- whisper.cpp

## Models

- Whisper Large v1
- Whisper Large v2
- Whisper Large v3
- Whisper Large v3 Turbo
- Distil-Whisper Large v2
- Distil-Whisper Large v3
- CrisperWhisper (based off large-v2)