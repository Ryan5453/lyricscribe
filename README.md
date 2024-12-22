# STT Research Project

> [!IMPORTANT]  
> This project is currently in progress - this repository is a draft.

## Introduction
Most public libraries of music lyrics are generally high quality, but the size of the libraries are quite small. For example, MusixMatch claims to have ~11 million human lyric transcriptions in their library, but Spotify has ~100 million songs in their library, leading to over 90% of songs having no lyrics!

In 2022, while working on a project that involved music lyrics, I had the idea of using speech-to-text models to transcribe songs. Sadly, I found that at the time most speech-to-text models were not able to transcribe music well. However, upon the release of OpenAI's Whisper series of models later that year, I decided to revisit my original idea. I found that Whisper was *much* better at transcribing music than other STT models. However, the output of the best model at the time (`large`, now referred to as `large-v1`) still did not compare to human-written transcriptions.

Transcribing music is very different from transcribing regular human speech as there is significant background noise that is not easily predictable, and the way words are sung is different. Machine learning models exist that extract the vocals (or acapella) from the audio. I found that applying such preprocessing to the audio before attempting to transcribe the music resulted in a significant increase in quality. 

This research project is to study the effectof such preprocessing and the impacts that different Whisper pipelines have on the word error rate (WER).

## Dataset

This project uses a private dataset that consists of 39,886 audio files, with 831 files reserved for testing and 39,055 files for training. The dataset is multilingual, comprising approximately 58.2% English, 28.5% Spanish, and the remaining 13.3% distributed across 18 other languages. It will not be released for copyright reasons.

## Audio Source Separators

Two of the most common open-source audio source separators are [Spleeter](https://github.com/deezer/spleeter) and [Demucs](https://github.com/facebookresearch/demucs). 

## Whisper Implementations

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