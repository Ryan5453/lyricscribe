# SingingWhisper: STT Research

> [!WARNING]  
> This project is currently a draft, take everything with a grain of salt.

## Introduction
Most public libraries of music lyrics are generally high quality, but the size of the libraries are quite small. For example, MusixMatch claims to have ~11 million human lyric transcriptions in their library, but Spotify has ~100 million songs in their library, indicating that approximately 89% of songs lack lyrics.

In 2022, while working on a project involving music lyrics, I had the idea to use speech-to-text models to transcribe songs. Initial tests showed that most speech-to-text models at the time couldn't transcribe music effectively. However, when OpenAI released their Whisper series of models later that year, I revisited my original idea. I found that Whisper performed *significantly* better at transcribing music than other STT models, though the output of the best model at the time (`large`, now `large-v1`) still didn't match the quality of human-written transcriptions.

Transcribing music presents unique challenges compared to transcribing regular speech, as there is significant background noise that is not easily predictable, and the vocal delivery or lyrical expression often differs from natural speaking patterns. Machine learning models exist that extract the vocals (or acapella) from the audio. I found that applying such preprocessing to the audio before attempting to transcribe the music resulted in a significant increase in quality. 

This research project is to study the effect of such preprocessing, the impacts that different Whisper pipelines have on the word error rate (WER) along with ways to counteract it.

## Dataset

This project uses a private dataset that consists of 39,886 audio files, with 831 files reserved for testing and 39,055 files for training. The dataset is multilingual, comprising approximately 58.2% English, 28.5% Spanish, and the remaining 13.3% distributed across 18 other languages.[^1] It will not be released for copyright reasons.

[^1]: Language distribution was synthetically calculated based off Whisper `large-v3` outputs

## Audio Source Separators

Two of the most prominent open-source audio source separators are [Spleeter](https://github.com/deezer/spleeter) and [Demucs](https://github.com/facebookresearch/demucs). Generally, Demucs produces higher quality output than Spleeter but requires more resources to do so. Both audio source separators were tested on a 40GB SMX4 A100 GPU with 6 AMD EPYC 7543 cores and 24GB of RAM.

### Spleeter
Spleeter offers three model variants - `2stems`, `4stems`, and `5stems`. By default, the Spleeter models cut all output above 11kHz, but additional model configurations (using the same weights) allow separation up to 16kHz. Spleeter is notably fast: when processing the test set, the 2stem 11kHz configuration took approximately 0.19 seconds per song (~1148x realtime), while the 16kHz configuration took approximately 0.21 seconds per song (~1039x realtime).

### Demucs
Demucs has multiple model generations, but this analysis focuses on the most recent (Hybrid Transformer Demucs). The latest generation includes three models: `htdemucs` (the base model), `htdemucs_ft` (finetuned version), and `htdemucs_6s` (adds piano and guitar sources). According to the authors, the finetuned version produces higher quality output than the base model but requires 4x longer for inference. When processing the test set, the base model took approximately 4.43 seconds per song (~49x realtime), while the finetuned version took 20.14 seconds per song (~11x realtime).

## Whisper Implementations

Since the release of Whisper, there have been many alternative implementations that generally claim to either run faster or have better quality. However, despite the use of the same model weights, these alternate implementations can have varying accuracies - not always for the good.  

This analysis will focus on the following implementations:

- [Original OpenAI Implementation](https://github.com/openai/whisper)
- [Hugging Face's Transformers Library](https://github.com/huggingface/transformers)
- [FasterWhisper](https://github.com/SYSTRAN/faster-whisper)
- [WhisperX](https://github.com/m-bain/whisperX)
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) / [pywhispercpp](https://github.com/absadiki/pywhispercpp)

## Models

- Whisper Large v1
- Whisper Large v2
- Whisper Large v3
- Whisper Large v3 Turbo
- Distil-Whisper Large v2
- Distil-Whisper Large v3
- CrisperWhisper (based off large-v2)