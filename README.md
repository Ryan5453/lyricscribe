# LyricScribe

[Eric Yao](https://github.com/Yaomister) and I ([Ryan Fahey](https://github.com/Ryan5453)) are working on a research project/paper regarding Automatic Lyric Transcription across multiple ASR model architectures. This codebase contains tooling written to support our research.

## Installation

Assuming you have `uv` installed, you can install the dependencies with the following commands:

```bash
git clone https://github.com/Ryan5453/lyricscribe
cd lyricscribe
uv venv
source .venv/bin/activate
uv pip install -e . --torch-backend=auto
```

## CLI Usage

The LyricScribe CLI contains three subcommands: `lyricscribe dataset` for downloading public ALT datasets, `lyricscribe separate` for mass vocal separation using [demucs-next](https://github.com/ryan5453/demucs-next), and `lyricscribe transcribe` for batch ASR transcription.

<details>
<summary><h3>lyricscribe dataset</h3></summary>

The `dataset` commands download public ALT benchmark datasets and convert them into a standardized per-song directory layout with a `lyrics.json` file matching the project's Pydantic schema.

#### `lyricscribe dataset jam-alt`

Downloads the [Jam-ALT](https://huggingface.co/datasets/jamendolyrics/jam-alt) dataset (79 songs in 4 languages) from HuggingFace. Each song gets a directory containing `audio.mp3` and `lyrics.json`.

```bash
uv run lyricscribe dataset jam-alt --output-dir ./dataset/jam_alt
```

Options:

- `--output-dir`: Directory to write the dataset into (required)

Output structure:

```text
jam_alt/
├── SONG_NAME/
│   ├── audio.mp3
│   └── lyrics.json
└── ...
```

#### `lyricscribe dataset musdb-alt`

Downloads the [MUSDB-ALT](https://huggingface.co/datasets/jazasyed/musdb-alt) dataset (39 English songs). Lyrics are downloaded from HuggingFace, and audio is automatically downloaded from [MUSDB18-HQ](https://zenodo.org/records/3338373) on Zenodo (~30 GB, one-time download cached in `/tmp/lyricscribe/musdb18hq/`). Each song gets a directory containing `mixture.wav`, `vocals.wav`, and `lyrics.json`.

```bash
uv run lyricscribe dataset musdb-alt --output-dir ./dataset/musdb_alt
```

Options:

- `--output-dir`: Directory to write the dataset into (required)

Output structure:

```text
musdb_alt/
├── SONG_NAME/
│   ├── mixture.wav
│   ├── vocals.wav
│   └── lyrics.json
└── ...
```

</details>

<details>
<summary><h3>lyricscribe separate</h3></summary>

The `separate` commands expect the dataset directory to contain **subdirectories**, each containing an audio file to be separated. The subdirectory names are used as identifiers for tracking progress. For example:

```text
dataset/
├── song_001/
│   └── mix.wav
├── song_002/
│   └── mix.wav
└── ...
```

#### `lyricscribe separate setup`

To be able to mass separate audio files, you need to set up a separation job.
This creates a job directory with a config file and per-chunk JSON manifests to coordinate work across multiple workers.
It divides your dataset into chunks, tracks the status of each file, and stores the job configuration so workers can process independently.
This makes it possible to resume interrupted jobs since already-processed files are automatically skipped.

To set up a separation job:

```bash
# Save all stems (default)
uv run lyricscribe separate setup /path/to/dataset \
    --job-dir ./jobs/htdemucs_ft \
    --filename mix.wav \
    --model htdemucs_ft \
    --chunks 5

# Or isolate just one stem
uv run lyricscribe separate setup /path/to/dataset \
    --job-dir ./jobs/htdemucs_ft \
    --filename mix.wav \
    --model htdemucs_ft \
    --stem vocals \
    --chunks 5
```

Options:

- `--job-dir`: Directory to create for job files (required)
- `--filename`: Audio filename to process within each subdirectory, e.g. `mix.wav` (required)
- `--model`: Demucs model to use (default: htdemucs)
- `--stem`: Which stem to isolate - vocals, drums, bass, or other. If not specified, all stems are saved.
- `--chunks`: Number of chunks to split dataset into (default: 5)

#### `lyricscribe separate run`

This command can only be ran after you have run the `lyricscribe separate setup` command which creates the job directory.
You need to run this command however many times you specified with the `--chunks` argument in the setup command.

```bash
uv run lyricscribe separate run --job-dir ./jobs/htdemucs_ft --chunk-id 1
```

Options:

- `--job-dir`: Path to job directory (required)
- `--chunk-id`: Which chunk to process, 1-indexed (required)

Output files will be saved in the same directory as the mixed audio with the template {model}_{stem}.wav.

#### `lyricscribe separate inspect`

This command allows you to inspect the job details and show processing statistics from the job directory.

```bash
uv run lyricscribe separate inspect --job-dir ./jobs/htdemucs_ft
```

Options:

- `--job-dir`: Path to job directory (required)

</details>

<details>
<summary><h3>lyricscribe transcribe</h3></summary>

The `transcribe` commands run ASR inference on audio files using Whisper, Parakeet, Canary, or other compatible models. Like the separation commands, transcription uses a chunk-based job system for parallel SLURM processing with automatic resuming.

#### `lyricscribe transcribe setup`

Set up a transcription job by scanning dataset directories for audio files and splitting them into chunks.

```bash
# Basic setup
uv run lyricscribe transcribe setup /path/to/dataset \
    --job-dir ./jobs/whisper_vocals \
    --filename vocals.wav \
    --model openai/whisper-large-v3

# With VAD segmentation and multiple chunks
uv run lyricscribe transcribe setup /path/to/dataset \
    --job-dir ./jobs/parakeet_mixture \
    --filename mixture.wav \
    --model nvidia/parakeet-tdt-0.6b-v3 \
    --chunks 5 \
    --vad
```

Options:

- `--job-dir`: Directory to create for job files (required)
- `--filename`: Audio filename to transcribe within each subdirectory (required)
- `--model`: HuggingFace model ID (required). Whisper models use HuggingFace Transformers, all others use NeMo.
- `--chunks`: Number of chunks to split dataset into (default: 1)
- `--batch-size`: Batch size for inference (default: 1)
- `--vad`: Enable Silero VAD-based segmentation (flag)

#### `lyricscribe transcribe run`

Process one chunk of a transcription job. Results are appended to a JSONL file in the job directory.

```bash
uv run lyricscribe transcribe run --job-dir ./jobs/whisper_vocals --chunk-id 1
```

Options:

- `--job-dir`: Path to job directory (required)
- `--chunk-id`: Which chunk to process, 1-indexed (required)

Output files (`results_{chunk_id}.jsonl`) are saved in the job directory. Each line contains:

```json
{"song_id": "...", "audio_file": "...", "transcription": "...", "model_name": "...", "duration_seconds": 0.0, "error": null}
```

#### `lyricscribe transcribe inspect`

Inspect transcription job details and show processing statistics.

```bash
uv run lyricscribe transcribe inspect --job-dir ./jobs/whisper_vocals
```

Options:

- `--job-dir`: Path to job directory (required)

</details>
