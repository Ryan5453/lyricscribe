# LyricScribe

[Eric Yao](https://github.com/Yaomister) and I ([Ryan Fahey](https://github.com/Ryan5453)) are working on a research project/paper regarding Automatic Lyric Transcription across multiple ASR model architectures. This codebase contains tooling written to support our research.

## Installation

Install [uv](https://docs.astral.sh/uv/) and [kaldi](https://kaldi-asr.org/).

Afterwards, run:
```bash
git clone https://github.com/Ryan5453/lyricscribe
cd lyricscribe
uv venv
source .venv/bin/activate
export KALDI_ROOT= ????
uv pip install -e . --torch-backend=auto
```

## CLI Usage

The LyricScribe CLI contains five subcommands: `lyricscribe dataset` for downloading public ALT datasets, `lyricscribe separate` for mass vocal separation using [demucs-next](https://github.com/ryan5453/demucs-next), `lyricscribe transcribe` for batch ASR transcription, `lyricscribe evaluate` for transcription quality evaluation and plotting, and `lyricscribe artifacts` for artifact feature extraction and correlation analysis.

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

#### `lyricscribe separate reset`

This command resets a separation job so it can be re-run from scratch. It deletes the tracked Demucs outputs for the job and resets all chunk entries back to `pending`.

```bash
uv run lyricscribe separate reset --job-dir ./jobs/htdemucs_ft
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

<details>
<summary><h3>lyricscribe evaluate</h3></summary>

The `evaluate` commands compute Word Error Rate (WER) and other metrics to check transcription quality against ground truth lyrics.

#### `lyricscribe evaluate run`

Evaluates a single transcription job directory against the dataset's ground truth lyrics.

```bash
uv run lyricscribe evaluate run --job-dir ./jobs/whisper_vocals
```

Options:

- `--job-dir`: Path to job directory (required)

#### `lyricscribe evaluate summarize`

Recursively evaluates all complete transcription subdirectories inside a base jobs directory, computes their statistics, and aggregates the results into a single CSV file sorted by the best Mean WER.

```bash
uv run lyricscribe evaluate summarize --jobs-dir ./jobs --output evaluation_summary.csv
```

Options:

- `--jobs-dir`: Path to base jobs directory containing model subdirectories (required)
- `--output`: Output CSV file path (default: `evaluation_summary.csv`)

#### `lyricscribe evaluate plot`

Generates analysis SVG plots by reading evaluation data directly from job directories. Produces six charts covering baseline WER comparisons, error type breakdowns, error distribution by dataset, and pipeline error-profile shifts.

```bash
# Core evaluation plots
uv run lyricscribe evaluate plot \
    --jobs-dir ./jobs \
    --output-dir ./plots

# Include the artifact quartile chart (builds word-level data in memory)
uv run lyricscribe evaluate plot \
    --jobs-dir ./jobs \
    --output-dir ./plots \
    --alignments-dir ./alignments \
    --features-dir ./features \
    --results-file ./jobs/whisper_vocals/results.jsonl \
    --musdb-dir ./dataset/musdb_alt
```

Options:

- `--jobs-dir`: Path to base jobs directory containing model subdirectories (required)
- `--output-dir`: Directory to save the generated SVG plots (required)
- `--alignments-dir`: Directory of MFA alignment JSON files (enables artifact chart)
- `--features-dir`: Directory of artifact feature JSON files (enables artifact chart)
- `--results-file`: Path to results.jsonl with model transcriptions (enables artifact chart)
- `--musdb-dir`: Root MUSDB directory for ground truth lyrics (enables artifact chart)

Output files:

| File | Description |
|------|-------------|
| `baseline_wer.svg` | Grouped bar chart of WER by dataset configuration & model |
| `error_type_rates.svg` | Grouped bar chart of normalised insertion/deletion/substitution rates per model |
| `error_distribution.svg` | Stacked bar chart of error type distribution by model and dataset |
| `wer_heatmap.svg` | Heatmap of WER across all models × pipeline configurations |
| `error_type_breakdown.svg` | Stacked percentage bar chart of error type breakdown per model |
| `pipeline_shift.svg` | Per-model scatter of pipeline error-profile shift vs clean-stems baseline |
| `artifact_quartile_error.svg` | Line chart of error rate across artifact noise quartiles (requires artifact options) |

</details>

<details>
<summary><h3>lyricscribe artifacts</h3></summary>

The `artifacts` commands handle artifact feature extraction, Montreal Forced Alignment (MFA) processing, and correlation analysis between audio artifacts and transcription errors. These are used to investigate how separation artifacts (residual instruments bleeding into the vocal stem) affect ASR accuracy.

#### `lyricscribe artifacts extract`

Extracts per-frame artifact features from MUSDB songs by comparing separated vocals against the ground-truth vocal stems. Computes artifact RMS, vocal RMS, artifact-to-signal ratio, spectral centroid, and spectral flatness.

```bash
uv run lyricscribe artifacts extract \
    --musdb-dir ./dataset/musdb_alt \
    --output-dir ./features
```

Options:

- `--musdb-dir`: Root MUSDB directory (required)
- `--output-dir`: Directory to write per-song feature JSON files (required)

#### `lyricscribe artifacts align`

Runs Montreal Forced Aligner on the MUSDB dataset to produce word-level alignments. Handles corpus preparation, alignment, and JSON export in a single step using MFA's Python API.

```bash
uv run lyricscribe artifacts align \
    --musdb-dir ./dataset/musdb_alt \
    --output-dir ./alignments
```

Options:

- `--musdb-dir`: Root MUSDB directory containing song subdirectories (required)
- `--output-dir`: Directory to write per-song alignment JSON files (required)
- `--dictionary`: MFA dictionary name or path (default: `english_mfa`)
- `--acoustic-model`: MFA acoustic model name or path (default: `english_mfa`)

#### `lyricscribe artifacts build`

Builds a word-level CSV dataset that combines MFA alignments, artifact features, ground-truth lyrics, and model transcription errors. Each row represents one word for one model, with the artifact features averaged over that word's time window, the error type (correct, deletion, substitution) from jiwer alignment, and the count of hypothesis words inserted adjacent to this reference word. This CSV is useful for notebook exploration; plotting is handled by `evaluate plot`.

```bash
uv run lyricscribe artifacts build \
    --alignments-dir ./alignments \
    --features-dir ./features \
    --results-file ./jobs/whisper_vocals/results.jsonl \
    --musdb-dir ./dataset/musdb_alt \
    --output ./word_dataset.csv
```

Options:

- `--alignments-dir`: Directory of MFA alignment JSON files (required)
- `--features-dir`: Directory of artifact feature JSON files (required)
- `--results-file`: Path to results.jsonl with model transcriptions (required)
- `--musdb-dir`: Root MUSDB directory for ground truth lyrics (required)
- `--output`: Path to write the word-level CSV (required)

</details>
