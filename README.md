# LyricScribe

[Eric Yao](https://github.com/Yaomister) and I ([Ryan Fahey](https://github.com/Ryan5453)) are working on a research project/paper regarding Automatic Lyric Transcription across multiple ASR model architectures. This codebase contains tooling written to support our research.

## Installation

Assuming you have [uv](https://docs.astral.sh/uv/) installed, run:

```bash
git clone https://github.com/Ryan5453/lyricscribe
cd lyricscribe
uv sync
source .venv/bin/activate
```

Note: The project is configured for CUDA 12.9 (`cu129`). If you need a different CUDA version, update the `[[tool.uv.index]]` URL in `pyproject.toml` (e.g. `cu126` for CUDA 12.6).

## CLI Usage

| Command | Purpose |
|---------|---------|
| `lyricscribe dataset` | Download public ALT datasets and add word-level forced alignments |
| `lyricscribe separate` | Batch vocal separation with [Unblend](https://pypi.org/project/unblend/) |
| `lyricscribe transcribe` | Batch transcription with Whisper, Parakeet, Canary, Qwen3-ASR, Cohere, Granite, Nemotron and audio-LLM backends |
| `lyricscribe evaluate` | WER evaluation, LaTeX tables and plots across transcription jobs |
| `lyricscribe artifacts` | Separation-artifact features and a word-level error dataset |
| `lyricscribe finetune` | Finetuning with step-based schedules, resumable on SLURM |
| `lyricscribe ensemble` | ROVER system combination and repetition repair |
| `lyricscribe analyze` | Scoring, reference self-WER, significance testing and checkpoint averaging |

`separate` and `transcribe` share a job model: `setup` writes a job directory split into chunks, `run --chunk-id N` processes one chunk (typically one SLURM array task), and finished work is skipped on rerun. A `finetune` job is one run that `finetune run` resumes. Every command takes `--help`.

<details>
<summary><h3>lyricscribe dataset</h3></summary>

The `dataset` commands download public ALT benchmark datasets and convert them into a per-song directory layout with a `lyrics.json` file matching the project's Pydantic schema.

#### `lyricscribe dataset jam-alt`

Downloads the [Jam-ALT](https://huggingface.co/datasets/jamendolyrics/jam-alt) dataset (79 songs in 4 languages) from HuggingFace. Each song gets a directory containing `audio.mp3` and `lyrics.json`.

```bash
lyricscribe dataset jam-alt --output-dir ./dataset/jam_alt
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

Downloads the [MUSDB-ALT](https://huggingface.co/datasets/jazasyed/musdb-alt) dataset (39 English songs). Lyrics come from HuggingFace and audio from [MUSDB18-HQ](https://zenodo.org/records/3338373) on Zenodo (~30 GB, one-time download cached in `/tmp/lyricscribe/musdb18hq/`). Each song gets a directory containing `mixture.wav`, `vocals.wav` and `lyrics.json`.

```bash
lyricscribe dataset musdb-alt --output-dir ./dataset/musdb_alt
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

#### `lyricscribe dataset align`

Runs Montreal Forced Aligner via Singularity/Apptainer and writes word-level alignments into each song's `lyrics.json` under `alignment`. Works on any dataset in the per-song layout. Each song is aligned with the MFA model for its detected language.

Pull the MFA image and download models once:

```bash
singularity pull mfa.sif docker://mmcauliffe/montreal-forced-aligner:latest
mkdir -p /path/to/mfa_cache
singularity exec --env MFA_ROOT_DIR=/mfa_root -B /path/to/mfa_cache:/mfa_root mfa.sif \
    mfa model download acoustic english_mfa
singularity exec --env MFA_ROOT_DIR=/mfa_root -B /path/to/mfa_cache:/mfa_root mfa.sif \
    mfa model download dictionary english_mfa
```

Then align, passing whichever audio file each song should be aligned against (e.g. a separated stem):

```bash
lyricscribe dataset align \
    --dataset-dir ./dataset/musdb_alt \
    --filename vocals.wav \
    --container ./mfa.sif \
    --mfa-root /path/to/mfa_cache
```

Options:

- `--dataset-dir`: Root dataset directory containing song subdirectories (required)
- `--filename`: Audio filename inside each song subdirectory (default: `vocals.wav`)
- `--container` / `-c`: Path to the MFA `.sif` image (or set `LYRICSCRIBE_MFA_CONTAINER`)
- `--mfa-root`: Host directory for cached MFA pretrained models
- `--num-chunks`: Total shards for parallel execution (default: 1)
- `--chunk-id`: 0-indexed shard to process, usually `$SLURM_ARRAY_TASK_ID` (default: 0)
- `--skip-existing` / `--no-skip-existing`: Skip songs that already have an alignment (default: on)

After alignment, each song's `lyrics.json` has an `alignment` field like:

```json
{
  "alignment": {
    "words": [
      {"word": "she's", "start": 2310, "duration": 530},
      {"word": "the", "start": 2840, "duration": 340}
    ],
    "source_audio": "vocals.wav",
    "mfa_model": "english_mfa",
    "generated_at": "2026-04-18T12:34:56Z"
  }
}
```

Times are in milliseconds, matching the `synced` field's convention.

</details>

<details>
<summary><h3>lyricscribe separate</h3></summary>

The `separate` commands run [Unblend](https://pypi.org/project/unblend/) over a dataset of song subdirectories, each containing the audio file to separate. Subdirectory names identify songs for progress tracking:

```text
dataset/
├── song_001/
│   └── mix.wav
├── song_002/
│   └── mix.wav
└── ...
```

#### `lyricscribe separate setup`

Creates a job directory with the config and per-chunk manifests.

```bash
# Save all stems
lyricscribe separate setup /path/to/dataset \
    --job-dir ./jobs/htdemucs_ft \
    --filename mix.wav \
    --model htdemucs_ft \
    --chunks 5

# Isolate one stem
lyricscribe separate setup /path/to/dataset \
    --job-dir ./jobs/htdemucs_ft \
    --filename mix.wav \
    --model htdemucs_ft \
    --stem vocals \
    --chunks 5
```

Options:

- `DIRECTORIES`: One or more dataset directories (positional, required)
- `--job-dir`: Directory to create for job files (required)
- `--filename`: Audio filename to separate within each subdirectory (required)
- `--model`: Unblend model (default: `htdemucs`). `unblend models list` shows the options; custom checkpoints are added with `unblend models import`.
- `--stem`: Stem to isolate (`vocals`, `drums`, `bass`, `other`). All stems are saved if omitted.
- `--output-filename`: Output filename for a single isolated stem (default: `{model}_{stem}.wav`)
- `--shifts`: Random time-shift passes to average; 0 is deterministic (default: 1)
- `--split-overlap`: Overlap between consecutive model segments, in [0, 1) (default: 0.25)
- `--seed`: Seed for the shift offsets (default: 0)
- `--dtype`: `auto` (FP16 on CUDA), `float32`, `float16` or `bfloat16` (default: `auto`)
- `--chunks`: Number of chunks (default: 5)

#### `lyricscribe separate run`

Processes one chunk. Stems are written next to the source audio as `{model}_{stem}.wav`.

```bash
lyricscribe separate run --job-dir ./jobs/htdemucs_ft --chunk-id 1
```

Options:

- `--job-dir`: Path to job directory (required)
- `--chunk-id`: Chunk to process, 1-indexed (required)

#### `lyricscribe separate inspect`

Shows the job config and processing statistics.

```bash
lyricscribe separate inspect --job-dir ./jobs/htdemucs_ft
```

Options:

- `--job-dir`: Path to job directory (required)

#### `lyricscribe separate reset`

Deletes the job's separation outputs and resets every chunk to `pending`.

```bash
lyricscribe separate reset --job-dir ./jobs/htdemucs_ft
```

Options:

- `--job-dir`: Path to job directory (required)

</details>

<details>
<summary><h3>lyricscribe transcribe</h3></summary>

The `transcribe` commands run batch ASR over a dataset of song subdirectories. The backend is detected from the model id (or a local checkpoint's `config.json`) and can be forced with `--backend`:

| Backend | Example models |
|---------|----------------|
| `whisper` | `openai/whisper-large-v3`, Whisper finetunes |
| `qwen3_asr` | `Qwen/Qwen3-ASR-1.7B` and finetunes |
| `qwen3_omni` | `Qwen/Qwen3-Omni-30B-A3B-Instruct`, optionally with a LoRA `--adapter` |
| `musicflamingo` | NVIDIA Music Flamingo |
| `cohere` | `CohereLabs/cohere-transcribe-03-2026` |
| `granite` | `ibm-granite/granite-speech-5.0-2b` |
| `nemotron` | `nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B` |
| `parakeet` | `nvidia/parakeet-tdt-0.6b-v3` and Parakeet TDT finetunes |
| `canary` | `nvidia/canary-1b-v2` and Canary finetunes |

Every backend takes the whole song, or VAD segments of it with `--vad`: Silero VAD gives the raw speech segments, `--vad-method rms` merges RMS-VAD segments into chunks of at most 30 s. Checkpoints must be in transformers format; `.nemo` files are not supported.

#### `lyricscribe transcribe setup`

Scans the dataset and writes the job config and chunk manifests.

```bash
# Whisper on separated vocals
lyricscribe transcribe setup /path/to/dataset \
    --job-dir ./jobs/whisper_vocals \
    --filename vocals.wav \
    --model openai/whisper-large-v3 \
    --lyrics-file lyrics.json

# Parakeet on the mix, segmented with Silero VAD, in 5 chunks
lyricscribe transcribe setup /path/to/dataset \
    --job-dir ./jobs/parakeet_mixture \
    --filename mixture.wav \
    --model nvidia/parakeet-tdt-0.6b-v3 \
    --chunks 5 \
    --vad
```

Job options:

- `DIRECTORIES`: One or more dataset directories (positional, required)
- `--job-dir`: Directory to create for job files (required)
- `--filename`: Audio filename to transcribe within each subdirectory (required)
- `--model`: HuggingFace model id or local checkpoint path (required)
- `--backend`: Force a backend instead of detecting it (see the table above)
- `--chunks`: Number of chunks (default: 1)
- `--batch-size`: Inference batch size (default: 1)
- `--lyrics-file`: JSON file in each song directory to read `detected_language` from (e.g. `lyrics.json`)

Segmentation:

- `--vad`: Segment the audio with VAD before transcribing
- `--vad-method`: `silero` (speech-trained) or `rms` (amplitude on separated vocals) (default: `silero`)
- `--vad-source`: Audio file to run VAD on, e.g. a separated vocal stem, while transcribing `--filename`. Required with `--vad-method rms`.
- `--chunked`: Fixed-length chunked inference instead of full context

Decoding:

- `--repetition-penalty`: Multiplies the log-probability of every already-generated token by this factor; the prompt is never penalised. 1.0 disables (default: 1.0)
- `--no-repeat-ngram-size`: Ban repeating a generated n-gram; the prompt is never counted. 0 disables (default: 0)
- `--mit`: Shortcut for `--repetition-penalty 1.7 --no-repeat-ngram-size 5`
- `--num-beams`: Beam width; 1 is greedy (default: 1). Parakeet uses NeMo's TDT beam search.

Every backend applies all three; a backend refuses a setting it can't apply.
- `--condition-on-prev` / `--no-condition-on-prev`: Whisper long-form conditioning on previous windows (default: on)
- `--temperature-fallback`: Whisper temperature fallback, triggered by `--compression-ratio-threshold` (default 2.4) and `--logprob-threshold` (default -1.0)
- `--prompt-from-dirname`: Use the song directory name as Whisper's decoder prompt or Qwen3-ASR's context
- `--max-new-tokens`: Per-chunk generation cap for the autoregressive backends other than Whisper; Canary also stops at its encoder length + 50 tokens (default: 1024)
- `--adapter`: LoRA adapter directory merged into Qwen3-Omni
- `--auto-language`: Let Qwen3-ASR identify the language instead of forcing the `--lyrics-file` one
- `--deloop-max-run`: Collapse any short cycle repeated more than this many times; the raw text is kept as `transcription_raw`

#### `lyricscribe transcribe run`

Processes one chunk and appends results to `results_{chunk_id}.jsonl` in the job directory.

```bash
lyricscribe transcribe run --job-dir ./jobs/whisper_vocals --chunk-id 1
```

Options:

- `--job-dir`: Path to job directory (required)
- `--chunk-id`: Chunk to process, 1-indexed (required)

Each result line looks like:

```json
{"song_id": "...", "audio_file": "...", "transcription": "...", "model_name": "...", "duration_seconds": 0.0, "error": null}
```

#### `lyricscribe transcribe inspect`

Shows the job config and processing statistics.

```bash
lyricscribe transcribe inspect --job-dir ./jobs/whisper_vocals
```

Options:

- `--job-dir`: Path to job directory (required)

#### `lyricscribe transcribe reset`

Deletes the job's `results*.jsonl` files and resets every chunk to `pending`.

```bash
lyricscribe transcribe reset --job-dir ./jobs/whisper_vocals
```

Options:

- `--job-dir`: Path to job directory (required)

</details>

<details>
<summary><h3>lyricscribe evaluate</h3></summary>

#### `lyricscribe evaluate run`

Evaluates every complete transcription job under a jobs directory and writes, in one pass: a CSV sorted by WER, LaTeX tables under `tables/`, and PDF plots of baseline WER, error-type breakdowns and pipeline error-profile shifts.

```bash
# CSV, LaTeX tables and plots
lyricscribe evaluate run \
    --jobs-dir ./jobs \
    --output-dir ./results

# Also the artifact quartile chart. Run `lyricscribe dataset align` on the MUSDB directory first.
lyricscribe evaluate run \
    --jobs-dir ./jobs \
    --output-dir ./results \
    --features-dir ./features \
    --results-file ./jobs/whisper_vocals/results.jsonl \
    --results-file ./jobs/parakeet_vocals/results.jsonl \
    --musdb-dir ./dataset/musdb_alt
```

Options:

- `--jobs-dir`: Base jobs directory containing model subdirectories (required)
- `--output-dir`: Directory for the CSV, `tables/*.tex` and PDF plots (required)
- `--features-dir`: Artifact feature JSON files (enables the artifact chart)
- `--results-file`: A `results.jsonl` to include in the artifact chart; repeatable
- `--results-job-name`: Job subdirectory name to auto-discover `results*.jsonl` across every model directory under `--jobs-dir`
- `--musdb-dir`: Root MUSDB directory, for alignments and references (enables the artifact chart)

Output files:

| File | Description |
|------|-------------|
| `evaluation_summary.csv` | Every job's WER and error counts, sorted by WER |
| `baseline_wer.pdf` | Grouped bar chart of WER by dataset configuration and model |
| `error_type_rates.pdf` | Normalised insertion/deletion/substitution rates per model |
| `error_distribution.pdf` | Stacked error-type distribution by model and dataset |
| `wer_heatmap.pdf` | WER across all models × pipeline configurations |
| `error_type_breakdown.pdf` | Stacked percentage error-type breakdown per model |
| `pipeline_shift.pdf` | Per-model error-profile shift vs the clean-stems baseline |
| `artifact_quartile_error.pdf` | Error rate across artifact noise quartiles (requires the artifact options) |

</details>

<details>
<summary><h3>lyricscribe artifacts</h3></summary>

The `artifacts` commands measure how separation artifacts (residual instruments bleeding into the vocal stem) relate to transcription errors. Both need MUSDB-ALT, whose clean vocal stems serve as the reference.

#### `lyricscribe artifacts extract`

Compares separated vocals against the clean vocal stems and writes per-frame features per song: artifact RMS, vocal RMS, artifact-to-signal ratio, spectral centroid and spectral flatness.

```bash
lyricscribe artifacts extract \
    --musdb-dir ./dataset/musdb_alt \
    --output-dir ./features
```

Options:

- `--musdb-dir`: Root MUSDB directory (required)
- `--output-dir`: Directory for the per-song feature JSON files (required)

#### `lyricscribe artifacts build`

Builds a word-level CSV: one row per reference word per model, with that word's averaged artifact features, its error type (correct, deletion, substitution) and the number of hypothesis words inserted next to it. Run `lyricscribe dataset align` on the MUSDB directory first.

```bash
lyricscribe artifacts build \
    --features-dir ./features \
    --results-file ./jobs/whisper_vocals/results.jsonl \
    --results-file ./jobs/parakeet_vocals/results.jsonl \
    --musdb-dir ./dataset/musdb_alt \
    --output ./word_dataset.csv
```

Options:

- `--features-dir`: Artifact feature JSON files (required)
- `--results-file`: A `results.jsonl` to include; repeatable (required)
- `--results-jobs-dir` / `--results-job-name`: Also auto-discover `results*.jsonl` under a jobs root
- `--musdb-dir`: Root MUSDB directory, for alignments and references (required)
- `--output`: Path for the word-level CSV (required)

</details>

<details>
<summary><h3>lyricscribe finetune</h3></summary>

The `finetune` commands train Whisper, Parakeet, Canary, Cohere Transcribe, Qwen3-ASR or Granite Speech (full finetune), or Qwen3-Omni, Nemotron 3.5 Transcribe or Music Flamingo (NVIDIA's adapter recipe: full-rank audio encoder and projector, LoRA on the LLM), on 30-second windows sampled from songs. Each song directory needs a `lyrics.json` and at least one of the `--filename` audio files.

Every model trains with the same loop: a seeded window order, a global batch made up by gradient accumulation, AdamW, SpecAugment on the log-mel features, a token-level loss over the whole global batch, and a step-based learning-rate schedule. A run resumes from its latest checkpoint, so rerunning `finetune run` after a time limit continues it.

**Audio files should be PCM WAV.** Lossy formats like MP3 report a header duration that can differ from the decoded sample count by up to ~1s, so windows near the end of a song can come back short.

#### `lyricscribe finetune setup`

Builds (or copies) the train/validation manifests and fixes the run's step plan.

```bash
# Whisper on separated vocals, 3 epochs, warmup-stable-decay with a 20% cooldown
lyricscribe finetune setup /path/to/train \
    --output-dir ./experiments \
    --val-dir /path/to/validation \
    --model openai/whisper-large-v3 \
    --filename vocals.wav

# Parakeet with a cosine schedule, exporting a model after every epoch
lyricscribe finetune setup /path/to/train \
    --output-dir ./experiments \
    --model nvidia/parakeet-tdt-0.6b-v3 \
    --filename vocals.wav \
    --schedule cosine --warmup-steps 5000 --min-lr 5e-6 --learning-rate 1e-4 \
    --export-every-epoch
```

Defaults depend on the architecture:

| Architecture | Batch size per device | Learning rate | Adaptation |
|--------------|----------------------:|--------------:|------------|
| Whisper | 16 | 1e-5 | full |
| Parakeet | 4 | 3.16e-5 | full |
| Canary | 16 | 3.16e-5 | full |
| Cohere Transcribe | 8 | 3.16e-5 | full |
| Qwen3-ASR | 4 | 3.16e-5 | full (or `nvidia_adapter`, LR 1e-4) |
| Granite Speech | 1 | 3.16e-5 | full |
| Nemotron 3.5 Transcribe | 1 | 1e-4 | `nvidia_adapter` |
| Qwen3-Omni | 1 | 1e-4 | `nvidia_adapter` |
| Music Flamingo | 1 | 1e-4 | `nvidia_adapter` |

Core options:

- `TRAIN_DIR`: Directory of training songs (positional, required)
- `--output-dir`: Directory for experiment outputs (required)
- `--model`: Base model id; the architecture is detected from it (required)
- `--filename`: Audio filename to train on; repeat for multi-file training (required)
- `--val-dir`: Directory of validation songs; a fixed subset is scored at every export
- `--exp-name`: Experiment directory name (default: `<arch>_<filenames>_<UTC timestamp>`)
- `--batch-size`, `--learning-rate`, `--adaptation`: Override the defaults above
- `--global-batch`: Windows per optimizer step; gradient accumulation makes up the rest (default: 64)
- `--weight-decay`: AdamW weight decay, not on biases or norms (default: 0.01)
- `--seed`: Seeds the window order, dither, SpecAugment masks and LoRA init (default: 0)
- `--prompt-mode`: Training prompt, named as in `transcribe setup --prompt-mode`; recorded with every export
- `--no-augment`: Disable SpecAugment
- `--eval-subset-size`: Validation windows scored at every export (default: 200)

Run length, schedule and outputs:

- `--epochs` / `--max-steps`: Run length in (possibly fractional) epochs or optimizer steps (default: 3 epochs)
- `--schedule`: `wsd` (warmup, constant, then a (1 − √) cooldown; default), `cosine` or `linear`
- `--warmup-steps`: Linear warmup steps (default: 200)
- `--cooldown`: `wsd` only: fraction of the run that cools down (default: 0.2). `0` gives a constant-LR run to continue with `extend` and branch cooldowns off with `branch`.
- `--min-lr`: Final learning rate for `cosine` and `linear` (default: 0)
- `--keep`: Step whose resumable checkpoint is kept; repeatable (rolling checkpoints are otherwise replaced)
- `--resume-every`: Steps between rolling resume checkpoints (default: 500)
- `--export-every-epoch` / `--export-step`: Also export inference models at these steps
- `--no-export-final`: Don't export at the final step

Training windows and labels:

- `--windows-per-song`: Random 30s windows sampled per ~30s of song (default: 1)
- `--line-overlap-threshold`: Fraction of a synced line that must fall inside a window for its text to count (default: 0.7)
- `--max-unlabeled-vocal-frac`: Reject windows where more than this fraction of sung time has no label text; offsets are re-drawn
- `--ref-selfwer-file` / `--max-ref-self-wer`: Skip songs whose two reference transcripts disagree (artifact from `lyricscribe analyze selfwer`; 0.3 is a good cutoff)
- `--no-mfa-alignment`: Build labels from synced lines only, ignoring MFA word timings
- `--dedup-max-run`: Cap runs of identical consecutive label words; 0 disables (default: 0)
- `--whisper-timestamp-prob`: Whisper only: fraction of samples trained with timestamp labels; 0 breaks >30s decoding (default: 0.5)
- `--audio-domain-mode`: With several `--filename` values, `balanced` puts each song wholly in the first file's domain or the rest, 50/50 by window count; `fallback` uses the rest. Default: a random file per song.
- `--manifests-from`: Copy the manifests from an existing experiment or a `prepare-manifests` set instead of building them
- `--max-train-windows` / `--train-window-seed`: Train on a seeded random subset of windows
- `--freeze-encoder`: Parakeet/Canary: train only the decoder and joint/head

#### `lyricscribe finetune run`

Trains a job to its final step, resuming from its latest checkpoint. For several GPUs, launch it through `torchrun`.

```bash
lyricscribe finetune run --job-dir ./experiments/my_experiment

# Four GPUs
torchrun --nproc_per_node=4 -m lyricscribe.cli.main finetune run --job-dir ./experiments/my_experiment
```

Each export is written to `exports/step_<N>` with a `lyricscribe_train.json` record of how it was trained; `metrics.jsonl` gets one row per export with the last training loss and, with a validation set, the evaluation loss.

#### `lyricscribe finetune extend` and `lyricscribe finetune branch`

A run set up with `--cooldown 0` holds its learning rate constant after warmup. `extend` continues it to a later step, and `branch` starts a cooldown off one of its kept checkpoints as a job of its own under `<job-dir>/branches/<name>`, which `finetune run` trains and exports at its end.

```bash
lyricscribe finetune setup /path/to/train --output-dir ./experiments --exp-name trunk \
    --model nvidia/parakeet-tdt-0.6b-v3 --filename vocals.wav --cooldown 0 --max-steps 780 --no-export-final
lyricscribe finetune run --job-dir ./experiments/trunk
lyricscribe finetune branch --job-dir ./experiments/trunk --name short --to 976
lyricscribe finetune run --job-dir ./experiments/trunk/branches/short
```

- `extend`: `--job-dir`, `--to` (new final step), `--keep` (extra kept steps, repeatable)
- `branch`: `--job-dir`, `--name`, `--to` or `--epoch` (end of the branch), `--at` (kept checkpoint to start from; default: `--cooldown` (0.2) of the end step before it)

#### `lyricscribe finetune prepare-manifests`

Prepares one model-agnostic manifest set from an experiment's manifests: drops excluded songs, re-points the audio file name, rebuilds Whisper's timestamp segments, drops windows over Whisper's label limit, and redraws the evaluation subset away from given songs. Use it with `setup --manifests-from`.

#### `lyricscribe finetune throughput`

Times training steps with the full training setup and reports seconds per step, GPU-hours per epoch and peak memory.

```bash
lyricscribe finetune throughput --model nvidia/parakeet-tdt-0.6b-v3 \
    --manifest ./prepared/train_manifest.jsonl --output-dir /tmp/throughput
```

#### `lyricscribe finetune inspect` and `lyricscribe finetune status`

`inspect --job-dir` shows one job's progress, exports and metrics; `status --experiments-dir` prints one line per job: step, status, latest export and last loss.

#### `lyricscribe finetune retry`, `retry-all` and `reset`

`retry --job-dir` marks a failed job pending again (its checkpoints are kept, so the next run resumes); `retry-all --experiments-dir` does so for every failed job in a directory. `reset --job-dir --force` deletes a job's checkpoints, exports and training state.

#### `lyricscribe finetune tune-batch`

Finds the largest batch size that fits a model on the current GPUs. Each trial is a short `torchrun` training run on real songs; the sweep steps up through candidate sizes until the first OOM, then bisects.

```bash
lyricscribe finetune tune-batch \
    --dataset-dir /path/to/train \
    --filename vocals.wav \
    --model nvidia/parakeet-tdt-0.6b-v3
```

Options:

- `--dataset-dir`: Directory of song subdirectories (required)
- `--filename`: Audio file to read in each song directory; match production (required)
- `--model`: Model id (required)
- `--freeze-encoder`: Freeze the encoder (Canary/Parakeet); match production
- `--start` / `--max`: Smallest and largest batch sizes to try (default: 1 and 128)
- `--min-gap`: Stop bisecting once the gap is this small (default: 2)
- `--safety-margin`: Percent taken off the VRAM ceiling when there is no throughput plateau (default: 15)
- `--timeout`: Seconds before a trial is killed (default: 900)

#### `lyricscribe finetune export-model`

Copies one of a job's exports elsewhere for use with `lyricscribe transcribe`.

```bash
lyricscribe finetune export-model \
    --job-dir ./experiments/my_experiment \
    --output ./models/my_finetuned_model
```

Options:

- `--job-dir`: Path to the experiment directory (required)
- `--output`: Path to copy the export to (required)
- `--step`: Export step to copy (default: latest)

#### `lyricscribe finetune eval-loss`

Compares a base Qwen model against finetuned checkpoints by mean per-token cross-entropy on held-out windows. Use this rather than the trainer's logged loss, which is accumulation-scaled and measured in train mode.

```bash
lyricscribe finetune eval-loss \
    --base-model Qwen/Qwen3-ASR-1.7B-hf \
    --checkpoint ./experiments/qasr_run/exports/step_11706 \
    --manifest ./experiments/qasr_run/val_manifest.jsonl
```

Options:

- `--base-model`: Base hub id or path (required)
- `--checkpoint`: Checkpoint to compare; repeatable. A full model for Qwen3-ASR, a LoRA adapter for Qwen3-Omni. (required)
- `--manifest`: Manifest to sample held-out windows from (required)
- `--rows`: Windows to evaluate (default: 32)
- `--seed`: Window sampling seed (default: 1234)
- `--batch-size`: Evaluation batch size (default: 2)
- `--output`: Write results as JSON

</details>

<details>
<summary><h3>lyricscribe ensemble</h3></summary>

The `ensemble` commands combine and repair transcription jobs. Their outputs are standard job directories, so `evaluate`, `analyze` and further ensembling all work on them.

#### `lyricscribe ensemble rover`

ROVER (Fiscus, 1997): aligns each system's hypothesis into a word confusion network and takes a frequency vote per slot, with ties going to the earlier member. Systems and datasets come from a spec file, ad-hoc flags, or both:

```yaml
root: /path/to/project   # relative paths resolve here; default: the spec's directory
datasets:
  dev: data/dev
  test: {dir: data/test, expected: 100}   # expected: optional song-count check
systems:
  whisper: {dev: jobs/whisper/dev, test: jobs/whisper/test}
  parakeet: {dev: jobs/parakeet/dev, test: jobs/parakeet/test}
  qwen: {dev: jobs/qwen/dev, test: jobs/qwen/test}
subsets:
  trio: [whisper, parakeet, qwen]   # the first member is the alignment pivot
```

```bash
# From a spec
lyricscribe ensemble rover --spec ensemble.yaml --subset trio --dataset test --out ./jobs/rover

# Ad hoc
lyricscribe ensemble rover \
    --dataset-dir test=./dataset/test \
    --system whisper=./jobs/whisper/test \
    --system qwen=./jobs/qwen/test \
    --subset whisper,qwen \
    --dataset test \
    --out ./jobs/rover
```

Output is written to `<out>/<tag>/vote/<dataset>`.

Options:

- `--subset`: Subset name from the spec, or comma-separated members in alignment order; repeatable (required)
- `--dataset`: Dataset name; repeatable (required)
- `--out`: Output root (required)
- `--spec`: Ensemble spec file (YAML or JSON)
- `--system`: Ad-hoc system as `name=job_dir` (single `--dataset`) or `name:dataset=job_dir`; repeatable
- `--dataset-dir`: Ad-hoc dataset as `name=dataset_dir`; repeatable
- `--orderings`: Vote under N member orderings (the given one plus N-1 seeded shuffles) and re-vote the results; 1 is plain ROVER (default: 1)
- `--seed`: Seed for the `--orderings` shuffles (default: 0)
- `--null-weight`: Weight of a NULL vote; 1.0 is plain ROVER (default: 1.0)
- `--oracle` / `--no-oracle`: Also report the oracle WER, the best any slot-wise selection could reach (default: on)
- `--tag`: Output directory name (default: the subset name, plus `_ORDER<N>` with `--orderings`)
- `--workers`: Alignment processes (default: 16)

#### `lyricscribe ensemble deloop`

Collapses runaway repetition, where a short cycle of words is repeated far more often than a real chorus would be, and writes a new job directory. Meant for output decoded without `--no-repeat-ngram-size`.

```bash
lyricscribe ensemble deloop ./jobs/qwen3omni/jam_alt --out ./jobs/qwen3omni_deloop/jam_alt --max-run 5
```

Options:

- `SRC`: Source job directory (positional, required)
- `--out`: Destination job directory (required)
- `--max-run`: Repeats kept of a looping cycle (default: 5)
- `--max-cycle`: Longest cycle length, in words (default: 25)

</details>

<details>
<summary><h3>lyricscribe analyze</h3></summary>

The `analyze` commands score and compare transcription jobs using the same normalisation as `lyricscribe evaluate`. Jobs with missing songs or corrupt result lines are refused unless explicitly allowed.

#### `lyricscribe analyze score`

Scores one or more job directories: WER with insertion/deletion/substitution counts and shares.

```bash
lyricscribe analyze score ./jobs/whisper/jam_alt ./jobs/qwen/jam_alt --by-language
```

Options:

- `JOB_DIRS`: Job directories to score (positional, required)
- `--selfwer-file`: Self-WER artifact from `analyze selfwer`; adds a slice restricted to songs with trustworthy references
- `--max-self-wer`: Inclusive self-WER cutoff for that slice (default: 0.3)
- `--common`: Score only songs present in every job (for comparing partial runs)
- `--by-language`: Per-language breakdown
- `--raw`: Also report WER on un-normalised text
- `--allow-partial`: Score jobs with missing songs or corrupt result lines
- `--out`: Write results as JSON

#### `lyricscribe analyze score-all`

Scores every complete job under a root and prints a ranking; partial or corrupt jobs are listed and skipped.

```bash
lyricscribe analyze score-all ./jobs --pattern "*/jam_alt" --top 20
```

Options:

- `ROOT`: Directory to search (positional, required)
- `--pattern`: Glob for job directories relative to `ROOT` (default: `**`)
- `--selfwer-file` / `--max-self-wer`: As for `score`; ranks by the filtered slice when given
- `--top`: Ranked rows to print (default: 20)
- `--out`: Write results as JSON

#### `lyricscribe analyze selfwer`

Each song's `lyrics.json` carries two independent reference transcripts (`unsynced`, which WER is graded against, and `synced`). This writes their mutual WER per song. A high self-WER flags a broken reference, such as the wrong song version or missing sections.

```bash
lyricscribe analyze selfwer ./dataset/jam_alt --out ./artifacts/jam_alt_selfwer.jsonl
```

Options:

- `DATASET_DIR`: Dataset directory (positional, required)
- `--out`: Destination JSONL file (required)

#### `lyricscribe analyze bootstrap`

Paired song-level bootstrap confidence interval on WER(A) − WER(B). Each resample recomputes WER from summed errors, and pairing on songs cancels song difficulty.

```bash
lyricscribe analyze bootstrap ./jobs/whisper/jam_alt ./jobs/qwen/jam_alt
```

Options:

- `JOB_A`, `JOB_B`: The two job directories (positional, required)
- `--n-resamples`: Bootstrap resamples (default: 2000)
- `--seed`: Resampling seed (default: 0)
- `--selfwer-file` / `--max-self-wer`: Restrict to songs with trustworthy references

#### `lyricscribe analyze merge`

Averages the weights of finetunes of the same base model into one model ("model soup"). Works on any HF safetensors checkpoints that share parameter names; keys missing from some members are dropped and reported.

```bash
lyricscribe analyze merge \
    --member ./experiments/run_a/checkpoints/epoch_3 \
    --member ./experiments/run_b/checkpoints/epoch_3 \
    --out ./models/soup
```

Options:

- `--member`: Checkpoint directory; repeatable. Config and tokenizer come from the first. (required)
- `--out`: Output model directory (required)
- `--weights`: Comma-separated weights (default: uniform)
- `--dtype`: Saved dtype (default: `float16`)

</details>
