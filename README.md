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

Currently, the LyricScribe CLI only contains one subcommand, `lyricscribe separate`. This allows for mass vocal separation of files using [demucs-next](https://github.com/ryan5453/demucs-next), designed for usage on SLURM clusters.

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
