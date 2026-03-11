import json
import logging
from pathlib import Path
import torchaudio
import numpy as np



logger = logging.getLogger(__name__)


n_ftt = 512
sample_rate = 16000
hop_length = 150

def _load_audio(path : Path) -> np.ndarray:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav.squeeze(0).numpy().astype(np.float32)



def _compute_mag_spectrogram(audio: np.ndarray) -> np.ndarray:
    window_length = 400

    window = np.hanning(window_length)

    n_frames = (len(audio) - window_length)// hop_length
    magnitude = np.zeros((n_frames, n_ftt//2 + 1), dtype=np.float32)

    for i in range(n_frames):
        start = i * hop_length
        frame = audio[start: start + window_length] * window
        spectrum = np.abs(np.fft.rfft(frame, n=n_ftt))
        magnitude[i] = spectrum
    return magnitude

def _hz_to_bin(hz:float):
    return int(round(hz * n_ftt / sample_rate))


def extract_artifact_features(song_dir: Path) -> dict:
    stems_path = song_dir / "vocals.wav"
    separated_path = song_dir / "htdemucs_ft_vocals.wav"

    if not (stems_path.exists()):
        raise FileNotFoundError("Could not find the clean stems")
    if not (separated_path.exists()):
        raise FileNotFoundError("Could not find separated vocals")
    

    logger.info(f"Loading audio for {song_dir.name}")
    stems = _load_audio(stems_path)
    separated = _load_audio(separated_path)

    min_len = min(len(stems), len(separated))

    artifacts = separated - stems

    artifacts_mag = _compute_mag_spectrogram(artifacts)
    stems_mag = _compute_mag_spectrogram(stems)

    n_frames, n_bins = artifacts.shape()

    freqs = np.fft.rfftfreq(n_ftt, d=1.0 / sample_rate)

    artifacts_rms = np.sqrt(np.mean(artifacts_mag **2, axis=1))

    vocal_rms = np.sqrt(np.mean(stems_mag **2, axis=1))

    # should theoretically add a really small number to prevent divide by zero, but we shouldn't run into that problem practically
    asr = artifacts_rms /(vocal_rms) 

    total_energy = artifacts_mag.sum(axis=1, keepdims=True)
    spectral_centroid = (artifacts_mag * freqs[None, :]).sum(axis=1) / total_energy.squeeze()


    log_mean = np.mean(np.log(artifacts_mag), axis=1)
    mean_log = np.log(np.mean(artifacts_mag, axis=1))
    spectral_flatness = np.exp(log_mean - mean_log)

    return {
        "song_id": song_dir.name,
        "sample_rate": sample_rate,
        "hop_length": hop_length,
        "n_frames": n_frames,
        "artifact_rms":           artifacts_rms.tolist(),
        "vocal_rms":              vocal_rms.tolist(),
        "artifact_to_signal_ratio": asr.tolist(),
        "spectral_centroid":      spectral_centroid.tolist(),
        "spectral_flatness":      spectral_flatness.tolist(),
    }


def process_database(musdb_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    song_dirs = sorted([d for d in musdb_dir.iterdir() if d.is_dir])
    logger.info(f"Found {len(song_dirs)} songs in {musdb_dir}")

    success, skipped, failed = 0

    for song_dir in song_dirs:
        output_path = output_dir / f"{song_dir.name}.json"

        if output_path.exists():
            logger.info(f"Skipping {song_dir.name}, it already exists")
            continue

        try:
            features = extract_artifact_features(song_dir)
            output_path.write_text(json.dump(features, indent=2))
            success += 1
        except Exception as e:
            logger.error(f"Failed on {song_dir.name}: {e}")
            failed += 1

    logger.info(f"Done: {success} success, {skipped} skipped, {failed} failed")









    



