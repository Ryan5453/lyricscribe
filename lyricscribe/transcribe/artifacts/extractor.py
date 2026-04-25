import json
import logging
from pathlib import Path
import torchaudio
import numpy as np



logger = logging.getLogger(__name__)

# the number of frequency bins (most efficient on powers of 2)
n_ftt = 512
sample_rate = 16000
hop_length = 150

def _load_audio(path : Path) -> np.ndarray:
    """Load a .wav file as a 1D float32 array, convert to mono, and resample if needed.

    :param path: the absolute path to the .wav file
    :returns the wav file as a 1d tensor with float32 values
    """
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav.squeeze(0).numpy().astype(np.float32)



def _compute_mag_spectrogram(audio: np.ndarray) -> np.ndarray:
    """
    Compute the magnitude spectrogram of a audio file

    The audio is divided into overlapping windows of 400 samples (25ms at 16kHz). Each frame is multiplied by a hanning 
    window to reduce spectral leakage, then transformed into frequency using FFT.

    :param audio: 1D array of audio samples at 16kHz in float32.
    :returns: 2D array containing the magnitude of each frequency bin for each frame.
    """

    # 25 ms windows, standard for ASR
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


def extract_artifact_features(song_dir: Path) -> dict:
    """
    Extract per-frame artifact features for one song in the MUSDB dataset.

    :param song_dir: absolute path to the directory containing vocals.wav and htdemucs_ft_vocals.wav
    :returns dictionary of per-frame features
    """

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
    # trim to prevent crash from rounding error from Demucs
    stems = stems[:min_len]
    separated = separated[:min_len]

    artifacts = separated - stems

    artifacts_mag = _compute_mag_spectrogram(artifacts)
    stems_mag = _compute_mag_spectrogram(stems)

    n_frames, _ = artifacts_mag.shape
 
    freqs = np.fft.rfftfreq(n_ftt, d=1.0 / sample_rate)

    artifacts_rms = np.sqrt(np.mean(artifacts_mag **2, axis=1))

    vocal_rms = np.sqrt(np.mean(stems_mag **2, axis=1))

    # should theoretically add a really small number to prevent divide by zero, but we shouldn't run into that problem practically
    asr = artifacts_rms /(vocal_rms + 1e-8) 


    total_energy = artifacts_mag.sum(axis=1, keepdims=True)
      # weighted average of frequency (bin value * magnitude)
    spectral_centroid = (artifacts_mag * freqs[None, :]).sum(axis=1) / total_energy.squeeze()

    # the formula for spectral flatness (geometric mean / arithmetic mean)
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


def process_dataset(musdb_dir: Path, output_dir: Path) -> None:
    """
        Set up the dataset of the frequency of artifacts for running Montreal Force Alignment.
        
        :param musdb_dir: absolute path to the directory containing vocals.wav from the MUSDB-18 dataset
        :param output_dir: absolute path to the direction to save the dataset
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    song_dirs = sorted([d for d in musdb_dir.iterdir() if d.is_dir()])
    logger.info(f"Found {len(song_dirs)} songs in {musdb_dir}")

    success= skipped= failed = 0

    for song_dir in song_dirs:
        output_path = output_dir / f"{song_dir.name}.json"

        if output_path.exists():
            logger.info(f"Skipping {song_dir.name}, it already exists")
            skipped += 1
            continue

        try:
            features = extract_artifact_features(song_dir)
            output_path.write_text(json.dumps(features, indent=2))
            success += 1
        except Exception as e:
            logger.error(f"Failed on {song_dir.name}: {e}")
            failed += 1

    logger.info(f"Done: {success} success, {skipped} skipped, {failed} failed")









    


