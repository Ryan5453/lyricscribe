

import json 
import logging
from pathlib import Path
import shutil
import re


logger = logging.getLogger(__name__)


def prepare_mfa_inputs(musdb_dir: Path, prep_dir: Path) -> None:
    prep_dir.mkdir(parents=True, exist_ok=True)

    song_dirs = sorted([d for d in musdb_dir.iterdir() if d.is_dir()])
    logger.info(f"Preparing Montreal Force Aligner inputs for {len(song_dirs)} songs")

    success = skipped = failed = 0

    for song_dir in song_dirs:
        vocals_path  = song_dir/ "vocals.wav"
        lyrics_path = song_dir / "lyrics.json"

        if not vocals_path.exists() or not lyrics_path.exists():
            logger.warning(f"Missing paths")
            failed += 1
            continue

        name = song_dir.name
        out_wav = prep_dir / f"{name}.wav"
        out_lab = prep_dir / f"{name}.lab"

        if out_wav.exists() and out_lab.exists():
            skipped += 1
            continue

        shutil.copy2(vocals_path, out_wav)

        with open(lyrics_path) as f:
            lyrics_data = json.load(f)

        text = lyrics_data.get("unsynced", {}).get("data", "")

        if not text:
            failed += 1
            continue

        text = text.lower()
        text = re.sub(r"[^\w\s']", " ", text)
        text = re.sub(r"\s+", " ", text).strip()


        out_lab.write_text(text)
        success += 1

    logger.info(f"Done: {success} prepared, {skipped} skipped, {failed} failed")
    logger.info(f"\nNow run MFA:")
    logger.info(f"  mfa align {prep_dir} english_us_arpa english_us_arpa <output-dir> --clean")




def _parse_textgrid(textgrid_path: Path) -> list[dict]:
    text = textgrid_path.read_text(encoding="utf-8")

    words = []
    in_words_tier = False
    current_xmin = None
    current_xmax = None

    for line in text.splitlines():
        line = line.strip()

        if "name = words" in line:
            in_words_tier = True
            continue

        if in_words_tier and line.startswith("name = ") and '"words"' not in line:
            break

        if not in_words_tier:
            continue
        
        if line.startswith("xmin ="):
            current_xmin = float(line.split("=")[1].strip())
        elif line.startswith("xmax ="):
            current_xmax = float(line.split("=")[1].strip())
        elif line.startswith("text ="):
            match = re.search(r'text = "(.*?)"', line)
            if match:
                word = match.group(1).strip()
                if word and word not in ("", "sp", "sil", "<eps>"):
                    words.append({
                        "word": word,
                        "start": current_xmin,
                        "end": current_xmax,
                    })
                current_xmin = None
                current_xmax = None

    return words

def parse_textgrid(textgrid_dir : Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    textgrid_files = sorted(textgrid_dir.glob("**/*.TextGrid"))

    success = failed = 0

    for tg_path in textgrid_files:
        song_id = tg_path.stem
        output_path = output_dir / f"{song_id}.json"

        try:
            words = _parse_textgrid(tg_path)
            output_path.write_text(json.dumps({
                "song_id": song_id,
                "words": words,
            }, indent=2))
            success += 1

        except Exception as e:
            failed += 1

    logger.info(f"Done: {success} parsed, {failed} failed")




































