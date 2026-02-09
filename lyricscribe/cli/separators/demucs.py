"""
Demucs separator for mass audio separation with job configuration in SQLite.
"""

import random
import sqlite3
import time
from pathlib import Path
from typing import Optional

import torch
from demucs.api import Separator


class DemucsSeparator:
    """
    Demucs separator with job configuration stored in SQLite database.
    """

    def __init__(self):
        self.model = None
        self.model_name = None
        self.device = None
        self.stem = None

    def _load_model(self) -> None:
        """
        Lazy-load the model.
        """
        if self.model is not None:
            return

        if self.stem:
            print(f"Loading Demucs: {self.model_name} on {self.device} (isolating {self.stem})")
            self.model = Separator(
                model=self.model_name,
                device=self.device,
                isolate_stem=self.stem
            )
        else:
            print(f"Loading Demucs: {self.model_name} on {self.device} (all stems)")
            self.model = Separator(
                model=self.model_name,
                device=self.device
            )

    def setup_job(
        self,
        directory: Path,
        db_path: Path,
        model: str,
        device: str,
        num_chunks: int,
        stem: Optional[str],
    ) -> None:
        """
        Initialize separation job by registering files and storing config.
        """
        all_isrcs = [d for d in directory.iterdir() if d.is_dir()]
        total = len(all_isrcs)

        if total == 0:
            print("No ISRC folders found")
            return

        random.seed(42)
        random.shuffle(all_isrcs)

        db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS job_config (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)

            config = {
                'directory': str(directory),
                'model': model,
                'device': device,
                'stem': stem,
                'num_chunks': num_chunks,
                'total_files': total,
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            }

            for key, value in config.items():
                conn.execute(
                    "INSERT OR REPLACE INTO job_config (key, value) VALUES (?, ?)",
                    (key, str(value))
                )

            conn.execute("""
                CREATE TABLE IF NOT EXISTS separations (
                    isrc TEXT PRIMARY KEY,
                    input_path TEXT NOT NULL,
                    output_path TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    chunk_id INTEGER NOT NULL,
                    duration_seconds REAL,
                    error_message TEXT,
                    processed_at TIMESTAMP
                )
            """)

            conn.execute("DELETE FROM separations")

            chunk_size = total // num_chunks

            for chunk_id in range(1, num_chunks + 1):
                start = (chunk_id - 1) * chunk_size
                end = total if chunk_id == num_chunks else chunk_id * chunk_size

                for isrc_path in all_isrcs[start:end]:
                    isrc = isrc_path.name
                    if stem:
                        output_path = isrc_path / f"{model}_{stem}.wav"
                    else:
                        output_path = isrc_path / f"{model}_stems"

                    conn.execute(
                        "INSERT INTO separations (isrc, input_path, output_path, chunk_id) VALUES (?, ?, ?, ?)",
                        (isrc, str(isrc_path), str(output_path), chunk_id)
                    )

        print(f"Registered {total} songs into {num_chunks} chunks: {db_path}")

    def _load_config(self, db_path: Path) -> dict:
        """
        Load job configuration from database.
        """
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute("SELECT key, value FROM job_config")
            return {row[0]: row[1] for row in cursor.fetchall()}

    def process_chunk(self, db_path: Path, chunk_id: int) -> None:
        """
        Process one chunk of the separation job.
        """
        config = self._load_config(db_path)

        self.model_name = config['model']
        self.device = config['device']
        self.stem = config['stem']
        directory = Path(config['directory'])

        self._load_model()

        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(
                "SELECT isrc, input_path, output_path FROM separations WHERE chunk_id = ? AND status = 'pending'",
                (chunk_id,)
            )
            pending = cursor.fetchall()

        if not pending:
            print(f"No pending songs in chunk {chunk_id}")
            return

        total = len(pending)
        print(f"Processing chunk {chunk_id}: {total} songs")

        success_count = 0
        failed_count = 0
        skipped_count = 0

        for i, (isrc, input_path, output_path) in enumerate(pending, 1):
            if self.stem:
                # Single stem mode
                if Path(output_path).exists():
                    skipped_count += 1
                    self._update_status(db_path, isrc, 'success', 0.0, None)
                    continue
            else:
                # All stems mode - check if output directory exists
                if Path(output_path).exists():
                    skipped_count += 1
                    self._update_status(db_path, isrc, 'success', 0.0, None)
                    continue

            start_time = time.time()

            try:
                sources = self.model.separate(input_path)

                if self.stem:
                    # Export single stem
                    sources.export_stem(self.stem, output_path)
                else:
                    # Export all stems
                    output_dir = Path(output_path)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    for stem_name in ["vocals", "drums", "bass", "other"]:
                        stem_output = output_dir / f"{self.model_name}_{stem_name}.wav"
                        sources.export_stem(stem_name, str(stem_output))

                duration = time.time() - start_time
                self._update_status(db_path, isrc, 'success', duration, None)
                success_count += 1

                print(f"[{i}/{total}] {isrc} ({duration:.2f}s)")

            except Exception as e:
                duration = time.time() - start_time
                self._update_status(db_path, isrc, 'failed', duration, str(e))
                failed_count += 1

                print(f"[{i}/{total}] {isrc}: {e}")

        print(f"Chunk {chunk_id} complete: {success_count} success, {failed_count} failed, {skipped_count} skipped")

    def _update_status(
        self,
        db_path: Path,
        isrc: str,
        status: str,
        duration: float,
        error: Optional[str]
    ) -> None:
        """
        Update separation status in database.
        """
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """UPDATE separations
                SET status = ?, duration_seconds = ?, error_message = ?, processed_at = CURRENT_TIMESTAMP
                WHERE isrc = ?""",
                (status, duration, error, isrc)
            )

    def show_stats(self, db_path: Path) -> None:
        """
        Display processing statistics from database.
        """
        config = self._load_config(db_path)

        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(
                "SELECT status, COUNT(*), AVG(duration_seconds) FROM separations GROUP BY status"
            )
            stats = cursor.fetchall()

            cursor = conn.execute("SELECT COUNT(*) FROM separations")
            total = cursor.fetchone()[0]

        print(f"\nJob: {db_path}")
        print(f"  Model: {config.get('model')}")
        print(f"  Device: {config.get('device')}")
        print(f"  Stem: {config.get('stem') or 'all'}")
        print(f"  Directory: {config.get('directory')}")

        print(f"\nProgress ({total} total):")

        status_data = {row[0]: {'count': row[1], 'avg': row[2] or 0} for row in stats}

        for status in ['pending', 'processing', 'success', 'failed']:
            if status in status_data:
                count = status_data[status]['count']
                pct = 100 * count / total
                print(f"  {status}: {count} ({pct:.1f}%)")

        success_count = status_data.get('success', {}).get('count', 0)
        completion = 100 * success_count / total if total > 0 else 0
        print(f"\nCompletion: {completion:.1f}%")

    def retry_failed(self, db_path: Path) -> None:
        """
        Retry all failed separations.
        """
        config = self._load_config(db_path)

        self.model_name = config['model']
        self.device = config['device']
        self.stem = config['stem']

        self._load_model()

        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(
                "SELECT isrc, input_path, output_path FROM separations WHERE status = 'failed'"
            )
            failed = cursor.fetchall()

        if not failed:
            print("No failed separations")
            return

        print(f"Retrying {len(failed)} failed separations...")

        success_count = 0
        still_failed = 0

        for isrc, input_path, output_path in failed:
            start_time = time.time()

            try:
                sources = self.model.separate(input_path)

                if self.stem:
                    # Export single stem
                    sources.export_stem(self.stem, output_path)
                else:
                    # Export all stems
                    output_dir = Path(output_path)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    for stem_name in ["vocals", "drums", "bass", "other"]:
                        stem_output = output_dir / f"{self.model_name}_{stem_name}.wav"
                        sources.export_stem(stem_name, str(stem_output))

                duration = time.time() - start_time
                self._update_status(db_path, isrc, 'success', duration, None)
                success_count += 1

                print(f"  {isrc}: success")

            except Exception as e:
                duration = time.time() - start_time
                self._update_status(db_path, isrc, 'failed', duration, str(e))
                still_failed += 1

                print(f"  {isrc}: failed ({e})")

        print(f"Retry complete: {success_count} fixed, {still_failed} still failed")
