import csv
import hashlib
from pathlib import Path
from typing import Union

MANIFEST_COLUMNS = [
    "file",
    "source",
    "publisher",
    "url_or_doi",
    "query",
    "fetch_time_utc",
    "sha256",
    "licence",
]

def calculate_sha256(file_path: Union[str, Path]) -> str:
    """Computes the SHA-256 checksum for a file."""
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"Target file for SHA-256 not found: {p}")
    sha256_hash = hashlib.sha256()
    with open(p, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def write_manifest_row(
    file_path: Union[str, Path],
    source: str,
    publisher: str,
    url_or_doi: str,
    query: str,
    fetch_time_utc: str,
    licence: str,
    manifest_path: Union[str, Path] = Path("data/manifest.csv"),
) -> str:
    """
    Appends an audit row into manifest.csv with exact SHA-256 checksum.
    Creates manifest.csv with canonical header if not already existing.
    """
    target_path = Path(file_path)
    manifest_file = Path(manifest_path)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)

    sha256_val = calculate_sha256(target_path)
    try:
        target_path = target_path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        pass  # outside the repo: keep the path as given
    file_exists = manifest_file.exists() and manifest_file.stat().st_size > 0

    with open(manifest_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(MANIFEST_COLUMNS)
        writer.writerow([
            target_path.as_posix(),
            source,
            publisher,
            url_or_doi,
            query,
            fetch_time_utc,
            sha256_val,
            licence,
        ])

    return sha256_val
