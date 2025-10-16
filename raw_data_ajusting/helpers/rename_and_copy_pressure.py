#!/usr/bin/env python3
"""
Scan raw_data_camp/pressure recursively, take CSV files and copy them into
raw_data_ajusted/ with filenames constructed from the relative path components
joined by underscores + original filename.

Example:
  raw_data_camp/pressure/E/session1/1758356608256_-1_max-pre.csv ->
  raw_data_ajusted/E_session1_1758356608256_-1_max-pre.csv

Files are written in lexicographic order of their source paths.
"""
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
# There are two observed layouts in this repo:
# 1) repo_root/raw_data_camp/pressure
# 2) repo_root/raw_data_ajusting/raw_data_camp/pressure
# Try both and pick the one that exists.
candidate1 = ROOT / 'raw_data_camp' / 'pressure'
candidate2 = ROOT / 'raw_data_ajusting' / 'raw_data_camp' / 'pressure'
if candidate1.exists():
    SRC = candidate1
elif candidate2.exists():
    SRC = candidate2
else:
    SRC = candidate1  # fallback; will result in empty list and an error message later

# Destination: place under raw_data_ajusting/raw_data_ajusted (matches created dir)
DST = ROOT / 'raw_data_ajusting' / 'raw_data_ajusted'


def collect_csvs(src: Path):
    return sorted([p for p in src.rglob('*.csv') if p.is_file()])


def make_target_name(src: Path, src_root: Path):
    # relative path components from src_root, excluding the src_root itself
    rel = src.relative_to(src_root)
    parts = list(rel.parts)
    # parts example: ['E', 'session1', '1758356608256_-1_max-pre.csv']
    # We want: E_session1_1758356608256_-1_max-pre.csv
    if len(parts) == 1:
        return parts[0]
    else:
        # join all but last for path components, then append last filename
        return '_'.join(parts)


def ensure_dst(dst: Path):
    dst.mkdir(parents=True, exist_ok=True)


def main():
    print('SRC', SRC)
    print('DST', DST)
    if not SRC.exists():
        print(f'ERROR: source directory does not exist: {SRC}')
        return
    ensure_dst(DST)
    csvs = collect_csvs(SRC)
    print(f'Found {len(csvs)} CSV files')
    for i, src in enumerate(csvs, start=1):
        new_name = make_target_name(src, SRC)
        dst_path = DST / new_name
        # avoid overwriting: if exists, append a counter
        if dst_path.exists():
            base = dst_path.stem
            suffix = dst_path.suffix
            k = 1
            while True:
                candidate = DST / f"{base}_{k}{suffix}"
                if not candidate.exists():
                    dst_path = candidate
                    break
                k += 1
        shutil.copy2(src, dst_path)
        if i % 50 == 0:
            print(f'Copied {i}/{len(csvs)}')
    print('Done')


if __name__ == '__main__':
    main()
