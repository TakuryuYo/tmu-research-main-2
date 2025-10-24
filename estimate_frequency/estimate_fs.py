import csv
import pickle
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Tuple


# 生データ 1 本ごとのサンプリング情報を保持する
@dataclass
class FrequencyStats:
    filename: str
    sample_count: int
    elapsed_sec: float
    sample_time: float
    frequency: float


# アノテーション単位の立ち上がり時間情報を保持する
@dataclass
class RiseTimeStats:
    sample_index: str
    filename: str
    duration_index: int
    frequency: float
    rise_time: float


def _read_timestamps(csv_path: Path) -> List[float]:
    """
    raw CSV から unixtime(ms) を読み出し、1 行につき 1 つの時刻として収集する。
    """
    timestamps: List[float] = []
    with csv_path.open() as f:
        reader = csv.reader(f)
        next(reader, None)  # metadata header
        next(reader, None)  # metadata values
        header = [h.strip() for h in next(reader, [])]
        if "unixtime(ms)" not in header:
            return timestamps
        time_idx = header.index("unixtime(ms)")

        for row in reader:
            if not row:
                continue
            value = row[time_idx].strip()
            if not value:
                continue
            try:
                timestamps.append(float(value))
            except ValueError:
                continue
    return timestamps


def estimate_frequency(csv_path: Path) -> Optional[FrequencyStats]:
    """
    1 本の raw CSV について、(最終時刻 - 初期時刻) / サンプル数 をもとに
    平均サンプリング周波数を推定する。
    """
    timestamps = _read_timestamps(csv_path)
    if len(timestamps) < 2:
        return None

    first, last = timestamps[0], timestamps[-1]
    elapsed_sec = (last - first) / 1000.0
    sample_count = len(timestamps)
    if elapsed_sec <= 0 or sample_count <= 1:
        return None

    sample_time = elapsed_sec / sample_count
    if sample_time <= 0:
        return None
    frequency = 1.0 / sample_time
    return FrequencyStats(
        filename=csv_path.name,
        sample_count=sample_count,
        elapsed_sec=elapsed_sec,
        sample_time=sample_time,
        frequency=frequency,
    )


def load_annotation_mapping(
    dataset_path: Path,
) -> Dict[Tuple[str, int, str], str]:
    """
    aligned_dataset/dataset.bin から (person, session, pre/post) → 元ファイル名 を引き当てる。
    """
    with dataset_path.open("rb") as f:
        dataset = pickle.load(f)

    mapping: Dict[Tuple[str, int, str], str] = {}
    for entry in dataset["dataset_info"]["data"]:
        if entry.get("label_id") != -2:
            continue
        filename = entry.get("filename", "")
        lower = filename.lower()
        if "max-speed" not in lower:
            continue
        if "pre" in lower:
            prepos = "pre"
        elif "post" in lower:
            prepos = "post"
        else:
            prepos = ""
        key = (str(entry.get("person")), int(entry.get("session")), prepos)
        mapping[key] = filename
    return mapping


def compute_rise_times(
    annotation_csv: Path, frequency_map: Dict[str, float], mapping: Dict[Tuple[str, int, str], str]
) -> List[RiseTimeStats]:
    """
    アノテーション CSV の start_pos/max_pos を用いて duration_index を求め、
    対応する raw ファイルの周波数から立ち上がり時間を計算する。
    """
    results: List[RiseTimeStats] = []
    with annotation_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                start_pos = int(row["start_pos"])
                max_pos = int(row["max_pos"])
            except (TypeError, ValueError):
                continue
            duration_index = max_pos - start_pos
            if duration_index <= 0:
                continue
            key = (row["person_id"], int(row["session"]), row["prepos"])
            filename = mapping.get(key)
            if not filename:
                continue
            freq = frequency_map.get(filename)
            if not freq:
                continue
            rise_time = duration_index / freq
            results.append(
                RiseTimeStats(
                    sample_index=row["sample_index"],
                    filename=filename,
                    duration_index=duration_index,
                    frequency=freq,
                    rise_time=rise_time,
                )
            )
    return results


def main() -> None:
    """
    raw ディレクトリ内の全 CSV を処理して平均サンプリング周波数を求め、
    アノテーション情報と組み合わせて立ち上がり時間の平均を表示する。
    """
    repo_root = Path("20251015_camp_exam_for_mdpi")
    raw_dir = repo_root / "raw_data"
    annotation_csv = (
        repo_root
        / "annotation_max-speed_results"
        / "annotation_multi_persons_11_sessions_1-2-3-4-5_Unknown_-2_id00_108samples_20251021_162914.csv"
    )
    dataset_path = repo_root / "aligned_dataset" / "dataset.bin"

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    frequency_stats: List[FrequencyStats] = []
    for csv_file in sorted(raw_dir.glob("*.csv")):
        stats = estimate_frequency(csv_file)
        if stats:
            frequency_stats.append(stats)

    if not frequency_stats:
        print("No frequencies could be estimated.")
        return

    freq_values = [stat.frequency for stat in frequency_stats]
    freq_map = {stat.filename: stat.frequency for stat in frequency_stats}
    avg_freq = sum(freq_values) / len(freq_values)
    median_freq = median(freq_values)

    mapping = load_annotation_mapping(dataset_path)
    rise_stats = compute_rise_times(annotation_csv, freq_map, mapping)
    if rise_stats:
        rise_values = [stat.rise_time for stat in rise_stats]
        avg_rise = sum(rise_values) / len(rise_values)
        median_rise = median(rise_values)
    else:
        avg_rise = 0.0
        median_rise = 0.0

    print(f"Processed raw files: {len(frequency_stats)}")
    print(f"Average sampling frequency: {avg_freq:.6f} Hz")
    print(f"Median sampling frequency:  {median_freq:.6f} Hz")
    print(f"Processed annotations: {len(rise_stats)}")
    print(f"Average rise time (duration_index / frequency): {avg_rise:.6f}")
    print(f"Median rise time:                         {median_rise:.6f}")


if __name__ == "__main__":
    main()
