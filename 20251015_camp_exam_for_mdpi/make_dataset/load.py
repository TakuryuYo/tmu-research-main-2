import csv
import pandas as pd
import numpy as np
import os
import sys
import glob
import re
from pathlib import Path
from collections import defaultdict

LABEL_PATH = "label.csv"
DEFAULT_SIGNAL_CHANNEL = "ch3"

# 被験者コード対応表（数字→アルファベット）
SUBJECT_MAPPING = {
    1: 'A',  # 稲村怜於奈
    2: 'B',  # 内海恒希
    3: 'C',  # 河野悠輝
    4: 'D',  # shuto_hatanaka
    5: 'E',  # 渡辺悠生
    6: 'F',  # Kikuchi Tomoo
    7: 'G',  # Yuki Yoshinaga
    8: 'H'   # Okamoto Yugo
}

# 被験者名→番号のマッピング
SUBJECT_NAME_TO_ID = {
    'inamura': 1,
    'utsumi': 2,
    'kawano': 3,
    'hatanaka': 4,
    'watanabe': 5,
    'kikuchi': 6,
    'yoshinaga': 7,
    'okamoto': 8
}

def load_labels(label_path=LABEL_PATH):
    """label.csvを読み込んでIDとオノマトペの対応辞書を作成"""
    if not os.path.exists(label_path):
        return {}
    try:
        df = pd.read_csv(label_path)
        labels = {}
        for _, row in df.iterrows():
            labels[row['id']] = row['onomatopoeia']
        return labels
    except Exception as e:
        print(f"Warning: Could not load labels from {label_path}: {e}")
        return {}

def parse_filename(filename):
    """ファイル名を解析してメタ情報を取得"""
    basename = os.path.basename(filename)
    
    # 新フォーマット: <person>_session<id>_<timestamp>_<label_id>_<label_name>.csv
    new_pattern = r'^(?P<person>[A-Za-z]+)_session(?P<session>\d+)_(?P<timestamp>\d+)_(?P<label_id>-?\d+)(?:_(?P<label_slug>[^.]+))?\.csv$'
    match = re.match(new_pattern, basename)
    if match:
        person = match.group('person')
        session = int(match.group('session'))
        label_id = int(match.group('label_id'))
        label_slug = match.group('label_slug') or ''
        return person, session, label_id, label_slug
    
    # 旧フォーマット: <date>_<time>_<person>_<session>kaime_<label>.csv
    legacy_pattern = r'\d+_\d+_([^_]+)_(\d+)kaime_(\d+)\.csv'
    match = re.match(legacy_pattern, basename)
    if match:
        person_name = match.group(1)
        session = int(match.group(2))
        label_id = int(match.group(3))
        person_id = SUBJECT_NAME_TO_ID.get(person_name.lower())
        if person_id is None:
            print(f"Warning: Unknown person name {person_name}")
            return None, None, None, None
        return person_id, session, label_id, ''
    
    print(f"Warning: Could not parse filename {basename}")
    return None, None, None, None

def _parse_new_csv(file_path, signal_channel=DEFAULT_SIGNAL_CHANNEL):
    """新フォーマットのCSVを読み込む"""
    metadata = {}
    data_rows = []
    
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        
        try:
            meta_header = next(reader)
            meta_values = next(reader)
            data_header = next(reader)
        except StopIteration:
            raise ValueError("CSV does not contain expected header rows")
        
        meta_header = [h.strip() for h in meta_header]
        meta_values = [v.strip() for v in meta_values]
        metadata = {key: value for key, value in zip(meta_header, meta_values)}
        
        data_header = [h.strip() for h in data_header]
        time_index = data_header.index('unixtime(ms)') if 'unixtime(ms)' in data_header else 0
        if signal_channel not in data_header:
            raise ValueError(f"Channel {signal_channel} not found in {file_path}")
        value_index = data_header.index(signal_channel)
        
        for row in reader:
            if not row or all(cell.strip() == '' for cell in row):
                continue
            data_rows.append([cell.strip() for cell in row])
    
    if not data_rows:
        raise ValueError("CSV contains no data rows")
    
    time_values = np.array([float(row[time_index]) for row in data_rows], dtype=np.float64)
    value_values = np.array([float(row[value_index]) for row in data_rows], dtype=np.float64)
    
    # サンプリング周波数を推定
    sampling_rate_hz = None
    nyquist_hz = None
    if len(time_values) > 1:
        diffs_ms = np.diff(time_values)
        valid_diffs = diffs_ms[diffs_ms > 0]
        if valid_diffs.size > 0:
            median_interval_ms = np.median(valid_diffs)
            if median_interval_ms > 0:
                sampling_rate_hz = 1000.0 / median_interval_ms
                nyquist_hz = sampling_rate_hz / 2.0
    
    start_time = time_values[0]
    time_array = (time_values - start_time) / 1000.0  # convert ms to seconds relative to start
    
    # 追加メタ情報を整形
    metadata_processed = {
        "unixtime_ms": int(metadata.get("unixtime", "0") or 0),
        "person_id": metadata.get("personID", ""),
        "label_id": int(metadata.get("onomatopeiaID", "0") or 0),
        "session": int(metadata.get("session", "0") or 0),
        "talk_tone": float(metadata.get("talk_tone", "0") or 0),
        "talk_speed": float(metadata.get("talk_speed", "0") or 0),
        "display": metadata.get("display", ""),
        "signal_channel": signal_channel,
        "sampling_rate_hz": sampling_rate_hz,
        "nyquist_hz": nyquist_hz
    }
    
    return time_array, value_values, metadata_processed

def load_csv_data(file_path, signal_channel=DEFAULT_SIGNAL_CHANNEL):
    """CSVファイルを読み込んでnumpy arrayに変換"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
        if first_line.lower().startswith('unixtime') and 'personid' in first_line.lower():
            return _parse_new_csv(file_path, signal_channel=signal_channel)
        
        df = pd.read_csv(file_path)
        time_array = df['ElapsedTime(s)'].values
        value_array = df['Channel_2'].values
        
        sampling_rate_hz = None
        nyquist_hz = None
        if len(time_array) > 1:
            diffs = np.diff(time_array)
            valid_diffs = diffs[diffs > 0]
            if valid_diffs.size > 0:
                median_interval_sec = np.median(valid_diffs)
                if median_interval_sec > 0:
                    sampling_rate_hz = 1.0 / median_interval_sec
                    nyquist_hz = sampling_rate_hz / 2.0
        
        metadata = {
            "signal_channel": "Channel_2",
            "sampling_rate_hz": sampling_rate_hz,
            "nyquist_hz": nyquist_hz
        }
        return time_array, value_array, metadata
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None, {}

def load(path):
    """指定されたパスからデータを読み込み、人物ごとに整理して返す
    
    Args:
        path (str): CSVファイルが格納されているディレクトリのパス
        
    Returns:
        list: 各人物のデータを含む辞書のリスト
        [
            {
                "person": "Mr.WatanabeYuki",
                "data": [
                    {
                        "session": 3,
                        "label_id": 25,
                        "label_name": "ぴん",
                        "time": numpy_array,
                        "value": numpy_array,
                        "filename": "original_filename.csv"
                    },
                    ...
                ]
            },
            ...
        ]
    """
    
    # ラベル辞書を読み込み
    labels = load_labels()
    
    # CSVファイルを検索
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    
    # label.csvを除外
    csv_files = [f for f in csv_files if not f.endswith(LABEL_PATH)]
    
    print(f"Found {len(csv_files)} CSV files in {path}")
    
    # 人物ごとにデータをグループ化
    person_data = defaultdict(list)
    
    for file_path in csv_files:
        # ファイル名をパース
        person_id, session, label_id, label_slug = parse_filename(file_path)
        
        if person_id is None:
            continue
        
        # CSVデータを読み込み
        time_array, value_array, metadata = load_csv_data(file_path)
        
        if time_array is None or value_array is None:
            continue
        
        # メタ情報優先で上書き
        if metadata:
            person_id = metadata.get("person_id") or person_id
            session = metadata.get("session") or session
            label_id = metadata.get("label_id") if metadata.get("label_id") is not None else label_id
        
        # ラベル名を決定
        label_name = labels.get(label_id)
        if not label_name:
            if label_slug:
                label_name = label_slug
            elif metadata:
                label_name = metadata.get("label_name") or metadata.get("display") or f"Unknown_{label_id}"
            else:
                label_name = f"Unknown_{label_id}"
        
        person_key = str(person_id)
        
        # データ情報を作成
        data_entry = {
            "session": session,
            "label_id": label_id,
            "label_name": label_name,
            "time": time_array,
            "value": value_array,
            "filename": os.path.basename(file_path),
            "label_slug": label_slug
        }
        if metadata:
            data_entry["metadata"] = metadata
        
        person_data[person_key].append(data_entry)
    
    # 結果をリスト形式に変換
    result = []
    for person_id, data_list in person_data.items():
        # セッションとラベルIDでソート
        data_list.sort(key=lambda x: (x["session"], x["label_id"], x["filename"]))
        
        result.append({
            "person": person_id,
            "data": data_list
        })
    
    # 人物名でソート
    result.sort(key=lambda x: x["person"])
    
    print(f"Loaded data for {len(result)} persons")
    for person_info in result:
        print(f"  {person_info['person']}: {len(person_info['data'])} files")
    
    return result

# # スクリプトとして実行された場合のテスト
# if __name__ == "__main__":
#     if len(sys.argv) > 1:
#         data = load(sys.argv[1])
        
#         # データの概要を表示
#         for person_info in data:
#             print(f"\n=== {person_info['person']} ===")
#             for entry in person_info['data'][:3]:  # 最初の3件のみ表示
#                 print(f"Session {entry['session']}, Label {entry['label_id']} ({entry['label_name']})")
#                 print(f"  Time shape: {entry['time'].shape}, Value shape: {entry['value'].shape}")
#                 print(f"  File: {entry['filename']}")
#     else:
#         print("Usage: python load.py <directory_path>")
