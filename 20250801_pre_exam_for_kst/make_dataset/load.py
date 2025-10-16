import pandas as pd
import numpy as np
import os
import sys
import glob
import re
from pathlib import Path
from collections import defaultdict

LABEL_PATH = "label.csv"

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
    """ファイル名から人物ID、セッション、ラベルIDを抽出
    例: 20250801_1235_watanabe_1kaime_1.csv
    """
    basename = os.path.basename(filename)
    
    # 正規表現でパース
    pattern = r'\d+_\d+_([^_]+)_(\d+)kaime_(\d+)\.csv'
    match = re.match(pattern, basename)
    
    if match:
        person_name = match.group(1)
        session = int(match.group(2))
        label_id = int(match.group(3))
        
        # 被験者名から数字IDを取得
        person_id = SUBJECT_NAME_TO_ID.get(person_name.lower(), None)
        if person_id is None:
            print(f"Warning: Unknown person name {person_name}")
            return None, None, None
            
        return person_id, session, label_id
    else:
        print(f"Warning: Could not parse filename {basename}")
        return None, None, None

def load_csv_data(file_path):
    """CSVファイルを読み込んでnumpy arrayに変換（Channel_2のみ使用）"""
    try:
        df = pd.read_csv(file_path)
        time_array = df['ElapsedTime(s)'].values
        value_array = df['Channel_2'].values
        return time_array, value_array
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

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
        person_id, session, label_id = parse_filename(file_path)
        
        if person_id is None:
            continue
            
        # CSVデータを読み込み
        time_array, value_array = load_csv_data(file_path)
        
        if time_array is None or value_array is None:
            continue
        
        # ラベル名を取得
        label_name = labels.get(label_id, f"Unknown_{label_id}")
        
        # データ情報を作成
        data_entry = {
            "session": session,
            "label_id": label_id,
            "label_name": label_name,
            "time": time_array,
            "value": value_array,
            "filename": os.path.basename(file_path)
        }
        
        person_data[person_id].append(data_entry)
    
    # 結果をリスト形式に変換
    result = []
    for person_id, data_list in person_data.items():
        # セッションとラベルIDでソート
        data_list.sort(key=lambda x: (x["session"], x["label_id"]))
        
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