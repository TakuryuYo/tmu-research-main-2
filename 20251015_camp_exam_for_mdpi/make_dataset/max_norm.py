import numpy as np
import load
import sys
import matplotlib.pyplot as plt
import copy
from collections import defaultdict

def _reference_priority(entry):
    """正規化に使う参照データの優先順位を計算"""
    name = (entry.get("label_name") or "").lower()
    label_id = entry.get("label_id")
    
    if 'max-pre' in name or 'max_pre' in name:
        rank = 0
    elif 'max' in name and 'speed' not in name:
        rank = 1
    elif isinstance(label_id, int) and label_id < 0:
        rank = 2
    else:
        rank = 3
    
    # 大きい振幅を優先
    value = entry.get("value")
    amplitude = float(np.max(np.abs(value))) if value is not None and len(value) > 0 else 0.0
    return (rank, -amplitude)

def max_norm(dir):
    """
    指定されたディレクトリからデータを読み込み、各セッションの最大値で正規化する
    
    Args:
        dir (str): データディレクトリのパス
        
    Returns:
        list: load.pyと同じ形式で、valueが正規化されたデータ
    """
    datas = load.load(dir)
    
    # 結果を格納するリスト
    normalized_datas = []
    
    for person in datas:
        # 人物ごとの正規化されたデータを格納
        normalized_person_data = []
        
        # セッションごとにグループ化
        sessions = defaultdict(list)
        for dat in person["data"]:
            session_id = int(dat["session"])
            sessions[session_id].append(dat)
        
        for session_id in sorted(sessions.keys()):
            exam = sessions[session_id]
            if not exam:
                continue
            
            # 参照データを選択
            ref_entry = min(exam, key=_reference_priority)
            ref_value = ref_entry.get("value")
            if ref_value is None or len(ref_value) == 0:
                print(f"Warning: Reference data empty for person={person['person']} session={session_id}")
                continue
            
            max_value = float(np.max(np.abs(ref_value)))
            if max_value == 0:
                print(f"Warning: Reference data max is 0 for person={person['person']} session={session_id}; skipping normalization")
                continue
            
            print(f"{person['person']} Session {session_id}: Max value = {max_value}")
            
            # ラベルIDで安定ソート
            exam_sorted = sorted(exam, key=lambda x: (x.get("label_id"), x.get("label_name"), x.get("filename")))
            
            for data in exam_sorted:
                normalized_data = copy.deepcopy(data)
                normalized_data["value"] = data["value"] / max_value
                normalized_person_data.append(normalized_data)
        
        # セッションとラベルIDでソート（元のload.pyと同じ順序を保持）
        normalized_person_data.sort(key=lambda x: (x["session"], x["label_id"], x.get("label_name"), x.get("filename")))
        
        # 人物データを結果に追加
        normalized_datas.append({
            "person": person["person"],
            "data": normalized_person_data
        })
    
    # 人物名でソート
    normalized_datas.sort(key=lambda x: x["person"])
    
    print(f"\nMax normalization completed for {len(normalized_datas)} persons")
    for person_info in normalized_datas:
        print(f"  {person_info['person']}: {len(person_info['data'])} files normalized")
    
    return normalized_datas

# スクリプトとして実行された場合のテスト
if __name__ == "__main__":
    if len(sys.argv) > 1:
        normalized_data = max_norm(sys.argv[1])
        
        # 正規化結果の確認
        for person_info in normalized_data:
            print(f"\n=== {person_info['person']} (Normalized) ===")
            for entry in person_info['data'][:3]:  # 最初の3件のみ表示
                print(f"Session {entry['session']}, Label {entry['label_id']} ({entry['label_name']})")
                print(f"  Time shape: {entry['time'].shape}, Value shape: {entry['value'].shape}")
                print(f"  Value range: {entry['value'].min():.6f} - {entry['value'].max():.6f}")
                print(f"  File: {entry['filename']}")
    else:
        print("Usage: python max_norm.py <directory_path>")
