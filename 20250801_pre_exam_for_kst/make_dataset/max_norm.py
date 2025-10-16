import numpy as np
import load
import sys
import matplotlib.pyplot as plt
import copy

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
        
        # セッションごとにグループ化（3回分）
        sessions = [[], [], []] 
        for dat in person["data"]:
            session_id = int(dat["session"]) - 1
            sessions[session_id].append(dat)
        
        for session_idx, exam in enumerate(sessions):
            if len(exam) == 0:
                continue
                
            # 最大値を取得（最初のデータ（Max）から）
            max_data = exam[0]  # 最初のデータはMaxラベル
            max_value = np.max(max_data["value"])
            
            print(f"{person['person']} Session {session_idx + 1}: Max value = {max_value}")
            
            # 各データを正規化
            for data in exam:
                # 元のデータをコピーして変更
                normalized_data = copy.deepcopy(data)
                
                # valueを最大値で正規化
                normalized_data["value"] = data["value"] / float(max_value)
                
                # 正規化されたデータを追加
                normalized_person_data.append(normalized_data)
        
        # セッションとラベルIDでソート（元のload.pyと同じ順序を保持）
        normalized_person_data.sort(key=lambda x: (x["session"], x["label_id"]))
        
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
