import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pickle
from pathlib import Path
from reposition import reposition


def create_aligned_dataset(repositioned_data, target_length=None, output_dir='aligned_dataset'):
    """
    位置合わせ済みデータから1次元の振幅データセットを作成
    
    Args:
        repositioned_data: 位置合わせ済みデータ
        target_length: 出力データの長さ（指定しない場合は最長データに合わせる）
        output_dir: 出力ディレクトリ
        
    Returns:
        dict: 作成されたデータセット情報
    """
    print(f"Creating aligned dataset in {output_dir}/...")
    
    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    
    # target_lengthが指定されていない場合、最長データを調べる
    if target_length is None:
        max_length = 0
        for person in repositioned_data:
            for entry in person["data"]:
                max_length = max(max_length, len(entry["value"]))
        target_length = max_length
        print(f"Target length set to maximum data length: {target_length}")
    
    dataset_info = {
        "target_length": target_length,
        "total_samples": 0,
        "samples_by_person": {},
        "samples_by_label": {},
        "data": []
    }
    
    total_samples = sum(len(person["data"]) for person in repositioned_data)
    processed_count = 0
    
    for person in repositioned_data:
        person_id = person["person"]
        print(f"Processing {person_id}...")
        
        person_samples = []
        
        for entry in person["data"]:
            # 元のデータを取得
            original_value = entry["value"]
            
            # 最大値の位置を検出
            peak_position = np.argmax(np.abs(original_value))
            
            # target_lengthの中央に最大値が来るようにシフト量を計算
            target_center = target_length // 2
            shift = target_center - peak_position
            
            # 新しい配列を作成
            aligned_value = np.zeros(target_length)
            
            # データの配置範囲を計算
            start_src = max(0, -shift)
            end_src = min(len(original_value), target_length - shift)
            start_dst = max(0, shift)
            end_dst = min(target_length, shift + len(original_value))
            
            # データをコピー
            if start_src < end_src and start_dst < end_dst:
                copy_length = min(end_src - start_src, end_dst - start_dst)
                aligned_value[start_dst:start_dst + copy_length] = original_value[start_src:start_src + copy_length]
            
            # エッジパディング
            if start_dst > 0:
                # 前方のパディング（最初の値で埋める）
                first_value = original_value[start_src] if start_src < len(original_value) else original_value[0]
                aligned_value[:start_dst] = first_value
            
            if end_dst < target_length:
                # 後方のパディング（最後の値で埋める）
                last_value = original_value[end_src-1] if end_src > 0 else original_value[-1]
                aligned_value[end_dst:] = last_value
            
            # データ情報を作成
            sample_info = {
                "person": person_id,
                "session": entry["session"],
                "label_id": entry["label_id"],
                "label_name": entry["label_name"],
                "filename": entry["filename"],
                "original_length": len(original_value),
                "original_peak_position": peak_position,
                "target_peak_position": target_center,
                "shift": shift,
                "aligned_value": aligned_value
            }
            
            person_samples.append(sample_info)
            dataset_info["data"].append(sample_info)
            
            # 統計情報を更新
            label_name = entry["label_name"]
            if label_name not in dataset_info["samples_by_label"]:
                dataset_info["samples_by_label"][label_name] = 0
            dataset_info["samples_by_label"][label_name] += 1
            
            processed_count += 1
            if processed_count % 50 == 0:
                print(f"  Progress: {processed_count}/{total_samples} samples processed ({100*processed_count/total_samples:.1f}%)")
        
        dataset_info["samples_by_person"][person_id] = len(person_samples)
    
    dataset_info["total_samples"] = len(dataset_info["data"])
    
    # 統計情報を表示
    print(f"\nDataset creation completed!")
    print(f"Total samples: {dataset_info['total_samples']}")
    print(f"Target length: {dataset_info['target_length']}")
    print(f"Samples by person:")
    for person, count in dataset_info["samples_by_person"].items():
        print(f"  {person}: {count}")
    print(f"Samples by label:")
    for label, count in sorted(dataset_info["samples_by_label"].items()):
        print(f"  {label}: {count}")
    
    return dataset_info


def save_dataset_as_pickle(dataset_info, output_dir='aligned_dataset'):
    """
    データセットをpickleファイルとして保存
    
    Args:
        dataset_info: create_aligned_datasetの戻り値
        output_dir: 出力ディレクトリ
    """
    print(f"Saving dataset as pickle to {output_dir}/dataset.bin...")
    
    # データと ラベルを配列に変換
    X = np.array([sample["aligned_value"] for sample in dataset_info["data"]])
    y_label_id = np.array([sample["label_id"] for sample in dataset_info["data"]])
    y_label_name = np.array([sample["label_name"] for sample in dataset_info["data"]])
    person_ids = np.array([sample["person"] for sample in dataset_info["data"]])
    sessions = np.array([sample["session"] for sample in dataset_info["data"]])
    
    # データセット辞書を作成
    dataset = {
        "X": X,
        "y_label_id": y_label_id,
        "y_label_name": y_label_name,
        "person_ids": person_ids,
        "sessions": sessions,
        "dataset_info": dataset_info
    }
    
    # pickleファイルとして保存
    with open(os.path.join(output_dir, "dataset.bin"), "wb") as f:
        pickle.dump(dataset, f)
    
    print(f"Dataset saved as dataset.bin!")
    print(f"  X: {X.shape}")
    print(f"  y_label_id: {y_label_id.shape}")
    print(f"  y_label_name: {y_label_name.shape}")
    print(f"  person_ids: {person_ids.shape}")
    print(f"  sessions: {sessions.shape}")


def plot_all_data(dataset_info, plot_dir='input_datas'):
    """
    全データのプロットを作成して保存
    
    Args:
        dataset_info: create_aligned_datasetの戻り値
        plot_dir: プロット保存ディレクトリ
    """
    print(f"Creating plots for all data in {plot_dir}/...")
    
    # 出力ディレクトリを作成
    os.makedirs(plot_dir, exist_ok=True)
    
    total_samples = len(dataset_info["data"])
    plot_count = 0
    
    for sample in dataset_info["data"]:
        # プロット作成
        plt.figure(figsize=(12, 6))
        
        # 位置合わせ前後のデータをプロット
        plt.subplot(1, 2, 1)
        # 元データの読み込み（repositioned_dataから取得するため、ここでは簡略化）
        plt.plot(sample["aligned_value"])
        plt.title(f'Aligned Data\n{sample["person"]} - {sample["label_name"]}')
        plt.xlabel('Time Sample')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        
        # ピーク位置を示す
        peak_pos = sample["target_peak_position"]
        plt.axvline(x=peak_pos, color='red', linestyle='--', alpha=0.7, label=f'Peak at {peak_pos}')
        plt.legend()
        
        # データ情報を表示
        plt.subplot(1, 2, 2)
        info_text = f"""Data Information:
Person: {sample["person"]}
Label: {sample["label_name"]} (ID: {sample["label_id"]})
Session: {sample["session"]}
Original Length: {sample["original_length"]}
Target Length: {len(sample["aligned_value"])}
Original Peak: {sample["original_peak_position"]}
Target Peak: {sample["target_peak_position"]}
Shift: {sample["shift"]}
Filename: {sample["filename"]}"""
        
        plt.text(0.05, 0.95, info_text, transform=plt.gca().transAxes, 
                verticalalignment='top', fontfamily='monospace', fontsize=9)
        plt.axis('off')
        
        plt.tight_layout()
        
        # ファイル名を生成（日本語文字を安全な文字に置換）
        label_name_safe = (sample['label_name']
                         .replace('（', '_').replace('）', '_').replace('/', '_')
                         .replace('ー', '_').replace(' ', '_'))
        filename = f"{sample['person']}_{label_name_safe}_session{sample['session']}_label{sample['label_id']:02d}.png"
        filepath = os.path.join(plot_dir, filename)
        
        # プロット保存
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        plot_count += 1
        
        # 進捗表示
        if plot_count % 50 == 0:
            print(f"  Progress: {plot_count}/{total_samples} plots saved ({100*plot_count/total_samples:.1f}%)")
    
    print(f"Successfully saved {plot_count} plots to {plot_dir}/")


# メイン処理
if __name__ == "__main__":
    print("=== Creating Aligned 1D Amplitude Dataset ===")
    
    # 位置合わせ済みデータを読み込み
    print("Loading repositioned data...")
    repositioned_data = reposition('./raw_data/')
    
    # 位置合わせデータセットを作成
    dataset_info = create_aligned_dataset(repositioned_data, output_dir='aligned_dataset')
    
    # pickleファイルとして保存
    save_dataset_as_pickle(dataset_info, output_dir='aligned_dataset')
    
    # 全データのプロットを作成
    plot_all_data(dataset_info, plot_dir='input_datas')
    
    print("Dataset creation completed!")