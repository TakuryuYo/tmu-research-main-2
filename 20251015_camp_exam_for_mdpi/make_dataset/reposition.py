import numpy as np
import max_norm
import sys
import copy
import os
from scipy.signal import find_peaks

_ENABLE_MATPLOTLIB_PLOTS = os.environ.get("ENABLE_MATPLOTLIB_PLOTS", "").lower() in ("1", "true", "yes")
if _ENABLE_MATPLOTLIB_PLOTS:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
else:
    plt = None


def _require_matplotlib():
    if plt is None:
        raise RuntimeError(
            "Matplotlib plotting helpers are disabled. "
            "Set ENABLE_MATPLOTLIB_PLOTS=1 before running if matplotlib is available."
        )
    return plt

def find_main_peak(values, window_size=10, height_threshold=0.3):
    """
    データの主要なピーク位置を検出（軽量版）
    
    Args:
        values (numpy.array): 正規化された値配列
        window_size (int): ピーク検出のための局所窓サイズ
        height_threshold (float): ピークの最小高さ
        
    Returns:
        int: 主要ピークの位置（インデックス）
    """
    # 基本的には最大値の位置を返す
    max_position = np.argmax(values)
    
    # 高さの条件をチェック
    if values[max_position] < height_threshold:
        # 閾値以下の場合は、より簡単な方法でピークを探す
        # 移動平均を使って平滑化
        if len(values) > window_size * 2:
            smoothed = np.convolve(values, np.ones(window_size) / window_size, mode='same')
            return np.argmax(smoothed)
    
    return max_position

def align_data_by_peak(data_entry, target_peak_position=None, total_length=None):
    """
    データをピーク位置で揃える
    
    Args:
        data_entry (dict): 個別のデータエントリ
        target_peak_position (int): 目標ピーク位置（Noneの場合は中央に配置）
        total_length (int): 最終的なデータ長（Noneの場合は元の長さを保持）
        
    Returns:
        dict: ピーク位置で揃えられたデータエントリ
    """
    values = data_entry["value"]
    time = data_entry["time"]
    
    # 実際のピーク位置を検出
    actual_peak_position = find_main_peak(values)
    
    # 目標位置が指定されていない場合は中央に設定
    if target_peak_position is None:
        if total_length is None:
            target_peak_position = len(values) // 2
        else:
            target_peak_position = total_length // 2
    
    # 最終的なデータ長を決定
    if total_length is None:
        total_length = len(values)
    
    # シフト量を計算（実際のピークを目標位置に移動）
    shift = target_peak_position - actual_peak_position
    
    # 新しい配列を初期化
    aligned_values = np.zeros(total_length)
    aligned_time = np.linspace(0, total_length-1, total_length)  # インデックスベースの時間軸
    
    # データの配置範囲を計算
    start_src = max(0, -shift)
    end_src = min(len(values), total_length - shift)
    start_dst = max(0, shift)
    end_dst = start_dst + (end_src - start_src)
    
    # データを配置
    if end_src > start_src and end_dst > start_dst:
        aligned_values[start_dst:end_dst] = values[start_src:end_src]
    
    # エッジパディング（前後の値を繰り返し）
    if len(values) > 0:
        # 前方パディング（最初の値で埋める）
        if start_dst > 0:
            first_value = values[start_src] if start_src < len(values) else values[0]
            aligned_values[:start_dst] = first_value
        
        # 後方パディング（最後の値で埋める）
        if end_dst < total_length:
            last_value = values[end_src-1] if end_src > 0 else values[-1]
            aligned_values[end_dst:] = last_value
    
    # 実際にピークが目標位置に配置されたかを確認
    aligned_peak_position = find_main_peak(aligned_values)
    
    # 新しいデータエントリを作成
    aligned_entry = copy.deepcopy(data_entry)
    aligned_entry["value"] = aligned_values
    aligned_entry["time"] = aligned_time
    aligned_entry["original_peak_position"] = actual_peak_position
    aligned_entry["target_peak_position"] = target_peak_position
    aligned_entry["actual_aligned_peak"] = aligned_peak_position  # 実際の配置後ピーク位置
    aligned_entry["shift"] = shift
    aligned_entry["original_length"] = len(values)
    aligned_entry["aligned_length"] = total_length
    aligned_entry["alignment_success"] = abs(aligned_peak_position - target_peak_position) <= 1  # 成功判定
    
    return aligned_entry

def reposition(dir, target_length=None, target_peak_position=None):
    """
    指定されたディレクトリからデータを読み込み、ピーク位置で揃える
    
    Args:
        dir (str): データディレクトリのパス
        target_length (int): 統一するデータ長（Noneの場合は最大長に統一）
        target_peak_position (int): 目標ピーク位置（Noneの場合は中央）
        
    Returns:
        list: load.pyと同じ形式で、ピーク位置で揃えられたデータ
    """
    # 正規化されたデータを読み込み
    normalized_datas = max_norm.max_norm(dir)
    
    # 最大データ長を計算（target_lengthが指定されていない場合）
    if target_length is None:
        max_length = 0
        for person in normalized_datas:
            for entry in person["data"]:
                max_length = max(max_length, len(entry["value"]))
        target_length = max_length
        print(f"Target length set to maximum data length: {target_length}")
    
    # 目標ピーク位置を設定
    if target_peak_position is None:
        target_peak_position = target_length // 2
        print(f"Target peak position set to center: {target_peak_position}")
    
    # 結果を格納するリスト
    repositioned_datas = []
    
    for person in normalized_datas:
        print(f"\nProcessing {person['person']}...")
        
        repositioned_person_data = []
        
        alignment_success_count = 0
        for entry in person["data"]:
            # ピーク位置でデータを揃える
            aligned_entry = align_data_by_peak(
                entry, 
                target_peak_position=target_peak_position,
                total_length=target_length
            )
            
            if aligned_entry["alignment_success"]:
                alignment_success_count += 1
            
            repositioned_person_data.append(aligned_entry)
        
        # 人物データを結果に追加
        repositioned_datas.append({
            "person": person["person"],
            "data": repositioned_person_data
        })
        
        print(f"  {person['person']}: {len(repositioned_person_data)} files repositioned, {alignment_success_count}/{len(repositioned_person_data)} successful alignments")
    
    print(f"\nPeak alignment completed for {len(repositioned_datas)} persons")
    print(f"All data aligned to length {target_length} with peak at position {target_peak_position}")
    
    return repositioned_datas

def plot_alignment_example(repositioned_data, person_idx=0, label_id=1, session=1, save_path=None):
    """
    アライメント結果の例を可視化
    
    Args:
        repositioned_data (list): reposition関数の結果
        person_idx (int): 表示する人物のインデックス
        label_id (int): 表示するラベルID
        session (int): 表示するセッション
        save_path (str): 保存パス（Noneの場合は表示のみ）
    """
    person = repositioned_data[person_idx]
    
    # 指定されたラベル・セッションのデータを検索
    target_entry = None
    for entry in person["data"]:
        if entry["label_id"] == label_id and entry["session"] == session:
            target_entry = entry
            break
    
    if target_entry is None:
        print(f"Data not found: person={person['person']}, label_id={label_id}, session={session}")
        return
    
    plt = _require_matplotlib()
    plt.figure(figsize=(12, 6))
    
    # アライメント結果をプロット
    plt.plot(target_entry["time"], target_entry["value"], 'b-', linewidth=2, label='Aligned data')
    
    # ピーク位置を表示
    peak_pos = target_entry["target_peak_position"]
    plt.axvline(x=target_entry["time"][peak_pos], color='red', linestyle='--', 
                label=f'Peak position (index {peak_pos})')
    
    # 元のピーク位置も表示（参考）
    original_peak = target_entry["original_peak_position"]
    shift = target_entry["shift"]
    
    plt.title(f'Peak Alignment Result\n'
              f'{person["person"]} - {target_entry["label_name"]} (Session {session})\n'
              f'Original peak: {original_peak}, Target peak: {peak_pos}, Shift: {shift}')
    plt.xlabel('Time')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_overlayed_data(repositioned_data, label_id=1, session=1, save_path=None, show_all_sessions=False):
    """
    複数の人物・セッションのデータを重ねてプロット
    
    Args:
        repositioned_data (list): reposition関数の結果
        label_id (int): 表示するラベルID
        session (int): 表示するセッション（show_all_sessions=Trueの場合は無視）
        save_path (str): 保存パス（Noneの場合は表示のみ）
        show_all_sessions (bool): 全セッションを表示するかどうか
    """
    plt = _require_matplotlib()
    plt.figure(figsize=(15, 10))
    
    if show_all_sessions:
        # 全セッションを表示
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        sessions_to_plot = [1, 2, 3]
        
        for sess_idx, sess in enumerate(sessions_to_plot):
            ax = axes[sess_idx]
            colors = plt.cm.tab10(np.linspace(0, 1, len(repositioned_data)))
            
            for person_idx, person in enumerate(repositioned_data):
                # 指定されたラベル・セッションのデータを検索
                target_entry = None
                for entry in person["data"]:
                    if entry["label_id"] == label_id and entry["session"] == sess:
                        target_entry = entry
                        break
                
                if target_entry is not None:
                    # データをプロット
                    ax.plot(target_entry["time"], target_entry["value"], 
                           color=colors[person_idx], linewidth=1.5, alpha=0.8,
                           label=f'{person["person"]}')
                    
                    # ピーク位置をマーク
                    peak_pos = target_entry["target_peak_position"]
                    ax.axvline(x=target_entry["time"][peak_pos], 
                              color=colors[person_idx], linestyle=':', alpha=0.7)
            
            # 共通のピーク位置を表示
            if len(repositioned_data) > 0 and len(repositioned_data[0]["data"]) > 0:
                common_peak_time = repositioned_data[0]["data"][0]["time"][repositioned_data[0]["data"][0]["target_peak_position"]]
                ax.axvline(x=common_peak_time, color='red', linestyle='--', linewidth=2, 
                          label='Target peak position')
            
            ax.set_title(f'Session {sess} - Label {label_id} (All Participants)')
            ax.set_xlabel('Time')
            ax.set_ylabel('Normalized Value')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
    else:
        # 単一セッションを表示
        colors = plt.cm.tab10(np.linspace(0, 1, len(repositioned_data)))
        
        for person_idx, person in enumerate(repositioned_data):
            # 指定されたラベル・セッションのデータを検索
            target_entry = None
            for entry in person["data"]:
                if entry["label_id"] == label_id and entry["session"] == session:
                    target_entry = entry
                    break
            
            if target_entry is not None:
                # データをプロット
                plt.plot(target_entry["time"], target_entry["value"], 
                        color=colors[person_idx], linewidth=2, alpha=0.8,
                        label=f'{person["person"]}')
                
                # ピーク位置をマーク
                peak_pos = target_entry["target_peak_position"]
                plt.axvline(x=target_entry["time"][peak_pos], 
                           color=colors[person_idx], linestyle=':', alpha=0.7)
        
        # 共通のピーク位置を表示
        if len(repositioned_data) > 0 and len(repositioned_data[0]["data"]) > 0:
            common_peak_time = repositioned_data[0]["data"][0]["time"][repositioned_data[0]["data"][0]["target_peak_position"]]
            plt.axvline(x=common_peak_time, color='red', linestyle='--', linewidth=3, 
                       label='Target peak position')
        
        # ラベル名を取得
        label_name = "Unknown"
        if len(repositioned_data) > 0:
            for entry in repositioned_data[0]["data"]:
                if entry["label_id"] == label_id:
                    label_name = entry["label_name"]
                    break
        
        plt.title(f'Overlayed Data - {label_name} (Label {label_id}, Session {session})\nAll Participants Aligned by Peak')
        plt.xlabel('Time')
        plt.ylabel('Normalized Value')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Overlayed plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_before_after_alignment(original_data, repositioned_data, person_idx=0, label_id=1, session=1, save_path=None):
    """
    アライメント前後の比較プロット
    
    Args:
        original_data (list): max_norm.max_norm()の結果（アライメント前）
        repositioned_data (list): reposition()の結果（アライメント後）
        person_idx (int): 表示する人物のインデックス
        label_id (int): 表示するラベルID
        session (int): 表示するセッション
        save_path (str): 保存パス
    """
    plt = _require_matplotlib()
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # アライメント前のデータを取得
    original_entries = []
    repositioned_entries = []
    
    for person in original_data:
        for entry in person["data"]:
            if entry["label_id"] == label_id and entry["session"] == session:
                original_entries.append((person["person"], entry))
    
    for person in repositioned_data:
        for entry in person["data"]:
            if entry["label_id"] == label_id and entry["session"] == session:
                repositioned_entries.append((person["person"], entry))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(original_entries)))
    
    # アライメント前
    ax1 = axes[0]
    for i, (person_name, entry) in enumerate(original_entries):
        ax1.plot(entry["time"], entry["value"], color=colors[i], linewidth=1.5, alpha=0.8, label=person_name)
        # 元のピーク位置をマーク
        peak_pos = find_main_peak(entry["value"])
        ax1.axvline(x=entry["time"][peak_pos], color=colors[i], linestyle=':', alpha=0.7)
    
    ax1.set_title(f'Before Alignment - {original_entries[0][1]["label_name"]} (Session {session})')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Normalized Value')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # アライメント後
    ax2 = axes[1]
    for i, (person_name, entry) in enumerate(repositioned_entries):
        ax2.plot(entry["time"], entry["value"], color=colors[i], linewidth=1.5, alpha=0.8, label=person_name)
        # ピーク位置をマーク
        peak_pos = entry["target_peak_position"]
        ax2.axvline(x=entry["time"][peak_pos], color=colors[i], linestyle=':', alpha=0.7)
    
    # 共通のピーク位置を表示
    if len(repositioned_entries) > 0:
        common_peak_time = repositioned_entries[0][1]["time"][repositioned_entries[0][1]["target_peak_position"]]
        ax2.axvline(x=common_peak_time, color='red', linestyle='--', linewidth=3, 
                   label='Common peak position')
    
    ax2.set_title(f'After Alignment - {repositioned_entries[0][1]["label_name"]} (Session {session})')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Normalized Value')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Before/after comparison saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

# スクリプトとして実行された場合のテスト
if __name__ == "__main__":
    if len(sys.argv) > 1:
        repositioned_data = reposition(sys.argv[1])
        
        # リポジション結果の確認
        for person_info in repositioned_data:
            print(f"\n=== {person_info['person']} (Repositioned) ===")
            for entry in person_info['data'][:3]:  # 最初の3件のみ表示
                print(f"Session {entry['session']}, Label {entry['label_id']} ({entry['label_name']})")
                print(f"  Original length: {entry['original_length']}, Aligned length: {entry['aligned_length']}")
                print(f"  Original peak: {entry['original_peak_position']}, Target peak: {entry['target_peak_position']}")
                print(f"  Shift: {entry['shift']}")
                print(f"  Value range: {entry['value'].min():.6f} - {entry['value'].max():.6f}")
        
        # 例のプロットを作成
        if len(repositioned_data) > 0:
            if plt is None:
                print("\nSkipping example alignment plot because matplotlib is disabled.")
            else:
                print("\nCreating example alignment plot...")
                plot_alignment_example(
                    repositioned_data, person_idx=0, label_id=1, session=1, save_path="alignment_example.png"
                )
    else:
        print("Usage: python reposition.py <directory_path>")
