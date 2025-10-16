#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
インタラクティブな波形アノテーションツール

使用方法:
- マウスポインタを当ててキーを押す:
  - 'a': 開始地点を設定
  - 's': 終了地点を設定
  - 'f': 次のデータへ進む
  - 'r': そのデータのやり直し（リセット）
  - 'p': 前のデータに戻る
  - 'q': 終了

動作:
1. create_dataset.pyで作成されたデータを読み込み
2. 波形を表示してユーザーがアノテーション
3. アノテーション範囲から3つの特徴量（Power, Speed, Time）を計算
4. 結果をCSVファイルに保存
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import os
import sys
import pandas as pd
from pathlib import Path
import csv
from datetime import datetime
from scipy import signal as scipy_signal

# 上位ディレクトリのモジュールをインポート
sys.path.append(str(Path(__file__).parent / '../ana_0613'))
sys.path.append(str(Path(__file__).parent / '../make_dataset'))

# load.pyから被験者コード対応表をインポート
try:
    from load import SUBJECT_MAPPING
except ImportError:
    print("Warning: Could not import SUBJECT_MAPPING from load.py")
    SUBJECT_MAPPING = {}

# matplotlibを対話モードに設定（アノテーション用）
plt.ion()  # 対話モードをオン

class WaveformAnnotator:
    def __init__(self, dataset_path='aligned_dataset/dataset.bin'):
        """
        波形アノテーターの初期化
        
        Args:
            dataset_path: データセットファイルのパス
        """
        self.dataset_path = dataset_path
        self.dataset = None
        self.current_index = 0
        self.start_pos = None
        self.end_pos = None
        
        # マウス位置とイベント管理
        self.mouse_x = 0
        self.line_start = None
        self.line_end = None
        
        # アノテーション結果保存
        self.annotations = []
        
        # プロット要素
        self.fig = None
        self.ax = None
        self.signal_line = None
        self.cursor_line = None
        
        # イベント接続ID
        self.motion_cid = None
        self.key_cid = None
        
        # 自動保存設定
        self.auto_save_dir = 'annotation_results/plots'
        os.makedirs(self.auto_save_dir, exist_ok=True)
        
        # データロード
        self.load_dataset()
        
    def load_dataset(self):
        """データセットを読み込み"""
        print(f"Loading dataset from {self.dataset_path}...")
        
        try:
            with open(self.dataset_path, 'rb') as f:
                loaded_data = pickle.load(f)
            
            # ラベルマッピングを読み込み
            self.label_mapping = self.load_label_mapping()
            
            # create_dataset.pyで作成されたデータ構造に対応し、ラベル名と被験者IDを修正
            data_list = loaded_data['dataset_info']['data']
            for sample in data_list:
                # ラベル名を修正
                label_id = sample.get('label_id', 0)
                if label_id in self.label_mapping:
                    sample['label_name'] = self.label_mapping[label_id]
                else:
                    sample['label_name'] = f'Unknown_{label_id}'
                
                # 被験者IDを数字からアルファベットに変換
                person_id = sample.get('person', 0)
                if person_id in SUBJECT_MAPPING:
                    sample['person_display'] = SUBJECT_MAPPING[person_id]
                else:
                    sample['person_display'] = f'Unknown_{person_id}'
                
                # 元のperson IDも保持
                sample['person_id'] = person_id
            
            self.dataset = {
                'data': data_list
            }
            
            print(f"Dataset loaded successfully!")
            print(f"Total samples: {len(self.dataset['data'])}")
            
            # データセット情報を表示
            if len(self.dataset['data']) > 0:
                sample_data = self.dataset['data'][0]
                print(f"First sample info:")
                print(f"  Person: {sample_data.get('person_display', 'Unknown')} (ID: {sample_data.get('person_id', 'Unknown')})")
                print(f"  Label: {sample_data.get('label_name', 'Unknown')}")
                print(f"  Signal length: {len(sample_data['aligned_value'])}")
                
        except FileNotFoundError:
            print(f"Error: Dataset file not found at {self.dataset_path}")
            print("Please run create_dataset.py first to generate the dataset.")
            raise
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
    
    def load_label_mapping(self, label_csv_path='../ana_0613/label.csv'):
        """ラベルマッピングを読み込み"""
        try:
            df = pd.read_csv(label_csv_path)
            label_mapping = dict(zip(df['id'], df['onomatopoeia']))
            print(f"Label mapping loaded: {len(label_mapping)} labels")
            return label_mapping
        except Exception as e:
            print(f"Warning: Could not load label mapping: {e}")
            return {}
    
    def extract_power_feature(self, signal, normalize=True):
        """
        特徴量1: パワー特徴量（convert_features_3_6.pyと同じ方式）
        
        Args:
            signal: 信号データ
            normalize: 正規化するかどうか
            
        Returns:
            float: パワー特徴量
        """
        # 絶対値の最大値を取得
        max_val = np.max(np.abs(signal))
        
        if normalize:
            # 簡単化: マップされた範囲で正規化 (0-1)
            # 実際はMaxラベルを参照すべきだが、アノテーションツールでは簡略化
            power = min(max_val / 100.0, 1.0)  # 仮の正規化基準
        else:
            power = max_val
            
        return power
    
    def extract_speed_feature(self, signal, normalize=True, rise_threshold=0.07):
        """
        特徴量2: 立ち上がり時間（Speed）- convert_features_3_6.pyと同じ方式
        
        Args:
            signal: 信号データ
            normalize: 正規化するかどうか
            rise_threshold: 立ち上がり検出の閾値
            
        Returns:
            float: Speed特徴量（立ち上がり時間）
        """
        abs_signal = np.abs(signal)
        max_val = np.max(abs_signal)
        max_pos = np.argmax(abs_signal)
        
        # ベースライン計算
        baseline_window = min(20, len(signal) // 10)
        baseline = np.mean(abs_signal[:baseline_window])
        
        # 立ち上がり閾値
        threshold = baseline + (max_val - baseline) * rise_threshold
        
        # 立ち上がり開始点を探す
        rise_start = 0
        for j in range(len(abs_signal)):
            if abs_signal[j] >= threshold:
                rise_start = j
                break
        
        # 立ち上がり時間を計算
        rise_time = max_pos - rise_start
        
        # 無効な値を防ぐ
        if rise_time < 0 or rise_start >= max_pos:
            # フォールバック: 最大値の50%に到達した点を探す
            fallback_threshold = baseline + (max_val - baseline) * 0.5
            for j in range(max_pos):
                if abs_signal[j] >= fallback_threshold:
                    rise_start = j
                    break
            rise_time = max_pos - rise_start
        
        rise_time = max(rise_time, 1)  # 最小1サンプル
        
        if normalize:
            # 簡単化: 信号長で正規化
            speed = rise_time / len(signal)
        else:
            speed = rise_time
            
        return speed
    
    def extract_time_feature(self, signal, normalize=True, duration_threshold=0.12):
        """
        特徴量3: 動作持続時間（Time）- convert_features_3_6.pyと同じ方式
        
        Args:
            signal: 信号データ
            normalize: 正規化するかどうか
            duration_threshold: 動作検出の閾値
            
        Returns:
            float: Time特徴量
        """
        abs_signal = np.abs(signal)
        max_val = np.max(abs_signal)
        max_pos = np.argmax(abs_signal)
        
        # ベースライン計算
        baseline_window = min(20, len(abs_signal) // 10)
        baseline = np.mean(abs_signal[:baseline_window])
        
        # 立ち上がり・立ち下がり検出の閾値を設定
        threshold = baseline + (max_val - baseline) * duration_threshold
        
        # 立ち上がりの開始点を検出
        rise_start = 0
        for j in range(len(abs_signal)):
            if abs_signal[j] >= threshold:
                rise_start = j
                break
        
        # 立ち下がりの終了点を検出
        fall_end = len(abs_signal) - 1
        for j in range(len(abs_signal) - 1, -1, -1):
            if abs_signal[j] >= threshold:
                fall_end = j
                break
        
        # 持続時間を計算
        duration = fall_end - rise_start + 1
        
        # 最小値チェック
        if duration < 5:
            window_size = min(10, len(abs_signal) // 20)
            rise_start = max(0, max_pos - window_size)
            fall_end = min(len(abs_signal) - 1, max_pos + window_size)
            duration = fall_end - rise_start + 1
        
        if normalize:
            # 全信号長で正規化
            time_feature = duration / len(signal)
        else:
            time_feature = duration
            
        return time_feature
    
    def extract_features_from_range(self, signal, start_idx, end_idx):
        """
        指定された範囲から3つの特徴量を抽出
        
        Args:
            signal: 元の信号
            start_idx: 開始インデックス
            end_idx: 終了インデックス
            
        Returns:
            dict: 抽出された特徴量
        """
        # 範囲を切り出し
        if start_idx >= end_idx:
            print("Warning: Invalid range (start >= end)")
            return None
            
        segment = signal[start_idx:end_idx]
        
        if len(segment) == 0:
            print("Warning: Empty segment")
            return None
        
        # 各特徴量を計算
        power = self.extract_power_feature(segment, normalize=True)
        speed = self.extract_speed_feature(segment, normalize=True)
        time = self.extract_time_feature(segment, normalize=True)
        
        return {
            'power': power,
            'speed': speed,
            'time': time,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'segment_length': len(segment)
        }
    
    def plot_current_data(self):
        """現在のデータをプロット"""
        if self.dataset is None or self.current_index >= len(self.dataset['data']):
            return
            
        # 現在のデータを取得
        current_data = self.dataset['data'][self.current_index]
        signal = current_data['aligned_value']
        
        # プロットをクリア
        if self.ax is not None:
            self.ax.clear()
        
        # 信号をプロット
        time_axis = np.arange(len(signal))
        self.signal_line, = self.ax.plot(time_axis, signal, 'b-', linewidth=1.0, label='Signal')
        
        # タイトルと情報を設定
        person_display = current_data.get('person_display', 'Unknown')
        label_name = current_data.get('label_name', 'Unknown')
        session = current_data.get('session', 'Unknown')
        title = f"Sample {self.current_index + 1}/{len(self.dataset['data'])}: {person_display} - {label_name} (Session {session})"
        self.ax.set_title(title)
        
        self.ax.set_xlabel('Sample Index')
        self.ax.set_ylabel('Amplitude')
        self.ax.grid(True, alpha=0.3)
        
        # アノテーション線を描画
        if self.start_pos is not None:
            self.line_start = self.ax.axvline(x=self.start_pos, color='green', linestyle='--', linewidth=2, label='Start')
        if self.end_pos is not None:
            self.line_end = self.ax.axvline(x=self.end_pos, color='red', linestyle='--', linewidth=2, label='End')
        
        # カーソル線を初期化
        self.cursor_line = self.ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
        
        # 凡例
        self.ax.legend()
        
        # 操作説明を追加
        info_text = "Controls: 'a'=Start, 's'=End, 'f'=Next, 'r'=Reset, 'p'=Previous, 'q'=Quit"
        self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
        
        # アノテーション状態表示
        if self.start_pos is not None and self.end_pos is not None:
            # 特徴量を計算して表示
            features = self.extract_features_from_range(signal, self.start_pos, self.end_pos)
            if features:
                feature_text = f"Power: {features['power']:.4f}, Speed: {features['speed']:.4f}, Time: {features['time']:.4f}"
                self.ax.text(0.02, 0.02, feature_text, transform=self.ax.transAxes, 
                            fontsize=10, verticalalignment='bottom',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
        
        self.fig.canvas.draw()
    
    def save_current_annotation(self):
        """現在のアノテーションを保存"""
        if self.start_pos is None or self.end_pos is None:
            print("Warning: Incomplete annotation (missing start or end position)")
            return
            
        if self.current_index >= len(self.dataset['data']):
            return
            
        current_data = self.dataset['data'][self.current_index]
        signal = current_data['aligned_value']
        
        # 特徴量を抽出
        features = self.extract_features_from_range(signal, self.start_pos, self.end_pos)
        
        if features is None:
            print("Warning: Could not extract features from current annotation")
            return
        
        # アノテーション結果を保存
        annotation = {
            'sample_index': self.current_index,
            'person_id': current_data.get('person_id', 'Unknown'),
            'person_display': current_data.get('person_display', 'Unknown'),
            'label_name': current_data.get('label_name', 'Unknown'),
            'session': current_data.get('session', 'Unknown'),
            'start_pos': self.start_pos,
            'end_pos': self.end_pos,
            'power': features['power'],
            'speed': features['speed'],
            'time': features['time'],
            'segment_length': features['segment_length'],
            'signal_length': len(signal),
            'timestamp': datetime.now().isoformat()
        }
        
        # 既存のアノテーションを更新または追加
        existing_idx = None
        for i, ann in enumerate(self.annotations):
            if ann['sample_index'] == self.current_index:
                existing_idx = i
                break
        
        if existing_idx is not None:
            self.annotations[existing_idx] = annotation
            print(f"Updated annotation for sample {self.current_index}")
        else:
            self.annotations.append(annotation)
            print(f"Saved annotation for sample {self.current_index}")
    
    def save_all_annotations(self):
        """すべてのアノテーションをCSVファイルに保存"""
        if len(self.annotations) == 0:
            print("No annotations to save")
            return
        
        # 出力ディレクトリを作成
        output_dir = 'annotation_results'
        os.makedirs(output_dir, exist_ok=True)
        
        # ファイル名を生成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"annotation_results_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)
        
        # CSVファイルに保存
        fieldnames = ['sample_index', 'person_id', 'person_display', 'label_name', 'session', 
                     'start_pos', 'end_pos', 'power', 'speed', 'time', 
                     'segment_length', 'signal_length', 'timestamp']
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for annotation in self.annotations:
                writer.writerow(annotation)
        
        print(f"Saved {len(self.annotations)} annotations to {filepath}")
        
        # 統計情報を表示
        self.print_annotation_statistics()
    
    def print_annotation_statistics(self):
        """アノテーション統計を表示"""
        if len(self.annotations) == 0:
            return
            
        powers = [ann['power'] for ann in self.annotations]
        speeds = [ann['speed'] for ann in self.annotations]
        times = [ann['time'] for ann in self.annotations]
        
        print(f"\nAnnotation Statistics:")
        print(f"  Total annotations: {len(self.annotations)}")
        print(f"  Power - min: {np.min(powers):.4f}, max: {np.max(powers):.4f}, mean: {np.mean(powers):.4f}")
        print(f"  Speed - min: {np.min(speeds):.4f}, max: {np.max(speeds):.4f}, mean: {np.mean(speeds):.4f}")
        print(f"  Time  - min: {np.min(times):.4f}, max: {np.max(times):.4f}, mean: {np.mean(times):.4f}")
    
    def on_mouse_move(self, event):
        """マウス移動時のイベント処理"""
        if event.inaxes != self.ax:
            return
            
        self.mouse_x = int(event.xdata) if event.xdata is not None else 0
        
        # カーソル線を更新
        if self.cursor_line is not None:
            self.cursor_line.set_xdata([self.mouse_x, self.mouse_x])
            self.fig.canvas.draw_idle()
    
    def on_key_press(self, event):
        """キー押下時のイベント処理"""
        if event.key == 'a':
            # 開始地点を設定
            self.start_pos = self.mouse_x
            print(f"Start position set to: {self.start_pos}")
            self.plot_current_data()
            
        elif event.key == 's':
            # 終了地点を設定
            self.end_pos = self.mouse_x
            print(f"End position set to: {self.end_pos}")
            self.plot_current_data()
            
        elif event.key == 'f':
            # 次のデータ
            self.save_current_annotation()
            self.next_data()
            
        elif event.key == 'r':
            # リセット
            self.start_pos = None
            self.end_pos = None
            print("Annotation reset")
            self.plot_current_data()
            
        elif event.key == 'p':
            # 前のデータ
            self.save_current_annotation()
            self.previous_data()
            
        elif event.key == 'q':
            # 終了
            self.save_current_annotation()
            self.save_all_annotations()
            plt.close(self.fig)
            print("Annotation tool closed.")
    
    def next_data(self):
        """次のデータに移動"""
        if self.current_index < len(self.dataset['data']) - 1:
            self.current_index += 1
            self.start_pos = None
            self.end_pos = None
            print(f"Moved to sample {self.current_index + 1}/{len(self.dataset['data'])}")
            self.plot_current_data()
        else:
            print("Already at the last sample")
    
    def previous_data(self):
        """前のデータに移動"""
        if self.current_index > 0:
            self.current_index -= 1
            self.start_pos = None
            self.end_pos = None
            print(f"Moved to sample {self.current_index + 1}/{len(self.dataset['data'])}")
            self.plot_current_data()
        else:
            print("Already at the first sample")
    
    def save_current_plot(self):
        """現在のプロットを自動保存"""
        if self.fig is None:
            return
            
        # ファイル名を生成
        current_data = self.dataset['data'][self.current_index]
        person_display = current_data.get('person_display', 'Unknown')
        label_name = current_data.get('label_name', 'Unknown')
        session = current_data.get('session', 'Unknown')
        
        # 安全なファイル名を作成
        safe_person = person_display.replace('.', '_')
        safe_label = label_name.replace('（', '_').replace('）', '_').replace('/', '_')
        filename = f"annotation_{safe_person}_{safe_label}_session{session}_sample{self.current_index:04d}.png"
        filepath = os.path.join(self.auto_save_dir, filename)
        
        # プロットを保存
        self.fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {filename}")
    
    def start_annotation(self):
        """アノテーションを開始"""
        print("=" * 60)
        print("Waveform Annotation Tool")
        print("=" * 60)
        print("Controls:")
        print("  'a' - Set start position")
        print("  's' - Set end position")
        print("  'f' - Next data")
        print("  'r' - Reset current annotation") 
        print("  'p' - Previous data")
        print("  'q' - Quit and save")
        print("=" * 60)
        
        # プロットウィンドウを設定
        self.fig, self.ax = plt.subplots(figsize=(15, 8))
        
        # イベントハンドラを接続
        self.motion_cid = self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.key_cid = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # 最初のデータをプロット
        self.plot_current_data()
        
        # プロットを表示
        plt.show()

def main():
    """メイン処理"""
    try:
        annotator = WaveformAnnotator()
        annotator.start_annotation()
    except Exception as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    main()