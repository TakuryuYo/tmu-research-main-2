#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
波形アノテーションツール

操作方法:
- マウスでホバー: カーソル表示
- 'a': 開始地点を設定
- 's': 終了地点を設定  
- 'd': 次のデータへ
- 'r': アノテーションをやり直し
- 'p': 前のデータへ
- 'q': 終了
- 't': throwフラグをトグル (0↔1)

特徴量:
1. Power: 最大値をMaxデータで正規化 (0-1)
2. Time: (s地点 - a地点) をその個人のTime最大値で正規化 (0-1)
3. Speed: (最大値地点 - a地点) をその個人のSpeed最大値で正規化 (0-1)
"""

import numpy as np
import matplotlib
import re
# バックエンドを設定してダイアログを無効化
def setup_matplotlib_backend():
    """利用可能なバックエンドを自動選択"""
    backends_to_try = ['Qt5Agg', 'TkAgg', 'MacOSX', 'Agg']
    
    for backend in backends_to_try:
        try:
            matplotlib.use(backend, force=True)
            print(f"Using matplotlib backend: {backend}")
            if backend == 'Agg':
                print("Warning: Using non-interactive backend. Plots will be saved but not displayed interactively.")
            return backend
        except (ImportError, ModuleNotFoundError):
            continue
    
    print("Warning: Using default matplotlib backend")
    return None

setup_matplotlib_backend()
import matplotlib.pyplot as plt

# ダイアログを完全に無効化
plt.ioff()
matplotlib.rcParams['savefig.directory'] = ''
matplotlib.rcParams['interactive'] = False
matplotlib.rcParams['toolbar'] = 'None'
if 'figure.raise_window' in matplotlib.rcParams:
    matplotlib.rcParams['figure.raise_window'] = False

# 存在するキーマップのみ無効化
try:
    matplotlib.rcParams['keymap.save'] = []
except KeyError:
    pass
try:
    matplotlib.rcParams['keymap.quit'] = []
except KeyError:
    pass
import pickle
import os
import sys
import pandas as pd
from pathlib import Path
import csv
from datetime import datetime

# 上位ディレクトリのモジュールをインポート
sys.path.append(str(Path(__file__).parent / '../ana_0613'))
sys.path.append(str(Path(__file__).parent / '../make_dataset'))

# load.pyから被験者コード対応表をインポート
try:
    from load import SUBJECT_MAPPING, SUBJECT_NAME_TO_ID
except ImportError:
    print("Warning: Could not import SUBJECT_MAPPING from load.py")
    SUBJECT_MAPPING = {}
    SUBJECT_NAME_TO_ID = {}

# 数字ID→名前のマッピングを作成
ID_TO_SUBJECT_NAME = {v: k for k, v in SUBJECT_NAME_TO_ID.items()}

class WaveformAnnotator:
    def __init__(self, dataset_path='./aligned_dataset/dataset.bin'):
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
        self.current_throw = 0
        
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
        
        # 結果保存設定
        self.results_dir = 'annotation_results'
        self.individual_dir = os.path.join(self.results_dir, 'individual')
        self.plots_dir = os.path.join(self.results_dir, 'plots')
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.individual_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # データロード
        self.load_dataset()
        
        # 既存のアノテーションを読み込み（再開機能）
        self.load_existing_annotations()
        
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
                
                # 被験者IDを数字から名前に変換
                person_id = sample.get('person', 0)
                if person_id in ID_TO_SUBJECT_NAME:
                    sample['person_display'] = ID_TO_SUBJECT_NAME[person_id]
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
    
    def load_label_mapping(self, label_csv_path='../make_dataset/label.csv'):
        """ラベルマッピングを読み込み"""
        try:
            df = pd.read_csv(label_csv_path)
            label_mapping = dict(zip(df['id'], df['onomatopoeia']))
            print(f"Label mapping loaded: {len(label_mapping)} labels")
            return label_mapping
        except Exception as e:
            print(f"Warning: Could not load label mapping: {e}")
            return {}
    
    def get_label_id_from_name(self, label_name):
        """ラベル名からラベルIDを取得"""
        for label_id, name in self.label_mapping.items():
            if name == label_name:
                return label_id
        return 0  # 見つからない場合は0を返す
    
    def generate_sample_filename(self, sample_data, sample_index):
        """個別サンプル用のファイル名を生成"""
        person_display = sample_data.get('person_display', 'Unknown').replace('.', '_')
        label_name = sample_data.get('label_name', 'Unknown').replace('（', '_').replace('）', '_').replace('/', '_')
        session = sample_data.get('session', 'Unknown')
        label_id = self.get_label_id_from_name(sample_data.get('label_name', 'Unknown'))
        
        filename = f"sample{sample_index:04d}_{person_display}_session{session}_{label_name}_id{label_id:02d}.csv"
        return filename

    def determine_trial_num(self, sample_data):
        """
        試行番号 (session-1 or session-2) を算出
        pre -> 1, post -> 2, 判別できなければ 0 とする
        """
        session_value = sample_data.get('session')
        try:
            session_int = int(session_value)
            session_str = str(session_int)
        except (TypeError, ValueError):
            session_str = str(session_value) if session_value is not None else 'unknown'
        
        filename = str(sample_data.get('filename', '')).lower()
        trial_code = 0
        if filename:
            if re.search(r'(^|[-_])pre(?=($|[-_.]))', filename):
                trial_code = 1
            elif re.search(r'(^|[-_])post(?=($|[-_.]))', filename):
                trial_code = 2
        
        return f"{session_str}-{trial_code}"
    
    def get_annotation_for_index(self, sample_index):
        """指定インデックスの既存アノテーションを取得"""
        for ann in self.annotations:
            if ann['sample_index'] == sample_index:
                return ann
        return None

    def determine_prepos(self, sample_data):
        """
        ファイル名から pre/post を推定
        
        Returns:
            str: 'pre', 'post', または ''（判別できない場合）
        """
        filename = sample_data.get('filename')
        if not filename:
            return ''
        stem = Path(filename).stem.lower()
        if stem.endswith('pre'):
            return 'pre'
        if stem.endswith('post'):
            return 'post'
        match = re.search(r'(^|[-_])(pre|post)(?=($|[-_.]))', stem)
        if match:
            return match.group(2)
        return ''
    
    def load_existing_annotations(self):
        """既存のアノテーションファイルを読み込み（再開機能）"""
        print("Checking for existing annotations...")
        
        annotated_samples = set()
        
        # individual ディレクトリ内のCSVファイルをスキャン
        for filename in os.listdir(self.individual_dir):
            if filename.endswith('.csv') and filename.startswith('sample'):
                try:
                    # ファイル名からサンプルインデックスを抽出
                    sample_index = int(filename.split('_')[0].replace('sample', ''))
                    annotated_samples.add(sample_index)
                    
                    # アノテーションデータを読み込み
                    filepath = os.path.join(self.individual_dir, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            # 数値フィールドを適切な型に変換
                            start_raw_value = row.get('start_raw')
                            end_raw_value = row.get('end_raw')

                            annotation = {
                                'sample_index': int(row['sample_index']),
                                'person_id': row['person_id'],
                                'person_display': row.get('person_display', 'Unknown'),
                                'label_name': row['label_name'],
                                'session': int(row['session']),
                                'prepos': row.get('prepos', ''),
                                'start_pos': int(row['start_pos']),
                                'end_pos': int(row['end_pos']),
                                'start_raw': float(start_raw_value) if start_raw_value not in (None, '') else None,
                                'end_raw': float(end_raw_value) if end_raw_value not in (None, '') else None,
                                'power_raw': float(row['power_raw']),
                                'time_raw': int(row['time_raw']),
                                'speed_raw': int(row['speed_raw']),
                                'max_pos': int(row['max_pos']),
                                'signal_length': int(row['signal_length']),
                                'timestamp': row['timestamp'],
                                'trial_num': row.get('trial_num'),
                                'throw': int(row.get('throw', 0)) if row.get('throw', '') != '' else 0
                            }
                            if not annotation['trial_num']:
                                sample_data = self.dataset['data'][sample_index]
                                annotation['trial_num'] = self.determine_trial_num(sample_data)
                            if not annotation['prepos']:
                                sample_data = self.dataset['data'][sample_index]
                                annotation['prepos'] = self.determine_prepos(sample_data)
                            self.annotations.append(annotation)
                            break  # 1つのファイルには1つのアノテーションのみ
                except Exception as e:
                    print(f"Warning: Could not load annotation from {filename}: {e}")
        
        if len(annotated_samples) > 0:
            print(f"Found {len(annotated_samples)} existing annotations")
            print(f"Annotated samples: {sorted(annotated_samples)}")
            
            # 次にアノテーションすべきサンプルに移動
            next_index = 0
            for i in range(len(self.dataset['data'])):
                if i not in annotated_samples:
                    next_index = i
                    break
            else:
                # 全てのサンプルがアノテーション済み
                next_index = len(self.dataset['data']) - 1
                print("All samples have been annotated!")
            
            self.current_index = next_index
            print(f"Starting from sample {next_index + 1}/{len(self.dataset['data'])}")
            existing = self.get_annotation_for_index(self.current_index)
            self.current_throw = existing.get('throw', 0) if existing else 0
        else:
            print("No existing annotations found. Starting from the beginning.")
            self.current_throw = 0
    
    def calculate_features(self, signal, start_pos, end_pos):
        """
        アノテーション範囲から3つの特徴量を計算
        
        Args:
            signal: 信号データ
            start_pos: 開始位置 (a地点)
            end_pos: 終了位置 (b地点)
            
        Returns:
            dict: 計算された特徴量
        """
        if start_pos >= end_pos or start_pos < 0 or end_pos >= len(signal):
            print("Warning: Invalid annotation range")
            return None
        
        # 1. Power特徴量: 信号の最大値 (Maxデータで正規化は後で実装)
        max_val = np.max(np.abs(signal))
        power_raw = max_val
        
        # 2. Time特徴量: b地点 - a地点 の時間差分
        time_raw = end_pos - start_pos
        
        # 3. Speed特徴量: 最大値地点 - a地点 の時間差分
        max_pos = np.argmax(np.abs(signal))
        speed_raw = abs(max_pos - start_pos)  # 絶対値を取る

        # 4. Start/End raw amplitude values
        start_raw = float(signal[start_pos])
        end_raw = float(signal[end_pos])
        
        return {
            'power_raw': power_raw,
            'time_raw': time_raw,
            'speed_raw': speed_raw,
            'max_pos': max_pos,
            'start_pos': start_pos,
            'end_pos': end_pos,
            'start_raw': start_raw,
            'end_raw': end_raw
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
        self.ax.set_title(title, fontsize=14)
        
        self.ax.set_xlabel('Sample Index', fontsize=12)
        self.ax.set_ylabel('Amplitude', fontsize=12)
        self.ax.grid(True, alpha=0.3)
        
        # アノテーション線を描画
        if self.start_pos is not None:
            self.line_start = self.ax.axvline(x=self.start_pos, color='green', linestyle='--', 
                                           linewidth=2, alpha=0.8, label=f'Start (a): {self.start_pos}')
        if self.end_pos is not None:
            self.line_end = self.ax.axvline(x=self.end_pos, color='red', linestyle='--', 
                                         linewidth=2, alpha=0.8, label=f'End (b): {self.end_pos}')
        
        # 最大値位置を表示
        max_pos = np.argmax(np.abs(signal))
        self.ax.axvline(x=max_pos, color='orange', linestyle=':', linewidth=2, alpha=0.6, 
                       label=f'Max: {max_pos}')
        
        # カーソル線を初期化
        self.cursor_line = self.ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
        
        # 凡例
        self.ax.legend(loc='upper right')
        
        # 操作説明を追加
        info_text = "Controls: 'a'=Start, 's'=End, 't'=Toggle throw, 'd'=Next, 'r'=Reset, 'p'=Previous, 'q'=Quit"
        self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
        throw_text = f"Throw flag: {self.current_throw} (0=use, 1=discard)"
        self.ax.text(0.02, 0.9, throw_text, transform=self.ax.transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # アノテーション状態表示
        if self.start_pos is not None and self.end_pos is not None:
            # 特徴量を計算して表示
            features = self.calculate_features(signal, self.start_pos, self.end_pos)
            if features:
                feature_text = (f"Features (raw):\n"
                              f"Power: {features['power_raw']:.2f}\n"
                              f"Time: {features['time_raw']} samples\n"
                              f"Speed: {features['speed_raw']} samples")
                self.ax.text(0.02, 0.02, feature_text, transform=self.ax.transAxes, 
                            fontsize=10, verticalalignment='bottom',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8))
        
        self.fig.canvas.draw()
    
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
        print(f"DEBUG: Key pressed: {event.key}")  # デバッグ用
        
        # 保存関連のキーを明示的にブロック
        if event.key in ['ctrl+s', 'cmd+s', 's', 'ctrl+shift+s', 'cmd+shift+s']:
            if event.key in ['ctrl+s', 'cmd+s', 'ctrl+shift+s', 'cmd+shift+s']:
                print("DEBUG: Blocked save shortcut")
                return  # 保存ショートカットをブロック
        
        if event.key == 'a':
            # 開始地点を設定
            self.start_pos = self.mouse_x
            print(f"Start position (a) set to: {self.start_pos}")
            self.plot_current_data()
            
        elif event.key == 's':
            # 終了地点を設定
            self.end_pos = self.mouse_x
            print(f"End position (s) set to: {self.end_pos}")
            self.plot_current_data()
        
        elif event.key == 't':
            # throwフラグをトグル
            self.current_throw = 0 if self.current_throw else 1
            state_label = "discard" if self.current_throw else "use"
            print(f"Toggled throw flag to {self.current_throw} ({state_label})")
            self.plot_current_data()
            
        elif event.key == 'd':
            # 次のデータ
            self.save_current_annotation()
            self.next_data()
            
        elif event.key == 'r':
            # リセット
            self.start_pos = None
            self.end_pos = None
            self.current_throw = 0
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
    
    def save_current_annotation(self):
        """現在のアノテーションを保存（個別ファイルとして即座に保存）"""
        if self.start_pos is None or self.end_pos is None:
            print("Warning: Incomplete annotation (missing start or end position)")
            return
            
        if self.current_index >= len(self.dataset['data']):
            return
            
        current_data = self.dataset['data'][self.current_index]
        signal = current_data['aligned_value']
        
        # 特徴量を計算
        features = self.calculate_features(signal, self.start_pos, self.end_pos)
        
        if features is None:
            print("Warning: Could not calculate features from current annotation")
            return
        
        # アノテーション結果を保存
        annotation = {
            'sample_index': self.current_index,
            'person_id': current_data.get('person_id', 'Unknown'),
            'person_display': current_data.get('person_display', 'Unknown'),
            'label_name': current_data.get('label_name', 'Unknown'),
            'session': current_data.get('session', 'Unknown'),
            'prepos': self.determine_prepos(current_data),
            'start_pos': self.start_pos,
            'end_pos': self.end_pos,
            'start_raw': features['start_raw'],
            'end_raw': features['end_raw'],
            'power_raw': features['power_raw'],
            'time_raw': features['time_raw'],
            'speed_raw': features['speed_raw'],
            'max_pos': features['max_pos'],
            'signal_length': len(signal),
            'timestamp': datetime.now().isoformat(),
            'trial_num': self.determine_trial_num(current_data),
            'throw': int(self.current_throw)
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
        
        # 個別ファイルとして即座に保存
        self.save_individual_annotation(annotation)
    
    def save_individual_annotation(self, annotation):
        """個別のアノテーションを単独のCSVファイルとして保存"""
        sample_index = annotation['sample_index']
        current_data = self.dataset['data'][sample_index]
        
        # ファイル名を生成
        filename = self.generate_sample_filename(current_data, sample_index)
        filepath = os.path.join(self.individual_dir, filename)
        
        # CSVファイルに保存
        fieldnames = ['sample_index', 'person_id', 'person_display', 'label_name', 'session', 'prepos',
                     'start_pos', 'end_pos', 'start_raw', 'end_raw', 'max_pos', 'signal_length',
                     'power_raw', 'time_raw', 'speed_raw', 'trial_num', 'throw', 'timestamp']
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(annotation)
        
        print(f"Individual annotation saved: {filename}")
    
    def save_current_plot(self):
        """プロット保存は無効化"""
        pass
    
    def normalize_features(self):
        """
        全アノテーションの特徴量を正規化
        
        1. Power: その個人のMaxラベルの最大値で正規化
        2. Time: その個人のTime最大値で正規化
        3. Speed: その個人のSpeed最大値で正規化
        """
        if len(self.annotations) == 0:
            return
        
        # 個人別にMaxラベルのPower値、Time、Speedの最大値を計算
        person_max_power = {}
        person_max_time = {}
        person_max_speed = {}
        
        for ann in self.annotations:
            person = ann['person_id']
            
            if person not in person_max_power:
                person_max_power[person] = 0
                person_max_time[person] = 0
                person_max_speed[person] = 0
            
            # その個人のMaxラベルのPower値を記録
            if ann['label_name'] == 'Max':
                person_max_power[person] = max(person_max_power[person], ann['power_raw'])
            
            person_max_time[person] = max(person_max_time[person], ann['time_raw'])
            person_max_speed[person] = max(person_max_speed[person], ann['speed_raw'])
        
        print("Person-wise normalization references:")
        for person in person_max_power:
            print(f"  {person}: Power(Max)={person_max_power[person]}, Time={person_max_time[person]}, Speed={person_max_speed[person]}")
        
        # 正規化を適用
        for ann in self.annotations:
            person = ann['person_id']
            
            # Power正規化 (0-1) - その個人のMaxラベルで正規化
            if person_max_power[person] > 0:
                ann['power_normalized'] = ann['power_raw'] / person_max_power[person]
                ann['power_normalized'] = min(ann['power_normalized'], 1.0)  # 1.0でクリップ
            else:
                print(f"Warning: No 'Max' label found for {person}. Setting power_normalized to 0.")
                ann['power_normalized'] = 0.0
            
            # Time正規化 (0-1)
            if person_max_time[person] > 0:
                ann['time_normalized'] = ann['time_raw'] / person_max_time[person]
            else:
                ann['time_normalized'] = 0.0
            
            # Speed正規化 (0-1)
            if person_max_speed[person] > 0:
                ann['speed_normalized'] = ann['speed_raw'] / person_max_speed[person]
            else:
                ann['speed_normalized'] = 0.0
    
    def save_all_annotations(self):
        """すべてのアノテーションをCSVファイルに保存"""
        if len(self.annotations) == 0:
            print("No annotations to save")
            return
        
        # 特徴量を正規化
        self.normalize_features()
        
        # 詳細なファイル名を生成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # アノテーション情報のサマリーを作成
        if len(self.annotations) == 1:
            # 単一のアノテーションの場合
            ann = self.annotations[0]
            person = ann['person_id'].replace('.', '_')
            label_name = ann['label_name'].replace('（', '_').replace('）', '_').replace('/', '_')
            session = ann['session']
            label_id = self.get_label_id_from_name(ann['label_name'])
            filename = f"annotation_{person}_session{session}_{label_name}_id{label_id:02d}_{timestamp}.csv"
        else:
            # 複数のアノテーションの場合
            persons = list(set(ann['person_id'] for ann in self.annotations))
            sessions = list(set(ann['session'] for ann in self.annotations))
            labels = list(set(ann['label_name'] for ann in self.annotations))
            
            if len(persons) == 1:
                person_str = persons[0].replace('.', '_')
            else:
                person_str = f"multi_persons_{len(persons)}"
            
            if len(sessions) == 1:
                session_str = f"session{sessions[0]}"
            else:
                session_str = f"sessions_{'-'.join(map(str, sorted(sessions)))}"
            
            if len(labels) == 1:
                label_str = labels[0].replace('（', '_').replace('）', '_').replace('/', '_')
                label_id = self.get_label_id_from_name(labels[0])
                label_str += f"_id{label_id:02d}"
            else:
                label_str = f"multi_labels_{len(labels)}"
            
            filename = f"annotation_{person_str}_{session_str}_{label_str}_{len(self.annotations)}samples_{timestamp}.csv"
        
        filepath = os.path.join(self.results_dir, filename)
        
        # CSVファイルに保存
        fieldnames = ['sample_index', 'person_id', 'person_display', 'label_name', 'session', 'prepos',
                     'start_pos', 'end_pos', 'start_raw', 'end_raw', 'max_pos', 'signal_length',
                     'power_raw', 'time_raw', 'speed_raw',
                     'power_normalized', 'time_normalized', 'speed_normalized',
                     'trial_num', 'throw', 'timestamp']
        
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
            
        powers = [ann['power_normalized'] for ann in self.annotations]
        times = [ann['time_normalized'] for ann in self.annotations]
        speeds = [ann['speed_normalized'] for ann in self.annotations]
        
        print(f"\nAnnotation Statistics (Normalized):")
        print(f"  Total annotations: {len(self.annotations)}")
        print(f"  Power - min: {np.min(powers):.4f}, max: {np.max(powers):.4f}, mean: {np.mean(powers):.4f}")
        print(f"  Time  - min: {np.min(times):.4f}, max: {np.max(times):.4f}, mean: {np.mean(times):.4f}")
        print(f"  Speed - min: {np.min(speeds):.4f}, max: {np.max(speeds):.4f}, mean: {np.mean(speeds):.4f}")
    
    def next_data(self):
        """次のデータに移動"""
        if self.current_index < len(self.dataset['data']) - 1:
            self.current_index += 1
            self.start_pos = None
            self.end_pos = None
            existing = self.get_annotation_for_index(self.current_index)
            self.current_throw = int(existing.get('throw', 0)) if existing else 0
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
            existing = self.get_annotation_for_index(self.current_index)
            self.current_throw = int(existing.get('throw', 0)) if existing else 0
            print(f"Moved to sample {self.current_index + 1}/{len(self.dataset['data'])}")
            self.plot_current_data()
        else:
            print("Already at the first sample")
    
    def start_annotation(self):
        """アノテーションを開始"""
        print("=" * 60)
        print("Waveform Annotation Tool")
        print("=" * 60)
        print("Controls:")
        print("  'a' - Set start position")
        print("  's' - Set end position")
        print("  't' - Toggle throw flag (0↔1)")
        print("  'd' - Next data")
        print("  'r' - Reset current annotation")
        print("  'p' - Previous data")
        print("  'q' - Quit and save")
        print("=" * 60)
        
        # バックエンドがAggの場合は非対話モードで実行
        current_backend = matplotlib.get_backend()
        if current_backend == 'Agg':
            print("Non-interactive mode: Automatic processing without GUI")
            self.run_non_interactive_mode()
            return
        
        # プロットウィンドウを設定
        self.fig, self.ax = plt.subplots(figsize=(16, 8))
        
        # イベントハンドラを接続
        self.motion_cid = self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.key_cid = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # 最初のデータをプロット
        self.plot_current_data()
        
        # 進捗状況を表示
        annotated_count = len([f for f in os.listdir(self.individual_dir) if f.endswith('.csv')])
        total_count = len(self.dataset['data'])
        print(f"Progress: {annotated_count}/{total_count} samples annotated ({100*annotated_count/total_count:.1f}%)")
        
        # プロットを表示
        plt.show()
    
    def run_non_interactive_mode(self):
        """非対話モードでの実行（バックエンドがAggの場合）"""
        print("Error: Interactive annotation requires a GUI backend.")
        print("Please install one of the following:")
        print("  - For Qt: pip install PyQt5 or pip install PySide2")
        print("  - For Tkinter: Install tkinter (usually comes with Python)")
        print("  - For macOS: Use built-in MacOSX backend")
        print("")
        print("Alternatively, you can run the tool in a different environment with GUI support.")
        return

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
