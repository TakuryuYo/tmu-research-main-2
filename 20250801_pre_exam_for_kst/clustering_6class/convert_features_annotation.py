#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
アノテーション結果を使用した特徴量変換プログラム

annotation_results/individual/ の個別CSVファイルから特徴量データを読み込み、
正規化処理を行ってplot_3_features_6フォーマットで出力する。

除外処理は行わず、アノテーションされた全てのデータを使用する。
"""

import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
import japanize_matplotlib

plt.rcParams['font.family'] = ['Times New Roman', 'IPAexGothic']  # 英数字はTimes New Roman、日本語はIPAexGothic
plt.rcParams['font.sans-serif'] = ['IPAexGothic']  # 日本語フォント
plt.rcParams['font.serif'] = ['Times New Roman']  #

plt.rcParams['axes.labelsize'] = 30  # 軸ラベルのフォントサイズ
plt.rcParams['xtick.labelsize'] = 30  # x軸の目盛りラベルのフォントサイズ
plt.rcParams['ytick.labelsize'] = 30  # y軸の目盛りラベルのフォントサイズ
plt.rcParams['axes.titlesize'] = 16   # タイトルのフォントサイズ

import os
import pickle
from pathlib import Path
import sys
import pandas as pd
import json
import glob
import csv
from datetime import datetime

# 上位ディレクトリのモジュールをインポート
sys.path.append(str(Path(__file__).parent / '../ana_0613'))

# 被験者IDマッピング（アノテーションデータ→アンケートデータ）
PERSON_ID_MAPPING = {
    1: '1',   # inamura → 稲村怜於奈
    2: '2',   # utsumi → 内海恒希
    3: '3',   # kawano → 河野悠輝
    4: '4',   # hatanaka → 畠中秀斗
    5: '5',   # watanabe → 渡辺悠生
    6: '6'    # kikuchi → 菊地
}

def load_annotation_results(annotation_dir='../annotation/annotation_results/individual'):
    """
    アノテーション結果から特徴量データを読み込み
    
    Args:
        annotation_dir: アノテーション結果ディレクトリのパス
        
    Returns:
        dict: 特徴量データ
    """
    print(f"Loading annotation results from {annotation_dir}...")
    
    if not os.path.exists(annotation_dir):
        print(f"Error: Annotation directory not found at {annotation_dir}")
        print("Please run annotation_tool.py first to generate annotation data.")
        return None
    
    # CSVファイルを読み込み
    csv_files = [f for f in os.listdir(annotation_dir) if f.endswith('.csv')]
    
    if len(csv_files) == 0:
        print(f"Error: No annotation CSV files found in {annotation_dir}")
        return None
    
    print(f"Found {len(csv_files)} annotation files")
    
    annotations = []
    
    for csv_file in csv_files:
        filepath = os.path.join(annotation_dir, csv_file)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # 数値フィールドを適切な型に変換
                    annotation = {
                        'sample_index': int(row['sample_index']),
                        'person_id': row['person_id'],
                        'label_name': row['label_name'],
                        'session': int(row['session']),
                        'start_pos': int(row['start_pos']),
                        'end_pos': int(row['end_pos']),
                        'power_raw': float(row['power_raw']),
                        'time_raw': int(row['time_raw']),
                        'speed_raw': int(row['speed_raw']),
                        'max_pos': int(row['max_pos']),
                        'signal_length': int(row['signal_length']),
                        'timestamp': row['timestamp']
                    }
                    annotations.append(annotation)
                    break  # 1つのファイルには1つのアノテーションのみ
        except Exception as e:
            print(f"Warning: Could not load annotation from {csv_file}: {e}")
            continue
    
    print(f"Successfully loaded {len(annotations)} annotations")
    
    # データを特徴量形式に変換
    features = []
    y_label_names = []
    person_ids = []
    sessions = []
    sample_indices = []
    
    for ann in annotations:
        # 特徴量: [power_raw, time_raw, speed_raw]
        features.append([ann['power_raw'], ann['time_raw'], ann['speed_raw']])
        y_label_names.append(ann['label_name'])
        person_ids.append(ann['person_id'])
        sessions.append(ann['session'])
        sample_indices.append(ann['sample_index'])
    
    features = np.array(features)
    y_label_names = np.array(y_label_names)
    person_ids = np.array(person_ids)
    sessions = np.array(sessions)
    sample_indices = np.array(sample_indices)
    
    print(f"Feature data shape: {features.shape}")
    print(f"Unique persons: {len(np.unique(person_ids))}")
    print(f"Unique labels: {len(np.unique(y_label_names))}")
    print(f"Unique sessions: {sorted(np.unique(sessions))}")
    
    return {
        'features': features,
        'y_label_name': y_label_names,
        'person_ids': person_ids,
        'sessions': sessions,
        'sample_indices': sample_indices,
        'annotations': annotations
    }


def normalize_annotation_features(annotation_data):
    """
    アノテーション特徴量を正規化
    
    個人別・セッション別に正規化:
    1. Power: 最小値=Max以外の最小値、最大値=Maxラベルの最大値でmin-max正規化
    2. Time: その個人のそのSessionのMaxラベル以外のデータを使用してmin-max正規化 (0-1)  
    3. Speed: その個人のそのSessionのMaxラベル以外のデータを使用してmin-max正規化 (0-1)
    
    プロットには正規化に使用したMaxデータのみを表示。
    
    Args:
        annotation_data: load_annotation_results()の戻り値
        
    Returns:
        dict: 正規化済み特徴量データ + 使用したMaxデータのID
    """
    print("Normalizing annotation features by person and session...")
    
    features = annotation_data['features'].copy()
    y_label_names = annotation_data['y_label_name']
    person_ids = annotation_data['person_ids']
    sessions = annotation_data['sessions']
    sample_indices = annotation_data['sample_indices']
    
    normalized_features = features.copy()
    
    # 個人別・セッション別の正規化参照値と使用したMaxデータを保存
    normalization_refs = {
        'person_session_max_power': {},
        'person_session_min_power': {},
        'person_session_max_time': {},
        'person_session_min_time': {},
        'person_session_max_speed': {},
        'person_session_min_speed': {},
        'used_max_sample_indices': {}  # 正規化に使用したMaxデータのサンプルインデックス
    }
    
    # 使用したMaxデータのサンプルインデックスを記録するためのセット
    used_max_indices = set()
    
    # 個人別・セッション別に処理
    unique_persons = np.unique(person_ids)
    unique_sessions = np.unique(sessions)
    
    for person in unique_persons:
        for session in unique_sessions:
            # その個人のそのセッションのデータを取得
            person_session_mask = (person_ids == person) & (sessions == session)
            
            if not np.any(person_session_mask):
                continue  # このperson-sessionの組み合わせにデータがない
            
            person_session_features = features[person_session_mask]
            person_session_labels = y_label_names[person_session_mask]
            
            print(f"\n{person} Session {session}: {len(person_session_features)} samples")
            
            # Power正規化: Maxラベルの最大値とMax以外の最小値でmin-max正規化
            max_label_mask = person_session_labels == 'Max'
            non_max_mask = person_session_labels != 'Max'
            
            if np.any(max_label_mask) and np.any(non_max_mask):
                # Maxラベルの最大値を取得
                max_indices_in_person_session = np.where(max_label_mask)[0]
                max_power_values = person_session_features[max_label_mask, 0]
                max_power_idx_in_subset = np.argmax(max_power_values)
                max_power_value = max_power_values[max_power_idx_in_subset]
                
                # 元のデータセット内でのサンプルインデックスを取得
                global_sample_idx = sample_indices[person_session_mask][max_indices_in_person_session[max_power_idx_in_subset]]
                used_max_indices.add(global_sample_idx)
                
                # Max以外の最小値を取得
                non_max_power_values = person_session_features[non_max_mask, 0]
                min_power_value = np.min(non_max_power_values)
                
                # Power正規化範囲: Max以外の最小値 から Maxの最大値まで
                max_power = max_power_value
                min_power = min_power_value
                
                print(f"  Power normalization range: [{min_power:.2f}, {max_power:.2f}]")
                print(f"  Using Max with largest Power: {max_power:.2f} (sample_index: {global_sample_idx})")
                print(f"  Using min from non-Max: {min_power:.2f} ({len(non_max_power_values)} non-Max samples)")
                
            elif np.any(max_label_mask):
                # Maxラベルしかない場合
                max_indices_in_person_session = np.where(max_label_mask)[0]
                max_power_values = person_session_features[max_label_mask, 0]
                max_power_idx_in_subset = np.argmax(max_power_values)
                max_power_value = max_power_values[max_power_idx_in_subset]
                
                global_sample_idx = sample_indices[person_session_mask][max_indices_in_person_session[max_power_idx_in_subset]]
                used_max_indices.add(global_sample_idx)
                
                max_power = max_power_value
                min_power = np.min(max_power_values)  # Maxラベルの最小値を使用
                print(f"  Only Max labels, using Max range: [{min_power:.2f}, {max_power:.2f}]")
                
            else:
                # Maxラベルがない場合は全Powerデータを使用
                max_power = np.max(person_session_features[:, 0])
                min_power = np.min(person_session_features[:, 0])
                print(f"  No Max label, using all Power: [{min_power:.2f}, {max_power:.2f}]")
            
            # Time/Speed正規化: Maxラベル以外のデータを使用
            non_max_mask = person_session_labels != 'Max'
            if np.any(non_max_mask):
                non_max_features = person_session_features[non_max_mask]
                max_time = np.max(non_max_features[:, 1])
                min_time = np.min(non_max_features[:, 1])
                max_speed = np.max(non_max_features[:, 2])
                min_speed = np.min(non_max_features[:, 2])
                print(f"  Non-Max Time: [{min_time:.2f}, {max_time:.2f}] ({len(non_max_features)} samples)")
                print(f"  Non-Max Speed: [{min_speed:.2f}, {max_speed:.2f}]")
            else:
                # Maxラベルしかない場合は全データを使用
                max_time = np.max(person_session_features[:, 1])
                min_time = np.min(person_session_features[:, 1])
                max_speed = np.max(person_session_features[:, 2])
                min_speed = np.min(person_session_features[:, 2])
                print(f"  Only Max labels, using all data for Time/Speed")
            
            # 正規化参照値を保存
            key = f"{person}_session_{session}"
            normalization_refs['person_session_max_power'][key] = max_power
            normalization_refs['person_session_min_power'][key] = min_power
            normalization_refs['person_session_max_time'][key] = max_time
            normalization_refs['person_session_min_time'][key] = min_time
            normalization_refs['person_session_max_speed'][key] = max_speed
            normalization_refs['person_session_min_speed'][key] = min_speed
            
            # 正規化を適用
            power_range = max_power - min_power
            time_range = max_time - min_time
            speed_range = max_speed - min_speed
            
            # Power正規化 (0-1)
            if power_range > 0:
                normalized_features[person_session_mask, 0] = ((features[person_session_mask, 0] - min_power) / power_range)
            else:
                normalized_features[person_session_mask, 0] = 0.5
            
            # Time正規化 (0-1)
            if time_range > 0:
                normalized_features[person_session_mask, 1] = ((features[person_session_mask, 1] - min_time) / time_range)
            else:
                normalized_features[person_session_mask, 1] = 0.5
            
            # Speed正規化 (0-1) - 逆転させる（小さい値ほど速い）
            if speed_range > 0:
                normalized_features[person_session_mask, 2] = 1.0 - ((features[person_session_mask, 2] - min_speed) / speed_range)
            else:
                normalized_features[person_session_mask, 2] = 0.5
            
            # 数値の安定性のため、0-1の範囲に収める
            normalized_features[person_session_mask] = np.clip(normalized_features[person_session_mask], 0.0, 1.0)
            
            # デバッグ: 正規化後の実際の範囲を確認
            person_session_norm = normalized_features[person_session_mask]
            print(f"  After normalization:")
            # Powerの範囲をMax/非Max別々に表示
            power_values = person_session_norm[:, 0]
            max_power_norm = power_values[person_session_labels == 'Max'] if np.any(person_session_labels == 'Max') else []
            non_max_power_norm = power_values[person_session_labels != 'Max'] if np.any(person_session_labels != 'Max') else []
            
            print(f"    Power overall: [{np.min(power_values):.6f}, {np.max(power_values):.6f}]")
            if len(max_power_norm) > 0:
                print(f"    Power (Max): [{np.min(max_power_norm):.6f}, {np.max(max_power_norm):.6f}]")
            if len(non_max_power_norm) > 0:
                print(f"    Power (non-Max): [{np.min(non_max_power_norm):.6f}, {np.max(non_max_power_norm):.6f}]")
            print(f"    Time: [{np.min(person_session_norm[:, 1]):.6f}, {np.max(person_session_norm[:, 1]):.6f}]")
            print(f"    Speed: [{np.min(person_session_norm[:, 2]):.6f}, {np.max(person_session_norm[:, 2]):.6f}]")
    
    # 使用したMaxデータのマスクを作成
    used_max_mask = np.array([idx in used_max_indices for idx in sample_indices])
    
    print(f"\nUsed {len(used_max_indices)} Max samples for normalization out of {np.sum(y_label_names == 'Max')} total Max samples")
    print(f"Used Max sample indices: {sorted(used_max_indices)}")
    
    return {
        'features': normalized_features,
        'y_label_name': y_label_names,
        'person_ids': person_ids,
        'sessions': sessions,
        'sample_indices': sample_indices,
        'raw_features': features,
        'normalization_refs': normalization_refs,
        'used_max_mask': used_max_mask,  # 正規化に使用したMaxデータのマスク
        'used_max_indices': used_max_indices  # 正規化に使用したMaxデータのサンプルインデックス
    }

def load_questionnaire_data(questionnaire_dir='../questionaire_result'):
    """
    アンケートデータを読み込み（convert_features_3_6.pyと同じ）
    
    Args:
        questionnaire_dir: アンケートデータディレクトリのパス
        
    Returns:
        dict: 被験者ID -> {オノマトペ: {power, speed, time}}
    """
    print(f"Loading questionnaire data from {questionnaire_dir}...")
    
    if not os.path.exists(questionnaire_dir):
        print(f"Warning: Questionnaire directory not found at {questionnaire_dir}")
        return {}
    
    json_files = glob.glob(os.path.join(questionnaire_dir, "*.json"))
    
    if len(json_files) == 0:
        print(f"Warning: No JSON files found in {questionnaire_dir}")
        return {}
    
    questionnaire_data = {}
    
    for file in json_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            responses = data.get('responses', {})
            participant_id = responses.get('【あなたについて】_被験者番号', 'Unknown')
            
            onomatopoeia_data = {}
            
            for key, value in responses.items():
                if "【オノマトペについて】" in key:
                    try:
                        if 'パワー' in key and '： 「' in key:
                            onomatopoeia = key.split('： 「')[1].split('」')[0]
                            if onomatopoeia not in onomatopoeia_data:
                                onomatopoeia_data[onomatopoeia] = {}
                            onomatopoeia_data[onomatopoeia]['power'] = float(value)
                        
                        elif 'スピード' in key and '： 「' in key:
                            onomatopoeia = key.split('： 「')[1].split('」')[0]
                            if onomatopoeia not in onomatopoeia_data:
                                onomatopoeia_data[onomatopoeia] = {}
                            onomatopoeia_data[onomatopoeia]['speed'] = float(value)
                        
                        elif '時間の長さ（持続性）' in key and '： 「' in key:
                            onomatopoeia = key.split('： 「')[1].split('」')[0]
                            if onomatopoeia not in onomatopoeia_data:
                                onomatopoeia_data[onomatopoeia] = {}
                            onomatopoeia_data[onomatopoeia]['time'] = float(value)
                    except (IndexError, ValueError) as e:
                        continue
            
            questionnaire_data[participant_id] = onomatopoeia_data
            print(f"Loaded {len(onomatopoeia_data)} onomatopoeia for participant {participant_id}")
            
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            continue
    
    print(f"Loaded questionnaire data for {len(questionnaire_data)} participants")
    return questionnaire_data

def normalize_questionnaire_data(questionnaire_data):
    """
    アンケートデータを正規化（convert_features_3_6.pyと同じ）
    
    - Power: 100で割る (0-1)
    - Speed/Time: 除外されていないオノマトペの最大値で正規化 (0-1)
    
    Args:
        questionnaire_data: アンケートデータ
        
    Returns:
        dict: 正規化済みアンケートデータ
    """
    print("Normalizing questionnaire data...")
    
    normalized_data = {}
    
    for participant_id, onomatopoeia_data in questionnaire_data.items():
        if len(onomatopoeia_data) == 0:
            normalized_data[participant_id] = {}
            continue
        
        # 正規化処理
        normalized_onomatopoeia = {}
        
        # Speed/Timeの最大値を計算（全オノマトペから）
        all_speeds = []
        all_times = []
        
        for onomatopoeia, features in onomatopoeia_data.items():
            if 'speed' in features:
                all_speeds.append(features['speed'])
            if 'time' in features:
                all_times.append(features['time'])
        
        max_speed = max(all_speeds) if all_speeds else 1.0
        max_time = max(all_times) if all_times else 1.0
        
        print(f"  {participant_id}: max_speed={max_speed}, max_time={max_time}")
        
        # 各オノマトペの正規化
        for onomatopoeia, features in onomatopoeia_data.items():
            normalized_features = {}
            
            # Power正規化: 100で割る
            if 'power' in features:
                normalized_features['power'] = features['power'] / 100.0
            
            # Speed正規化: 最大値で割る
            if 'speed' in features:
                normalized_features['speed'] = features['speed'] / max_speed
            
            # Time正規化: 最大値で割る
            if 'time' in features:
                normalized_features['time'] = features['time'] / max_time
            
            normalized_onomatopoeia[onomatopoeia] = normalized_features
        
        normalized_data[participant_id] = normalized_onomatopoeia
    
    print("Questionnaire data normalization completed")
    return normalized_data

def save_annotation_features(sensor_data, questionnaire_data, output_dir='plot_3_features_annotation', normalized_dir='normalized_datas'):
    """
    アノテーション由来の特徴量データを保存
    
    Args:
        sensor_data: 正規化済みセンサー特徴量データ
        questionnaire_data: 正規化済みアンケートデータ
        output_dir: 出力ディレクトリ
        normalized_dir: 正規化データ保存ディレクトリ
    """
    print(f"Saving annotation-based feature data to {output_dir}/...")
    print(f"Saving normalized data to {normalized_dir}/...")
    
    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(normalized_dir, exist_ok=True)
    
    # センサーデータを保存
    sensor_output = {
        'features': sensor_data['features'],
        'y_label_name': sensor_data['y_label_name'],
        'person_ids': sensor_data['person_ids'],
        'sessions': sensor_data['sessions'],
        'sample_indices': sensor_data['sample_indices'],
        'raw_features': sensor_data['raw_features'],
        'normalization_refs': sensor_data['normalization_refs'],
        'used_max_mask': sensor_data['used_max_mask'],
        'used_max_indices': sensor_data['used_max_indices']
    }
    
    # plot_3_features_annotationディレクトリに保存
    sensor_path = os.path.join(output_dir, '3feature_annotation_sensor.bin')
    with open(sensor_path, 'wb') as f:
        pickle.dump(sensor_output, f)
    
    # normalized_datasディレクトリにも保存
    sensor_normalized_path = os.path.join(normalized_dir, 'annotation_sensor_normalized.bin')
    with open(sensor_normalized_path, 'wb') as f:
        pickle.dump(sensor_output, f)
    
    print(f"Sensor feature data saved to {sensor_path}")
    print(f"Sensor normalized data saved to {sensor_normalized_path}")
    print(f"  Features shape: {sensor_data['features'].shape}")
    print(f"  Unique persons: {len(np.unique(sensor_data['person_ids']))}")
    print(f"  Unique labels: {len(np.unique(sensor_data['y_label_name']))}")
    
    # アンケートデータを保存
    questionnaire_path = os.path.join(output_dir, '3feature_annotation_questionnaire.bin')
    with open(questionnaire_path, 'wb') as f:
        pickle.dump(questionnaire_data, f)
    
    # normalized_datasディレクトリにも保存
    questionnaire_normalized_path = os.path.join(normalized_dir, 'annotation_questionnaire_normalized.bin')
    with open(questionnaire_normalized_path, 'wb') as f:
        pickle.dump(questionnaire_data, f)
    
    print(f"Questionnaire data saved to {questionnaire_path}")
    print(f"Questionnaire normalized data saved to {questionnaire_normalized_path}")
    print(f"  Participants: {len(questionnaire_data)}")
    
    # 正規化情報の詳細をnormalized_datasに保存
    save_normalization_details(sensor_data, questionnaire_data, normalized_dir)
    
    # 統計サマリーを出力
    save_summary_stats(sensor_data, questionnaire_data, output_dir)

def save_normalization_details(sensor_data, questionnaire_data, normalized_dir):
    """
    正規化情報の詳細を保存
    
    Args:
        sensor_data: センサーデータ
        questionnaire_data: アンケートデータ
        normalized_dir: 正規化データ保存ディレクトリ
    """
    details_path = os.path.join(normalized_dir, 'normalization_details.txt')
    
    with open(details_path, 'w', encoding='utf-8') as f:
        f.write("=== Annotation-based Normalization Details ===\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 正規化ルールの説明
        f.write("【正規化ルール】\n")
        f.write("1. Power: 最小値=Max以外の最小値、最大値=Maxラベルの最大値でmin-max正規化\n")
        f.write("2. Time: その個人のそのSessionのMaxラベル以外のデータを使用してmin-max正規化 (0-1)\n")
        f.write("3. Speed: その個人のそのSessionのMaxラベル以外のデータを使用してmin-max正規化 (0-1)\n")
        f.write("※Speedは逆転させる（小さい値ほど速い）\n\n")
        
        # 使用したMaxデータ情報
        f.write("【使用したMaxデータ】\n")
        f.write(f"全Maxサンプル数: {np.sum(sensor_data['y_label_name'] == 'Max')}\n")
        f.write(f"正規化に使用: {len(sensor_data['used_max_indices'])}\n")
        f.write(f"使用したサンプルインデックス: {sorted(sensor_data['used_max_indices'])}\n\n")
        
        # 正規化範囲情報
        f.write("【正規化範囲情報】\n")
        normalization_refs = sensor_data['normalization_refs']
        for key in sorted(normalization_refs['person_session_max_power'].keys()):
            max_power = normalization_refs['person_session_max_power'][key]
            min_power = normalization_refs['person_session_min_power'][key]
            max_time = normalization_refs['person_session_max_time'][key]
            min_time = normalization_refs['person_session_min_time'][key]
            max_speed = normalization_refs['person_session_max_speed'][key]
            min_speed = normalization_refs['person_session_min_speed'][key]
            
            f.write(f"{key}:\n")
            f.write(f"  Power: [{min_power:.4f}, {max_power:.4f}]\n")
            f.write(f"  Time: [{min_time:.4f}, {max_time:.4f}]\n")
            f.write(f"  Speed: [{min_speed:.4f}, {max_speed:.4f}]\n")
        
        f.write("\n")
        
        # 正規化後の統計
        features = sensor_data['features']
        f.write("【正規化後統計】\n")
        f.write(f"Power - min: {features[:, 0].min():.6f}, max: {features[:, 0].max():.6f}, mean: {features[:, 0].mean():.6f}\n")
        f.write(f"Time  - min: {features[:, 1].min():.6f}, max: {features[:, 1].max():.6f}, mean: {features[:, 1].mean():.6f}\n")
        f.write(f"Speed - min: {features[:, 2].min():.6f}, max: {features[:, 2].max():.6f}, mean: {features[:, 2].mean():.6f}\n")
    
    print(f"Normalization details saved to {details_path}")

def save_summary_stats(sensor_data, questionnaire_data, output_dir):
    """統計サマリーを保存"""
    
    summary_path = os.path.join(output_dir, 'annotation_summary.txt')
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=== Annotation-based Feature Data Summary ===\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # センサーデータ統計
        f.write("【センサーデータ統計】\n")
        f.write(f"Total samples: {len(sensor_data['features'])}\n")
        f.write(f"Features shape: {sensor_data['features'].shape}\n")
        f.write(f"Unique persons: {len(np.unique(sensor_data['person_ids']))}\n")
        f.write(f"Unique labels: {len(np.unique(sensor_data['y_label_name']))}\n")
        f.write(f"Sessions: {sorted(np.unique(sensor_data['sessions']))}\n\n")
        
        # 被験者別統計
        f.write("【被験者別サンプル数】\n")
        unique_persons = np.unique(sensor_data['person_ids'])
        for person in unique_persons:
            count = np.sum(sensor_data['person_ids'] == person)
            f.write(f"  {person}: {count} samples\n")
        f.write("\n")
        
        # オノマトペ別統計
        f.write("【オノマトペ別サンプル数】\n")
        unique_labels = np.unique(sensor_data['y_label_name'])
        for label in unique_labels:
            count = np.sum(sensor_data['y_label_name'] == label)
            f.write(f"  {label}: {count} samples\n")
        f.write("\n")
        
        # 特徴量統計
        features = sensor_data['features']
        f.write("【正規化済み特徴量統計】\n")
        f.write(f"Power - min: {features[:, 0].min():.4f}, max: {features[:, 0].max():.4f}, mean: {features[:, 0].mean():.4f}\n")
        f.write(f"Time  - min: {features[:, 1].min():.4f}, max: {features[:, 1].max():.4f}, mean: {features[:, 1].mean():.4f}\n")
        f.write(f"Speed - min: {features[:, 2].min():.4f}, max: {features[:, 2].max():.4f}, mean: {features[:, 2].mean():.4f}\n\n")
        
        # アンケートデータ統計
        f.write("【アンケートデータ統計】\n")
        f.write(f"Participants: {len(questionnaire_data)}\n")
        for participant_id, data in questionnaire_data.items():
            f.write(f"  {participant_id}: {len(data)} onomatopoeia\n")
    
    print(f"Summary statistics saved to {summary_path}")

def get_common_onomatopoeia(sensor_data, questionnaire_data):
    """
    センサーデータとアンケートデータの共通オノマトペを取得
    
    Args:
        sensor_data: センサー特徴量データ
        questionnaire_data: アンケートデータ
        
    Returns:
        list: 共通オノマトペのリスト
    """
    # センサーデータのオノマトペ
    sensor_labels = set(sensor_data['y_label_name'])
    
    # アンケートデータのオノマトペ（全参加者分）
    questionnaire_labels = set()
    for participant_data in questionnaire_data.values():
        questionnaire_labels.update(participant_data.keys())
    
    # 共通部分を取得
    common = sensor_labels.intersection(questionnaire_labels)
    
    # Maxラベルは除外しない（個別セッションプロットでは表示する）
    return sorted(list(common))

def plot_sensor_questionnaire_combined(sensor_data, questionnaire_data, output_dir='plot_3_features_annotation'):
    """
    センサーデータとアンケートデータを結合してプロット
    """
    print(f"Creating combined sensor-questionnaire plots in {output_dir}/...")
    
    # デバッグ: 入力データの範囲を確認
    print(" ========= DEBUG: sensor_data features range:")
    features = sensor_data['features']
    print(f"  Power: [{np.min(features[:, 0]):.6f}, {np.max(features[:, 0]):.6f}]")
    print(f"  Time: [{np.min(features[:, 1]):.6f}, {np.max(features[:, 1]):.6f}]")
    print(f"  Speed: [{np.min(features[:, 2]):.6f}, {np.max(features[:, 2]):.6f}]")
    
    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    print("Output directory created successfully")
    
    # 共通オノマトペを取得
    print("Getting common onomatopoeia...")
    common_onomatopoeia = get_common_onomatopoeia(sensor_data, questionnaire_data)
    print(f"Common onomatopoeia found: {len(common_onomatopoeia)} items")
    print(f"Common onomatopoeia: {common_onomatopoeia}")
    
    # 正規化済みデータを使用（正規化に使用したMaxラベルのみ含む）
    # 使用しなかったMaxラベルを除外
    used_max_mask = sensor_data['used_max_mask']
    non_max_mask = sensor_data['y_label_name'] != 'Max'
    plot_mask = non_max_mask | used_max_mask  # 非Max + 使用したMax
    
    features_filtered = sensor_data['features'][plot_mask]
    y_label_name_filtered = sensor_data['y_label_name'][plot_mask]
    person_ids_filtered = sensor_data['person_ids'][plot_mask]
    sessions_filtered = sensor_data['sessions'][plot_mask]
    
    total_max = np.sum(sensor_data['y_label_name'] == 'Max')
    used_max = len(sensor_data['used_max_indices'])
    filtered_max = np.sum(y_label_name_filtered == 'Max')
    
    print(f"Total Max samples: {total_max}, Used for normalization: {used_max}, Included in plots: {filtered_max}")
    print(f"Total samples for plotting: {len(features_filtered)} (excluded {total_max - used_max} unused Max samples)")
    
    # 各被験者の各セッションについて処理
    unique_persons = np.unique(person_ids_filtered)
    print(f"Processing {len(unique_persons)} participants: {list(unique_persons)}")
    
    for person in unique_persons:
        print(f"Processing participant: {person}")
        person_mask = person_ids_filtered == person
        person_features = features_filtered[person_mask]
        person_labels = y_label_name_filtered[person_mask]
        person_sessions = sessions_filtered[person_mask]
        
        if len(person_features) == 0:
            print(f"  No data for {person}, skipping")
            continue
        
        # 各セッション用のプロットを作成
        unique_sessions = np.unique(person_sessions)
        for session in unique_sessions:
            session_mask = person_sessions == session
            session_features = person_features[session_mask]
            session_labels = person_labels[session_mask]
            
            if len(session_features) == 0:
                continue
            
            # デバッグ出力を追加
            max_count = np.sum(session_labels == 'Max')
            non_max_count = np.sum(session_labels != 'Max')
            print(f"\nDebug - {person} Session {session}:")
            print(f"  Features shape: {session_features.shape} (Max: {max_count}, Non-Max: {non_max_count})")
            print(f"  Power range: [{np.min(session_features[:, 0]):.4f}, {np.max(session_features[:, 0]):.4f}]")
            print(f"  Time range: [{np.min(session_features[:, 1]):.4f}, {np.max(session_features[:, 1]):.4f}]")
            print(f"  Speed range: [{np.min(session_features[:, 2]):.4f}, {np.max(session_features[:, 2]):.4f}]")
            
            plot_individual_session_combined(person, session, session_features, session_labels,
                                          questionnaire_data, output_dir)
        
        # 全セッション結合プロットを作成
        plot_all_sessions_combined(person, person_features, person_labels, person_sessions, 
                                 questionnaire_data, output_dir)
        
        # セントロイドプロットを作成
        plot_sessions_centroid(person, person_features, person_labels, person_sessions, 
                             questionnaire_data, output_dir)

def plot_individual_session_combined(person, session, session_features, session_labels, 
                                   questionnaire_data, output_dir):
    """個別セッションの結合プロット（6分割：センサー3＋アンケート3）"""
    
    # アンケートデータから対応する被験者IDを取得
    # personはint型の可能性があるので、str型に変換してから取得
    participant_id = PERSON_ID_MAPPING.get(int(person), str(person))
    if participant_id not in questionnaire_data:
        print(f"  Warning: No questionnaire data for {person} (mapped to {participant_id})")
        return
    
    questionnaire_person_data = questionnaire_data[participant_id]
    
    # 2x3のサブプロット (上段：センサー、下段：アンケート)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # ユニークなラベルを取得
    unique_labels = np.unique(session_labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    label_color_map = dict(zip(unique_labels, colors))
    
    label_f_size = 10
    # === 上段：センサーデータ ===
    
    # センサー: Power vs Time
    ax1 = axes[0, 0]
    texts = []
    for i, (feat, label) in enumerate(zip(session_features, session_labels)):
        color = label_color_map[label]
        # Maxラベルは少し大きいマーカーで表示
        if label == 'Max':
            ax1.scatter(feat[0], feat[1], c=[color], s=30, alpha=0.9, marker="o", edgecolors='black', linewidth=1)
        else:
            ax1.scatter(feat[0], feat[1], c=[color], s=30, alpha=0.8, marker="o", edgecolors='black', linewidth=0.5)
        # オノマトペ名を点の上に表示（枠なし）
        texts.append(ax1.text(feat[0], feat[1], label, ha='center', va='bottom', fontsize=label_f_size, color='black'))
    
    adjust_text(texts,
           ax=ax1,  # 明示的にaxを指定
           expand_points=(50.0, 50.0),      # 基本的な離れ具合
                force_points=(20.0, 23.0),       # ポイントからの反発力を弱く
                force_text=(20.0, 23.0),         # テキスト同士の反発力も弱く
                force_objects=(20.0, 23.0),      # オブジェクト間の反発力
                arrowprops=dict(
                    arrowstyle='->',
                    color='gray',
                    lw=0.5,
                    alpha=0.6
                ),
                only_move={'points':'xy', 'text':'xy', 'objects':'xy'},
                add_objects=None,              # 追加のオブジェクト
                avoid_self=True,              # 自身との衝突を避ける
                avoid_points=True,            # ポイントとの衝突を避ける
                avoid_text=True,              # テキスト同士の衝突を避ける
                text_from_text=True,          # テキスト同士の位置関係を考慮
                text_from_points=True,        # ポイントからの相対位置を保持
                save_steps=False,
                lim=100,                      # 探索範囲の制限
                precision=0.01,                # 位置調整の精度
                maxiter=3000,                 # 最大反復回数を大幅に増やす
                va='center',
                ha='center'
            )

    ax1.set_xlabel('Power')
    ax1.set_ylabel('Duration')
    # ax1.set_title(f'{person}: Sensor (Session {session})')
    ax1.grid(True, alpha=0.3)
    # Power軸は自動スケール、Timeは0-1範囲
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_xlim(-0.1, 1.1)
    
    # センサー: Power vs Speed
    texts = []
    ax2 = axes[0, 1]
    for i, (feat, label) in enumerate(zip(session_features, session_labels)):
        color = label_color_map[label]
        # Maxラベルは少し大きいマーカーで表示
        if label == 'Max':
            ax2.scatter(feat[0], feat[2], c=[color], s=30, alpha=0.9, marker="o", edgecolors='black', linewidth=1)
        else:
            ax2.scatter(feat[0], feat[2], c=[color], s=30, alpha=0.8, marker="o", edgecolors='black', linewidth=0.5)
        # オノマトペ名を点の上に表示（枠なし）
        texts.append(ax2.text(feat[0], feat[2], label, ha='center', va='bottom', 
                fontsize=label_f_size, color='black'))
    ax2.set_xlabel('Power')
    ax2.set_ylabel('Speed')
    # ax2.set_title(f'{person}: Sensor (Session {session})')
    ax2.grid(True, alpha=0.3)
    # Power軸は自動スケール、Speedは0-1範囲
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_xlim(-0.1, 1.1)
    
    adjust_text(texts,
                ax=ax2,
                expand_points=(50.0, 50.0),      # 基本的な離れ具合
                force_points=(20.0, 23.0),       # ポイントからの反発力を弱く
                force_text=(20.0, 23.0),         # テキスト同士の反発力も弱く
                force_objects=(20.0, 23.0),      # オブジェクト間の反発力
                arrowprops=dict(
                    arrowstyle='->',
                    color='gray',
                    lw=0.5,
                    alpha=0.6
                ),
                only_move={'points':'xy', 'text':'xy', 'objects':'xy'},
                add_objects=None,              # 追加のオブジェクト
                avoid_self=True,              # 自身との衝突を避ける
                avoid_points=True,            # ポイントとの衝突を避ける
                avoid_text=True,              # テキスト同士の衝突を避ける
                text_from_text=True,          # テキスト同士の位置関係を考慮
                text_from_points=True,        # ポイントからの相対位置を保持
                save_steps=False,
                lim=100,                      # 探索範囲の制限
                precision=0.01,                # 位置調整の精度
                maxiter=3000,                 # 最大反復回数を大幅に増やす
                va='center',
                ha='center'
            )

    # センサー: Time vs Speed
    texts = []
    ax3 = axes[0, 2]
    for i, (feat, label) in enumerate(zip(session_features, session_labels)):
        color = label_color_map[label]
        # Maxラベルは少し大きいマーカーで表示
        if label == 'Max':
            ax3.scatter(feat[1], feat[2], c=[color], s=30, alpha=0.9, marker="o", edgecolors='black', linewidth=1)
        else:
            ax3.scatter(feat[1], feat[2], c=[color], s=30, alpha=0.8, marker="o", edgecolors='black', linewidth=0.5)
        # オノマトペ名を点の上に表示（枠なし）
        texts.append(ax3.text(feat[1], feat[2], label, ha='center', va='bottom', 
                fontsize=label_f_size, color='black'))
    
    adjust_text(texts,
           ax=ax3,  # 明示的にaxを指定
           expand_points=(50.0, 50.0),      # 基本的な離れ具合
                force_points=(20.0, 23.0),       # ポイントからの反発力を弱く
                force_text=(20.0, 23.0),         # テキスト同士の反発力も弱く
                force_objects=(20.0, 23.0),      # オブジェクト間の反発力
                arrowprops=dict(
                    arrowstyle='->',
                    color='gray',
                    lw=0.5,
                    alpha=0.6
                ),
                only_move={'points':'xy', 'text':'xy', 'objects':'xy'},
                add_objects=None,              # 追加のオブジェクト
                avoid_self=True,              # 自身との衝突を避ける
                avoid_points=True,            # ポイントとの衝突を避ける
                avoid_text=True,              # テキスト同士の衝突を避ける
                text_from_text=True,          # テキスト同士の位置関係を考慮
                text_from_points=True,        # ポイントからの相対位置を保持
                save_steps=False,
                lim=100,                      # 探索範囲の制限
                precision=0.01,                # 位置調整の精度
                maxiter=3000,                 # 最大反復回数を大幅に増やす
                va='center',
                ha='center'
            )

    ax3.set_xlabel('Duration')
    ax3.set_ylabel('Speed')
    # ax3.set_title(f'{person}: センサー Time vs Speed (Session {session})')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-0.1, 1.1)
    ax3.set_ylim(-0.1, 1.1)
    
    # === 下段：アンケートデータ ===
    
    # アンケート: Power vs Time
    ax4 = axes[1, 0]
    texts = []
    for label in unique_labels:
        if label in questionnaire_person_data:
            q_data = questionnaire_person_data[label]
            if 'power' in q_data and 'time' in q_data:
                color = label_color_map[label]
                ax4.scatter(q_data['power'], q_data['time'], c=[color], s=30, alpha=0.8, marker="o", edgecolors='black', linewidth=0.5)
                # オノマトペ名を点の上に表示（枠なし）
                texts.append(ax4.text(q_data['power'], q_data['time'], label, 
                        ha='center', va='bottom', fontsize=label_f_size, color='black'))
    
    adjust_text(texts,
           ax=ax4,  # 明示的にaxを指定
          expand_points=(50.0, 50.0),      # 基本的な離れ具合
                force_points=(20.0, 23.0),       # ポイントからの反発力を弱く
                force_text=(20.0, 23.0),         # テキスト同士の反発力も弱く
                force_objects=(20.0, 23.0),      # オブジェクト間の反発力
                arrowprops=dict(
                    arrowstyle='->',
                    color='gray',
                    lw=0.5,
                    alpha=0.6
                ),
                only_move={'points':'xy', 'text':'xy', 'objects':'xy'},
                add_objects=None,              # 追加のオブジェクト
                avoid_self=True,              # 自身との衝突を避ける
                avoid_points=True,            # ポイントとの衝突を避ける
                avoid_text=True,              # テキスト同士の衝突を避ける
                text_from_text=True,          # テキスト同士の位置関係を考慮
                text_from_points=True,        # ポイントからの相対位置を保持
                save_steps=False,
                lim=100,                      # 探索範囲の制限
                precision=0.01,                # 位置調整の精度
                maxiter=3000,                 # 最大反復回数を大幅に増やす
                va='center',
                ha='center'
            )
    ax4.set_xlabel('Power')
    ax4.set_ylabel('Duration')
    # ax4.set_title(f'{person}: アンケート Power vs Time')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-0.1, 1.1)
    ax4.set_ylim(-0.1, 1.1)
    
    # アンケート: Power vs Speed
    ax5 = axes[1, 1]
    texts = []
    for label in unique_labels:
        if label in questionnaire_person_data:
            q_data = questionnaire_person_data[label]
            if 'power' in q_data and 'speed' in q_data:
                color = label_color_map[label]
                ax5.scatter(q_data['power'], q_data['speed'], c=[color], s=30, alpha=0.8, marker="o", edgecolors='black', linewidth=0.5)
                # オノマトペ名を点の上に表示（枠なし）
                texts.append(ax5.text(q_data['power'], q_data['speed'], label, 
                        ha='center', va='bottom', fontsize=label_f_size, color='black'))
    
    adjust_text(texts, 
                ax=ax5,
                expand_points=(50.0, 50.0),      # 基本的な離れ具合
                force_points=(20.0, 23.0),       # ポイントからの反発力を弱く
                force_text=(20.0, 23.0),         # テキスト同士の反発力も弱く
                force_objects=(20.0, 23.0),      # オブジェクト間の反発力
                arrowprops=dict(
                    arrowstyle='->',
                    color='gray',
                    lw=0.5,
                    alpha=0.6
                ),
                only_move={'points':'xy', 'text':'xy', 'objects':'xy'},
                add_objects=None,              # 追加のオブジェクト
                avoid_self=True,              # 自身との衝突を避ける
                avoid_points=True,            # ポイントとの衝突を避ける
                avoid_text=True,              # テキスト同士の衝突を避ける
                text_from_text=True,          # テキスト同士の位置関係を考慮
                text_from_points=True,        # ポイントからの相対位置を保持
                save_steps=False,
                lim=100,                      # 探索範囲の制限
                precision=0.01,                # 位置調整の精度
                maxiter=3000,                 # 最大反復回数を大幅に増やす
                va='center',
                ha='center'
            )
    ax5.set_xlabel('Power')
    ax5.set_ylabel('Speed')
    # ax5.set_title(f'{person}: アンケート Power vs Speed')
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(-0.1, 1.1)
    ax5.set_ylim(-0.1, 1.1)
    
    # アンケート: Time vs Speed
    texts = []
    ax6 = axes[1, 2]
    for label in unique_labels:
        if label in questionnaire_person_data:
            q_data = questionnaire_person_data[label]
            if 'time' in q_data and 'speed' in q_data:
                color = label_color_map[label]
                ax6.scatter(q_data['time'], q_data['speed'], c=[color], s=30, alpha=0.8, marker="o", edgecolors='black', linewidth=0.5)
                # オノマトペ名を点の上に表示（枠なし）
                texts.append(ax6.text(q_data['time'], q_data['speed'], label, 
                        ha='center', va='bottom', fontsize=label_f_size, color='black'))
    
    adjust_text(texts,
            ax=ax6, 
           expand_points=(50.0, 50.0),      # 基本的な離れ具合
                force_points=(20.0, 23.0),       # ポイントからの反発力を弱く
                force_text=(20.0, 23.0),         # テキスト同士の反発力も弱く
                force_objects=(20.0, 23.0),      # オブジェクト間の反発力
                arrowprops=dict(
                    arrowstyle='->',
                    color='gray',
                    lw=0.5,
                    alpha=0.6
                ),
                only_move={'points':'xy', 'text':'xy', 'objects':'xy'},
                add_objects=None,              # 追加のオブジェクト
                avoid_self=True,              # 自身との衝突を避ける
                avoid_points=True,            # ポイントとの衝突を避ける
                avoid_text=True,              # テキスト同士の衝突を避ける
                text_from_text=True,          # テキスト同士の位置関係を考慮
                text_from_points=True,        # ポイントからの相対位置を保持
                save_steps=False,
                lim=100,                      # 探索範囲の制限
                precision=0.01,                # 位置調整の精度
                maxiter=3000,                 # 最大反復回数を大幅に増やす
                va='center',
                ha='center'
            )
    ax6.set_xlabel('Duration')
    ax6.set_ylabel('Speed')
    # ax6.set_title(f'{person}: アンケート Time vs Speed')
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(-0.1, 1.1)
    ax6.set_ylim(-0.1, 1.1)
    
    for ax in axes.flat:
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)

    #plt.tight_layout()
    #plt.tight_layout()
    
    # ファイル名を生成
    filename = f"combined_{person}_session_{session}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved session plot: {filename}")

def plot_all_sessions_combined(person, person_features, person_labels, person_sessions, 
                             questionnaire_data, output_dir):
    """全セッション結合プロット（6分割：センサー3＋アンケート3）"""
    
    # アンケートデータから対応する被験者IDを取得
    participant_id = PERSON_ID_MAPPING.get(int(person), str(person))
    if participant_id not in questionnaire_data:
        return
    
    questionnaire_person_data = questionnaire_data[participant_id]
    
    # 2x3のサブプロット (上段：センサー、下段：アンケート)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # ユニークなラベルとセッションを取得
    unique_labels = np.unique(person_labels)
    unique_sessions = np.unique(person_sessions)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    markers = ['o', ',', '^']  # セッション別マーカー
    
    label_color_map = dict(zip(unique_labels, colors))
    session_marker_map = dict(zip(unique_sessions, markers[:len(unique_sessions)]))
    
    # === 上段：センサーデータ ===
    
    # センサー: Power vs Time
    ax1 = axes[0, 0]
    for i, (feat, label, session) in enumerate(zip(person_features, person_labels, person_sessions)):
        color = label_color_map[label]
        marker = session_marker_map[session]
        ax1.scatter(feat[0], feat[1], c=[color], s=50, alpha=0.7, marker=marker)
        # オノマトペ名を点の上に表示（枠なし）
        ax1.text(feat[0], feat[1] + 0.03, label, ha='center', va='bottom', 
                fontsize=8, color='black')
    
    ax1.set_xlabel('Power')
    ax1.set_ylabel('Duration')
    #ax1.set_title(f'{person}: センサー Power vs Time (All Sessions)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1, 1.1)
    
    # センサー: Power vs Speed
    ax2 = axes[0, 1]
    for i, (feat, label, session) in enumerate(zip(person_features, person_labels, person_sessions)):
        color = label_color_map[label]
        marker = session_marker_map[session]
        ax2.scatter(feat[0], feat[2], c=[color], s=50, alpha=0.7, marker=marker)
        # オノマトペ名を点の上に表示（枠なし）
        ax2.text(feat[0], feat[2] + 0.03, label, ha='center', va='bottom', 
                fontsize=8, color='black')
    
    ax2.set_xlabel('Power')
    ax2.set_ylabel('Speed')
    #ax2.set_title(f'{person}: センサー Power vs Speed (All Sessions)')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(-0.1, 1.1)
    
    # センサー: Time vs Speed
    ax3 = axes[0, 2]
    for i, (feat, label, session) in enumerate(zip(person_features, person_labels, person_sessions)):
        color = label_color_map[label]
        marker = session_marker_map[session]
        ax3.scatter(feat[1], feat[2], c=[color], s=50, alpha=0.7, marker=marker)
        # オノマトペ名を点の上に表示（枠なし）
        ax3.text(feat[1], feat[2] + 0.03, label, ha='center', va='bottom', 
                fontsize=8, color='black')
    
    ax3.set_xlabel('Duration')
    ax3.set_ylabel('Speed')
    #ax3.set_title(f'{person}: センサー Time vs Speed (All Sessions)')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-0.1, 1.1)
    ax3.set_ylim(-0.1, 1.1)
    
    # === 下段：アンケートデータ ===
    
    # アンケート: Power vs Time
    ax4 = axes[1, 0]
    for label in unique_labels:
        if label in questionnaire_person_data:
            q_data = questionnaire_person_data[label]
            if 'power' in q_data and 'time' in q_data:
                color = label_color_map[label]
                ax4.scatter(q_data['power'], q_data['time'], c=[color], s=100, 
                          marker='D', alpha=0.9, edgecolors='black', linewidth=2)
                # オノマトペ名を点の上に表示（枠なし）
                ax4.text(q_data['power'], q_data['time'] + 0.03, label, 
                        ha='center', va='bottom', fontsize=8, color='black')
    
    ax4.set_xlabel('Power')
    ax4.set_ylabel('Duration')
    #ax4.set_title(f'{person}: アンケート Power vs Time')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-0.1, 1.1)
    ax4.set_ylim(-0.1, 1.1)
    
    # アンケート: Power vs Speed
    ax5 = axes[1, 1]
    for label in unique_labels:
        if label in questionnaire_person_data:
            q_data = questionnaire_person_data[label]
            if 'power' in q_data and 'speed' in q_data:
                color = label_color_map[label]
                ax5.scatter(q_data['power'], q_data['speed'], c=[color], s=100, 
                          marker='D', alpha=0.9, edgecolors='black', linewidth=2)
                # オノマトペ名を点の上に表示（枠なし）
                ax5.text(q_data['power'], q_data['speed'] + 0.03, label, 
                        ha='center', va='bottom', fontsize=8, color='black')
    
    ax5.set_xlabel('Power')
    ax5.set_ylabel('Speed')
    #ax5.set_title(f'{person}: アンケート Power vs Speed')
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(-0.1, 1.1)
    ax5.set_ylim(-0.1, 1.1)
    
    # アンケート: Time vs Speed
    ax6 = axes[1, 2]
    for label in unique_labels:
        if label in questionnaire_person_data:
            q_data = questionnaire_person_data[label]
            if 'time' in q_data and 'speed' in q_data:
                color = label_color_map[label]
                ax6.scatter(q_data['time'], q_data['speed'], c=[color], s=100, 
                          marker='D', alpha=0.9, edgecolors='black', linewidth=2)
                # オノマトペ名を点の上に表示（枠なし）
                ax6.text(q_data['time'], q_data['speed'] + 0.03, label, 
                        ha='center', va='bottom', fontsize=8, color='black')
    
    ax6.set_xlabel('Duration')
    ax6.set_ylabel('Speed')
    # ax6.set_title(f'{person}: アンケート Time vs Speed')
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(-0.1, 1.1)
    ax6.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    
    # ファイル名を生成
    filename = f"combined_{person}_all_sessions.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved all-sessions plot: {filename}")

def plot_sessions_centroid(person, person_features, person_labels, person_sessions, 
                         questionnaire_data, output_dir):
    """セッション平均値（セントロイド）プロット（6分割：センサー3＋アンケート3）"""
    
    # アンケートデータから対応する被験者IDを取得
    participant_id = PERSON_ID_MAPPING.get(int(person), str(person))
    if participant_id not in questionnaire_data:
        return
    
    questionnaire_person_data = questionnaire_data[participant_id]
    
    # セッション別平均値を計算
    unique_labels = np.unique(person_labels)
    unique_sessions = np.unique(person_sessions)
    
    centroids = {}
    for label in unique_labels:
        centroids[label] = {}
        for session in unique_sessions:
            mask = (person_labels == label) & (person_sessions == session)
            if np.sum(mask) > 0:
                session_features = person_features[mask]
                centroids[label][session] = np.mean(session_features, axis=0)
    
    # 2x3のサブプロット (上段：センサー、下段：アンケート)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    label_color_map = dict(zip(unique_labels, colors))
    
    # === 上段：センサーセントロイド ===
    
    # センサー: Power vs Time (セントロイド)
    ax1 = axes[0, 0]
    for label in unique_labels:
        color = label_color_map[label]
        sessions_data = centroids[label]
        
        # セッション別セントロイドをプロット
        for session, centroid in sessions_data.items():
            ax1.scatter(centroid[0], centroid[1], c=[color], s=80, alpha=0.8, 
                       marker=f'${session}$')
            # オノマトペ名を点の上に表示（枠なし）
            ax1.text(centroid[0], centroid[1] + 0.03, label, ha='center', va='bottom', 
                    fontsize=8, color='black')
    
    ax1.set_xlabel('Power (正規化済み)')
    ax1.set_ylabel('Time (正規化済み)')
    ax1.set_title(f'{person}: センサー Power vs Time (Centroids)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1, 1.1)
    
    # センサー: Power vs Speed (セントロイド)
    ax2 = axes[0, 1]
    for label in unique_labels:
        color = label_color_map[label]
        sessions_data = centroids[label]
        
        # セッション別セントロイドをプロット
        for session, centroid in sessions_data.items():
            ax2.scatter(centroid[0], centroid[2], c=[color], s=80, alpha=0.8, 
                       marker=f'${session}$')
            # オノマトペ名を点の上に表示（枠なし）
            ax2.text(centroid[0], centroid[2] + 0.03, label, ha='center', va='bottom', 
                    fontsize=8, color='black')
    
    ax2.set_xlabel('Power (正規化済み)')
    ax2.set_ylabel('Speed (正規化済み)')
    ax2.set_title(f'{person}: センサー Power vs Speed (Centroids)')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(-0.1, 1.1)
    
    # センサー: Time vs Speed (セントロイド)
    ax3 = axes[0, 2]
    for label in unique_labels:
        color = label_color_map[label]
        sessions_data = centroids[label]
        
        # セッション別セントロイドをプロット
        for session, centroid in sessions_data.items():
            ax3.scatter(centroid[1], centroid[2], c=[color], s=80, alpha=0.8, 
                       marker=f'${session}$')
            # オノマトペ名を点の上に表示（枠なし）
            ax3.text(centroid[1], centroid[2] + 0.03, label, ha='center', va='bottom', 
                    fontsize=8, color='black')
    
    ax3.set_xlabel('Time (正規化済み)')
    ax3.set_ylabel('Speed (正規化済み)')
    ax3.set_title(f'{person}: センサー Time vs Speed (Centroids)')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-0.1, 1.1)
    ax3.set_ylim(-0.1, 1.1)
    
    # === 下段：アンケートデータ ===
    
    # アンケート: Power vs Time
    ax4 = axes[1, 0]
    for label in unique_labels:
        if label in questionnaire_person_data:
            q_data = questionnaire_person_data[label]
            if 'power' in q_data and 'time' in q_data:
                color = label_color_map[label]
                ax4.scatter(q_data['power'], q_data['time'], c=[color], s=120, 
                          marker='.', alpha=0.9, edgecolors='black', linewidth=2)
                # オノマトペ名を点の上に表示（枠なし）
                ax4.text(q_data['power'], q_data['time'] + 0.03, label, 
                        ha='center', va='bottom', fontsize=8, color='black')
    
    ax4.set_xlabel('Power (正規化済み)')
    ax4.set_ylabel('Time (正規化済み)')
    ax4.set_title(f'{person}: アンケート Power vs Time')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-0.1, 1.1)
    ax4.set_ylim(-0.1, 1.1)
    
    # アンケート: Power vs Speed
    ax5 = axes[1, 1]
    for label in unique_labels:
        if label in questionnaire_person_data:
            q_data = questionnaire_person_data[label]
            if 'power' in q_data and 'speed' in q_data:
                color = label_color_map[label]
                ax5.scatter(q_data['power'], q_data['speed'], c=[color], s=120, 
                          marker='.', alpha=0.9, edgecolors='black', linewidth=2)
                # オノマトペ名を点の上に表示（枠なし）
                ax5.text(q_data['power'], q_data['speed'] + 0.03, label, 
                        ha='center', va='bottom', fontsize=8, color='black')
    
    ax5.set_xlabel('Power (正規化済み)')
    ax5.set_ylabel('Speed (正規化済み)')
    ax5.set_title(f'{person}: アンケート Power vs Speed')
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(-0.1, 1.1)
    ax5.set_ylim(-0.1, 1.1)
    
    # アンケート: Time vs Speed
    ax6 = axes[1, 2]
    for label in unique_labels:
        if label in questionnaire_person_data:
            q_data = questionnaire_person_data[label]
            if 'time' in q_data and 'speed' in q_data:
                color = label_color_map[label]
                ax6.scatter(q_data['time'], q_data['speed'], c=[color], s=120, 
                          marker='.', alpha=0.9, edgecolors='black', linewidth=2)
                # オノマトペ名を点の上に表示（枠なし）
                ax6.text(q_data['time'], q_data['speed'] + 0.03, label, 
                        ha='center', va='bottom', fontsize=8, color='black')
    
    ax6.set_xlabel('Time (正規化済み)')
    ax6.set_ylabel('Speed (正規化済み)')
    ax6.set_title(f'{person}: アンケート Time vs Speed')
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(-0.1, 1.1)
    ax6.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    
    # ファイル名を生成
    filename = f"centroid_{person}_3session_avg.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    # セントロイドデータも保存
    centroid_data = {
        'person': person,
        'centroids': centroids,
        'questionnaire_data': questionnaire_person_data
    }
    
    centroid_filename = f"centroid_{person}_3session_avg.pkl"
    centroid_filepath = os.path.join(output_dir, centroid_filename)
    with open(centroid_filepath, 'wb') as f:
        pickle.dump(centroid_data, f)
    
    print(f"  Saved centroid plot and data: {filename}")

def plot_3d_features(feature_data, output_dir='plot_3_features_annotation'):
    """
    3つの特徴量を3D散布図でプロット
    
    Args:
        feature_data: 正規化済み特徴量データ
        output_dir: 出力ディレクトリ
    """
    print(f"Creating 3D feature plots in {output_dir}/...")
    
    # 出力ディレクトリを作成
    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    features = feature_data['features']
    y_label_name = feature_data['y_label_name']
    person_ids = feature_data['person_ids']
    
    # ユニークなラベルを取得
    unique_labels = np.unique(y_label_name)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    # 3D散布図を作成
    fig = plt.figure(figsize=(20, 15))
    
    # 全体の3D散布図
    ax1 = fig.add_subplot(221, projection='3d')
    
    for i, label in enumerate(unique_labels):
        mask = y_label_name == label
        ax1.scatter(features[mask, 0], features[mask, 1], features[mask, 2], 
                   c=[colors[i]], label=label, alpha=0.7, s=30)
    
    ax1.set_xlabel('Power (正規化済み)')
    ax1.set_ylabel('Time (正規化済み)')
    ax1.set_zlabel('Speed (正規化済み)')
    ax1.set_title('3D特徴量空間 (オノマトペ別)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # 人物別の3D散布図
    ax2 = fig.add_subplot(222, projection='3d')
    
    unique_persons = np.unique(person_ids)
    person_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_persons)))
    
    for i, person in enumerate(unique_persons):
        mask = person_ids == person
        ax2.scatter(features[mask, 0], features[mask, 1], features[mask, 2], 
                   c=[person_colors[i]], label=person, alpha=0.7, s=30)
    
    ax2.set_xlabel('Power (正規化済み)')
    ax2.set_ylabel('Time (正規化済み)')
    ax2.set_zlabel('Speed (正規化済み)')
    ax2.set_title('3D特徴量空間 (被験者別)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # 2D投影図 (Power vs Time)
    ax3 = fig.add_subplot(223)
    
    for i, label in enumerate(unique_labels):
        mask = y_label_name == label
        ax3.scatter(features[mask, 0], features[mask, 1], 
                   c=[colors[i]], label=label, alpha=0.7, s=30)
    
    ax3.set_xlabel('Power (正規化済み)')
    ax3.set_ylabel('Time (正規化済み)')
    ax3.set_title('2D投影 (Power vs Time)')
    ax3.grid(True, alpha=0.3)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # 2D投影図 (Power vs Speed)
    ax4 = fig.add_subplot(224)
    
    for i, label in enumerate(unique_labels):
        mask = y_label_name == label
        ax4.scatter(features[mask, 0], features[mask, 2], 
                   c=[colors[i]], label=label, alpha=0.7, s=30)
    
    ax4.set_xlabel('Power (正規化済み)')
    ax4.set_ylabel('Speed (正規化済み)')
    ax4.set_title('2D投影 (Power vs Speed)')
    ax4.grid(True, alpha=0.3)
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    
    # プロット保存
    plot_path = os.path.join(plot_dir, '3d_feature_plot.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"3D feature plot saved to {plot_path}")

def plot_annotation_details(annotation_data, output_dir='plot_3_features_annotation'):
    """
    アノテーション詳細を可視化
    
    Args:
        annotation_data: アノテーション結果データ
        output_dir: 出力ディレクトリ
    """
    print(f"Creating annotation detail plots in {output_dir}/...")
    
    # 出力ディレクトリを作成
    detail_dir = os.path.join(output_dir, 'annotation_details')
    os.makedirs(detail_dir, exist_ok=True)
    
    annotations = annotation_data['annotations']
    
    # サンプルをいくつか選んでプロット（最初の20個）
    sample_indices = list(range(min(20, len(annotations))))
    
    for idx in sample_indices:
        ann = annotations[idx]
        
        # 信号データは取得できないので、アノテーション情報のみをプロット
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # アノテーション情報を表示
        info_text = f"""Sample {ann['sample_index']}
被験者: {ann['person_id']}
オノマトペ: {ann['label_name']}
セッション: {ann['session']}

アノテーション範囲:
開始位置: {ann['start_pos']}
終了位置: {ann['end_pos']}
最大値位置: {ann['max_pos']}

抽出特徴量（生値）:
Power: {ann['power_raw']:.4f}
Time: {ann['time_raw']} サンプル
Speed: {ann['speed_raw']} サンプル

信号情報:
信号長: {ann['signal_length']} サンプル
タイムスタンプ: {ann['timestamp']}"""
        
        ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
                verticalalignment='top', fontfamily='monospace', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title(f'アノテーション詳細 - Sample {ann["sample_index"]}', fontsize=14)
        
        # ファイル名を生成
        person_safe = ann['person_id'].replace('.', '_')
        label_safe = ann['label_name'].replace('（', '_').replace('）', '_').replace('/', '_')
        filename = f"annotation_detail_sample{ann['sample_index']:04d}_{person_safe}_{label_safe}.png"
        filepath = os.path.join(detail_dir, filename)
        
        # プロット保存
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {len(sample_indices)} annotation detail plots to {detail_dir}/")

def plot_feature_distributions(feature_data, output_dir='plot_3_features_annotation'):
    """
    特徴量分布をプロット
    
    Args:
        feature_data: 正規化済み特徴量データ
        output_dir: 出力ディレクトリ
    """
    print(f"Creating feature distribution plots in {output_dir}/...")
    
    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    features = feature_data['features']
    y_label_name = feature_data['y_label_name']
    person_ids = feature_data['person_ids']
    
    # 特徴量分布プロット
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    feature_names = ['Power', 'Time', 'Speed']
    
    # 全体分布
    for i, feature_name in enumerate(feature_names):
        ax = axes[0, i]
        ax.hist(features[:, i], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel(f'{feature_name} (正規化済み)')
        ax.set_ylabel('頻度')
        ax.set_title(f'{feature_name}の分布（全体）')
        ax.grid(True, alpha=0.3)
    
    # 被験者別分布
    unique_persons = np.unique(person_ids)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_persons)))
    
    for i, feature_name in enumerate(feature_names):
        ax = axes[1, i]
        for j, person in enumerate(unique_persons):
            mask = person_ids == person
            ax.hist(features[mask, i], bins=20, alpha=0.6, 
                   color=colors[j], label=person, density=True)
        ax.set_xlabel(f'{feature_name} (正規化済み)')
        ax.set_ylabel('密度')
        ax.set_title(f'{feature_name}の分布（被験者別）')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # プロット保存
    dist_path = os.path.join(plot_dir, 'feature_distributions.png')
    plt.savefig(dist_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Feature distribution plot saved to {dist_path}")

def main():
    """メイン処理"""
    print("=== Annotation-based Feature Conversion ===")
    
    # アノテーション結果を読み込み
    annotation_data = load_annotation_results('../annotation/annotation_results/individual')
    if annotation_data is None:
        return
    
    # センサー特徴量を正規化
    normalized_sensor_data = normalize_annotation_features(annotation_data)
    
    # アンケートデータを読み込み・正規化
    questionnaire_data = load_questionnaire_data('../questionaire_result')
    normalized_questionnaire_data = normalize_questionnaire_data(questionnaire_data)
    
    # 特徴量データを保存
    save_annotation_features(normalized_sensor_data, normalized_questionnaire_data)
    
    # プロット作成
    print("\n=== Creating visualization plots ===")
    plot_sensor_questionnaire_combined(normalized_sensor_data, normalized_questionnaire_data)
    
    print("\n=== Processing completed successfully! ===")
    print("Generated files:")
    print("  - plot_3_features_annotation/3feature_annotation_sensor.bin")
    print("  - plot_3_features_annotation/3feature_annotation_questionnaire.bin")
    print("  - plot_3_features_annotation/annotation_summary.txt")
    print("  - plot_3_features_annotation/combined_*.png (セッション別プロット)")
    print("  - plot_3_features_annotation/centroid_*.png (セントロイドプロット)")
    print("  - plot_3_features_annotation/centroid_*.pkl (セントロイドデータ)")

if __name__ == "__main__":
    main()