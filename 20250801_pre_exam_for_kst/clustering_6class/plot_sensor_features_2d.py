#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
センサー特徴量の2Dプロット作成
アノテーションから抽出した特徴量をセッション別・人別にプロット
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from pathlib import Path
import japanize_matplotlib

plt.rcParams['font.family'] = ['Times New Roman', 'IPAexGothic']
plt.rcParams['font.sans-serif'] = ['IPAexGothic']
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 16

def load_sensor_features(feature_path='plot_3_features_annotation/3feature_annotation_sensor.bin'):
    """センサー特徴量データを読み込み"""
    print(f"Loading sensor features from {feature_path}...")
    
    try:
        with open(feature_path, 'rb') as f:
            sensor_data = pickle.load(f)
        
        print(f"Loaded sensor data successfully!")
        print(f"  Features shape: {sensor_data['features'].shape}")
        print(f"  Unique persons: {len(np.unique(sensor_data['person_ids']))}")
        print(f"  Unique sessions: {sorted(np.unique(sensor_data['sessions']))}")
        print(f"  Unique labels: {len(np.unique(sensor_data['y_label_name']))}")
        
        return sensor_data
        
    except FileNotFoundError:
        print(f"Error: Feature file not found at {feature_path}")
        print("Please run convert_features_annotation.py first.")
        return None
    except Exception as e:
        print(f"Error loading sensor features: {e}")
        return None

def plot_session_features_by_person(sensor_data, output_base_dir='check_features'):
    """セッション別・人別に特徴量をプロット"""
    
    features = sensor_data['features']
    y_label_names = sensor_data['y_label_name']
    person_ids = sensor_data['person_ids']
    sessions = sensor_data['sessions']
    
    # 被験者名マッピング
    person_name_mapping = {
        1: 'inamura',
        2: 'utsumi', 
        3: 'kawano',
        4: 'hatanaka',
        5: 'watanabe',
        6: 'kikuchi',
        8: 'okamoto'
    }
    
    unique_sessions = sorted(np.unique(sessions))
    unique_persons = sorted(np.unique(person_ids))
    
    print(f"Creating plots for {len(unique_sessions)} sessions and {len(unique_persons)} persons...")
    
    for session in unique_sessions:
        print(f"\nProcessing Session {session}...")
        
        # セッション用ディレクトリを作成
        session_dir = os.path.join(output_base_dir, f'session_{session:02d}')
        os.makedirs(session_dir, exist_ok=True)
        
        for person_id in unique_persons:
            person_name = person_name_mapping.get(person_id, f'person_{person_id}')
            print(f"  Processing {person_name} (ID: {person_id})...")
            
            # 該当するデータを抽出
            mask = (person_ids == person_id) & (sessions == session)
            
            if not np.any(mask):
                print(f"    No data for {person_name} in session {session}")
                continue
            
            person_features = features[mask]
            person_labels = y_label_names[mask]
            
            # プロットを作成
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # ユニークなラベルを取得してカラーマップを作成
            unique_labels = np.unique(person_labels)
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
            label_color_map = dict(zip(unique_labels, colors))
            
            # 1. Power vs Duration
            ax1 = axes[0]
            for i, (feat, label) in enumerate(zip(person_features, person_labels)):
                color = label_color_map[label]
                if label == 'Max':
                    ax1.scatter(feat[0], feat[1], c=[color], s=50, alpha=0.9, 
                              marker="o", edgecolors='black', linewidth=1, label=label if i == 0 or label not in [axes[0].get_legend_handles_labels()[1][j] for j in range(len(axes[0].get_legend_handles_labels()[1]))] else "")
                else:
                    ax1.scatter(feat[0], feat[1], c=[color], s=40, alpha=0.8, 
                              marker="o", edgecolors='black', linewidth=0.5, label=label if label not in [h.get_label() for h in ax1.get_children() if hasattr(h, 'get_label')] else "")
                
                # ラベル名をプロット
                ax1.text(feat[0], feat[1] + 0.02, label, ha='center', va='bottom', 
                        fontsize=8, color='black', alpha=0.8)
            
            ax1.set_xlabel('Power (正規化済み)')
            ax1.set_ylabel('Duration (正規化済み)')
            ax1.set_title('Power vs Duration')
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(-0.05, 1.05)
            ax1.set_ylim(-0.05, 1.05)
            
            # 2. Duration vs Speed  
            ax2 = axes[1]
            for i, (feat, label) in enumerate(zip(person_features, person_labels)):
                color = label_color_map[label]
                if label == 'Max':
                    ax2.scatter(feat[1], feat[2], c=[color], s=50, alpha=0.9, 
                              marker="o", edgecolors='black', linewidth=1)
                else:
                    ax2.scatter(feat[1], feat[2], c=[color], s=40, alpha=0.8, 
                              marker="o", edgecolors='black', linewidth=0.5)
                
                # ラベル名をプロット
                ax2.text(feat[1], feat[2] + 0.02, label, ha='center', va='bottom', 
                        fontsize=8, color='black', alpha=0.8)
            
            ax2.set_xlabel('Duration (正規化済み)')
            ax2.set_ylabel('Speed (正規化済み)')
            ax2.set_title('Duration vs Speed')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(-0.05, 1.05)
            ax2.set_ylim(-0.05, 1.05)
            
            # 3. Power vs Speed
            ax3 = axes[2]
            for i, (feat, label) in enumerate(zip(person_features, person_labels)):
                color = label_color_map[label]
                if label == 'Max':
                    ax3.scatter(feat[0], feat[2], c=[color], s=50, alpha=0.9, 
                              marker="o", edgecolors='black', linewidth=1)
                else:
                    ax3.scatter(feat[0], feat[2], c=[color], s=40, alpha=0.8, 
                              marker="o", edgecolors='black', linewidth=0.5)
                
                # ラベル名をプロット
                ax3.text(feat[0], feat[2] + 0.02, label, ha='center', va='bottom', 
                        fontsize=8, color='black', alpha=0.8)
            
            ax3.set_xlabel('Power (正規化済み)')
            ax3.set_ylabel('Speed (正規化済み)')
            ax3.set_title('Power vs Speed')
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim(-0.05, 1.05)
            ax3.set_ylim(-0.05, 1.05)
            
            # 全体のタイトル設定
            fig.suptitle(f'{person_name} - Session {session} ({len(person_features)} samples)', 
                        fontsize=18, y=0.98)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.90)
            
            # ファイル保存
            filename = f'{person_name}_session_{session:02d}.png'
            filepath = os.path.join(session_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"    Saved: {filename}")
    
    print(f"\nAll plots saved to {output_base_dir}/")

def create_summary_info(sensor_data, output_dir='check_features'):
    """プロット作成の概要情報を保存"""
    
    summary_path = os.path.join(output_dir, 'plot_summary.txt')
    
    features = sensor_data['features']
    y_label_names = sensor_data['y_label_name']
    person_ids = sensor_data['person_ids']
    sessions = sensor_data['sessions']
    
    unique_sessions = sorted(np.unique(sessions))
    unique_persons = sorted(np.unique(person_ids))
    unique_labels = sorted(np.unique(y_label_names))
    
    # 被験者名マッピング
    person_name_mapping = {
        1: 'inamura',
        2: 'utsumi', 
        3: 'kawano',
        4: 'hatanaka',
        5: 'watanabe',
        6: 'kikuchi',
        8: 'okamoto'
    }
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=== センサー特徴量2Dプロット概要 ===\n\n")
        
        f.write("【データ概要】\n")
        f.write(f"総サンプル数: {len(features)}\n")
        f.write(f"被験者数: {len(unique_persons)}\n")
        f.write(f"セッション数: {len(unique_sessions)}\n")
        f.write(f"オノマトペ数: {len(unique_labels)}\n\n")
        
        f.write("【被験者一覧】\n")
        for person_id in unique_persons:
            person_name = person_name_mapping.get(person_id, f'person_{person_id}')
            count = np.sum(person_ids == person_id)
            f.write(f"  ID {person_id}: {person_name} ({count} samples)\n")
        f.write("\n")
        
        f.write("【セッション一覧】\n")
        for session in unique_sessions:
            count = np.sum(sessions == session)
            f.write(f"  Session {session}: {count} samples\n")
        f.write("\n")
        
        f.write("【プロット構成】\n")
        f.write("各プロットは3つのサブプロット構成:\n")
        f.write("  1. Power vs Duration (左)\n")
        f.write("  2. Duration vs Speed (中央)\n")
        f.write("  3. Power vs Speed (右)\n\n")
        
        f.write("【出力ファイル構成】\n")
        for session in unique_sessions:
            f.write(f"session_{session:02d}/\n")
            for person_id in unique_persons:
                person_name = person_name_mapping.get(person_id, f'person_{person_id}')
                f.write(f"  {person_name}_session_{session:02d}.png\n")
        
        f.write("\n【特徴量説明】\n")
        f.write("Power: 信号の最大値を正規化 (0-1)\n")
        f.write("Duration: アノテーション範囲の時間長を正規化 (0-1)\n")
        f.write("Speed: 最大値到達までの時間を逆転・正規化 (0-1, 小さいほど速い)\n")
    
    print(f"Summary saved to {summary_path}")

def main():
    """メイン処理"""
    print("=== センサー特徴量2Dプロット作成 ===")
    
    # センサー特徴量データを読み込み
    sensor_data = load_sensor_features()
    if sensor_data is None:
        return
    
    # 出力ディレクトリを作成
    output_dir = 'check_features'
    os.makedirs(output_dir, exist_ok=True)
    
    # セッション別・人別プロットを作成
    plot_session_features_by_person(sensor_data, output_dir)
    
    # 概要情報を保存
    create_summary_info(sensor_data, output_dir)
    
    print("\n=== プロット作成完了 ===")
    print(f"出力ディレクトリ: {output_dir}/")
    print("各セッションディレクトリに被験者別プロットが保存されました。")

if __name__ == "__main__":
    main()