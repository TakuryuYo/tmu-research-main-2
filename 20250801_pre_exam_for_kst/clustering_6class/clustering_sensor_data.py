#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
センサーデータの階層型クラスタリング分析

normalized_datasから正規化済みセンサーデータを読み込み、
個人・セッション別にPower, Speed, Timeの3次元データを用いて階層型クラスタリングを実行。
"""

import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle
import os
import sys
from pathlib import Path

plt.rcParams['font.family'] = ['Times New Roman', 'IPAexGothic']  # 英数字はTimes New Roman、日本語はIPAexGothic
plt.rcParams['font.sans-serif'] = ['IPAexGothic']  # 日本語フォント
plt.rcParams['font.serif'] = ['Times New Roman']  #

plt.rcParams['axes.labelsize'] = 30  # 軸ラベルのフォントサイズ
plt.rcParams['xtick.labelsize'] = 30  # x軸の目盛りラベルのフォントサイズ
plt.rcParams['ytick.labelsize'] = 30  # y軸の目盛りラベルのフォントサイズ
plt.rcParams['axes.titlesize'] = 16   # タイトルのフォントサイズ

def load_normalized_sensor_data(data_path='normalized_datas/annotation_sensor_normalized.bin'):
    """正規化済みセンサーデータを読み込み"""
    print(f"Loading sensor data from {data_path}...")
    
    try:
        with open(data_path, 'rb') as f:
            sensor_data = pickle.load(f)
        
        print(f"Data loaded successfully!")
        print(f"  Features shape: {sensor_data['features'].shape}")
        print(f"  Unique persons: {len(np.unique(sensor_data['person_ids']))}")
        print(f"  Unique labels: {len(np.unique(sensor_data['y_label_name']))}")
        print(f"  Unique sessions: {sorted(np.unique(sensor_data['sessions']))}")
        
        return sensor_data
        
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        print("Please run convert_features_annotation.py first to generate normalized data.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def prepare_person_session_data(sensor_data, person, session):
    """特定の個人・セッションのクラスタリング用データを準備"""
    
    # 対象の個人・セッションのデータを抽出
    person_session_mask = (sensor_data['person_ids'] == person) & (sensor_data['sessions'] == session)
    
    if not np.any(person_session_mask):
        print(f"No data found for {person} session {session}")
        return None, None
    
    features = sensor_data['features'][person_session_mask]
    labels = sensor_data['y_label_name'][person_session_mask]
    
    # 重複するオノマトペがある場合は平均値を取る
    unique_labels = np.unique(labels)
    averaged_features = []
    averaged_labels = []
    
    for label in unique_labels:
        label_mask = labels == label
        label_features = features[label_mask]
        
        # 複数のサンプルがある場合は平均を取る
        if len(label_features) > 1:
            avg_feature = np.mean(label_features, axis=0)
        else:
            avg_feature = label_features[0]
        
        averaged_features.append(avg_feature)
        averaged_labels.append(label)
    
    data_matrix = np.array(averaged_features)
    
    print(f"\n{person} Session {session}:")
    print(f"  Samples: {len(averaged_labels)}")
    print(f"  Labels: {averaged_labels}")
    print(f"  Power range: [{data_matrix[:, 0].min():.3f}, {data_matrix[:, 0].max():.3f}]")
    print(f"  Time range: [{data_matrix[:, 1].min():.3f}, {data_matrix[:, 1].max():.3f}]")
    print(f"  Speed range: [{data_matrix[:, 2].min():.3f}, {data_matrix[:, 2].max():.3f}]")
    
    return averaged_labels, data_matrix

def perform_hierarchical_clustering(data_matrix, method='ward'):
    """階層型クラスタリングを実行"""
    if len(data_matrix) < 2:
        print("Error: Need at least 2 samples for clustering")
        return None, None
    
    # データの標準化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_matrix)
    
    # 階層型クラスタリング実行
    if method == 'ward':
        linkage_matrix = linkage(data_scaled, method='ward')
    else:
        distances = pdist(data_scaled, metric='euclidean')
        linkage_matrix = linkage(distances, method=method)
    
    return linkage_matrix, data_scaled

def plot_clusters_with_dendrogram(data_matrix, onomatopoeia_labels, linkage_matrix, n_clusters, person, session, output_dir):
    """クラスタ結果をデンドログラムと2次元散布図で可視化"""
    
    if len(data_matrix) < n_clusters:
        print(f"Warning: Not enough samples ({len(data_matrix)}) for {n_clusters} clusters. Skipping.")
        return
    
    # クラスタラベルを取得
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    
    # 2×2のサブプロット構成（上段：デンドログラム、下段左：Power vs Time、下段右：クラスタ統計）
    fig = plt.figure(figsize=(20, 12))
    
    # デンドログラム（上段全体）
    ax_dendro = plt.subplot(2, 2, (1, 2))
    
    # カラーマップの設定
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    # デンドログラムのプロット
    dendro = dendrogram(linkage_matrix, 
                       labels=onomatopoeia_labels,
                       ax=ax_dendro,
                       leaf_rotation=45,
                       color_threshold=linkage_matrix[-n_clusters+1, 2] if n_clusters <= len(data_matrix) else 0)
    
    # ax_dendro.set_title(f'階層型クラスタリングデンドログラム ({n_clusters}クラスタ)')
    ax_dendro.set_xlabel('Onomatopoeia')
    ax_dendro.set_ylabel('Distance')
    ax_dendro.grid(True, alpha=0.3)
    
    # Power vs Time 散布図（下段左）
    ax_scatter = plt.subplot(2, 2, 3)
    
    # クラスタごとに色分けしてプロット
    for cluster_id in range(1, n_clusters + 1):
        mask = cluster_labels == cluster_id
        if np.any(mask):
            ax_scatter.scatter(data_matrix[mask, 2], data_matrix[mask, 1],
                             c=[colors[cluster_id-1]], s=100, alpha=0.7,
                             label=f'クラスタ {cluster_id}')
            
            # オノマトペ名を表示
            cluster_data = data_matrix[mask]
            cluster_names = np.array(onomatopoeia_labels)[mask]
            for i, name in enumerate(cluster_names):
                ax_scatter.annotate(name, 
                                  (cluster_data[i, 2], cluster_data[i, 1]),
                                  xytext=(5, 5), textcoords='offset points',
                                  alpha=0.8)
    
    ax_scatter.set_xlabel('Speed')
    ax_scatter.set_ylabel('Duration')
    ax_scatter.grid(True, alpha=0.3)
    ax_scatter.legend()
    
    # クラスタ統計（下段右）
    ax_stats = plt.subplot(2, 2, 4)
    ax_stats.axis('off')
    
    # クラスタ統計情報を表示
    stats_text = f"【{person} Session {session} - {n_clusters}クラスタ統計】\n\n"
    
    for cluster_id in range(1, n_clusters + 1):
        mask = cluster_labels == cluster_id
        cluster_onomatopoeia = np.array(onomatopoeia_labels)[mask]
        
        if len(cluster_onomatopoeia) > 0:
            stats_text += f"クラスタ {cluster_id} ({len(cluster_onomatopoeia)}個):\n"
            stats_text += f"  {', '.join(cluster_onomatopoeia)}\n"
            
            # クラスタ内の平均値を計算
            cluster_data = data_matrix[mask]
            if len(cluster_data) > 0:
                stats_text += f"  Power: {np.mean(cluster_data[:, 0]):.3f} ±{np.std(cluster_data[:, 0]):.3f}\n"
                stats_text += f"  Duration: {np.mean(cluster_data[:, 1]):.3f} ±{np.std(cluster_data[:, 1]):.3f}\n"
                stats_text += f"  Speed: {np.mean(cluster_data[:, 2]):.3f} ±{np.std(cluster_data[:, 2]):.3f}\n\n"
    
    ax_stats.text(0.05, 0.95, stats_text, 
                  transform=ax_stats.transAxes, 
                  verticalalignment='top',
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    
    plt.tight_layout()
    
    # ファイル名を生成
    filename = f"{person}_session{session}_{n_clusters}class_with_dendro.png"
    filepath = os.path.join(output_dir, filename)
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()  # メモリ節約のため閉じる
    
    print(f"  Saved: {filepath}")

def plot_clusters_2d(data_matrix, onomatopoeia_labels, linkage_matrix, n_clusters, person, session, output_dir):
    """クラスタ結果を2次元散布図で可視化（従来版）"""
    
    if len(data_matrix) < n_clusters:
        print(f"Warning: Not enough samples ({len(data_matrix)}) for {n_clusters} clusters. Skipping.")
        return
    
    # クラスタラベルを取得
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    
    # 3つの2次元プロットを作成
    combinations = [
        (2, 1, 'Speed', 'Duration'),
        (2, 0, 'Speed', 'Power'),
        (0, 1, 'Power', 'Duration')
    ]
    
    # カラーマップの設定
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    for idx, (x_idx, y_idx, x_label, y_label) in enumerate(combinations):
        fig, axes = plt.subplots(1, 1, figsize=(7, 6))
        ax = axes
        
        # クラスタごとに色分けしてプロット
        for cluster_id in range(1, n_clusters + 1):
            mask = cluster_labels == cluster_id
            if np.any(mask):
                ax.scatter(data_matrix[mask, x_idx], data_matrix[mask, y_idx],
                          c=[colors[cluster_id-1]], s=100, alpha=0.7,
                          label=f'クラスタ {cluster_id}')
                
                # オノマトペ名を表示
                cluster_data = data_matrix[mask]
                cluster_names = np.array(onomatopoeia_labels)[mask]
                for i, name in enumerate(cluster_names):
                    ax.annotate(name, 
                               (cluster_data[i, x_idx], cluster_data[i, y_idx]),
                               xytext=(5, 5), textcoords='offset points',
                              alpha=0.8)
        
        ax.set_xlabel(f'{x_label}')
        ax.set_ylabel(f'{y_label}')
        ax.set_title(f'{x_label} vs {y_label}')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
        #plt.suptitle(f'{person} Session {session} - 階層型クラスタリング結果 ({n_clusters}クラスタ)')
        plt.tight_layout()
        
        # ファイル名を生成
        filename = f"{person}_session{session}_{n_clusters}class_{x_label}_{y_label}.png"
        filepath = os.path.join(output_dir, filename)
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()  # メモリ節約のため閉じる
    
        print(f"  Saved: {filepath}")

def save_clustering_results_csv(onomatopoeia_labels, data_matrix, linkage_matrix, n_clusters, person, session, output_dir):
    """クラスタリング結果をCSVファイルとして保存"""
    
    if len(data_matrix) < n_clusters:
        return None
    
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    
    # クラスタリング結果をデータフレームに変換
    results_data = []
    for i, onomatopoeia in enumerate(onomatopoeia_labels):
        results_data.append({
            'オノマトペ': onomatopoeia,
            'クラスターID': cluster_labels[i],
            'パワー': data_matrix[i, 0],
            '持続性': data_matrix[i, 1],
            'スピード': data_matrix[i, 2]
        })
    
    df = pd.DataFrame(results_data)
    
    # CSVファイル名を生成
    csv_filename = f'sensor_clustering_results_{person}_session{session}_{n_clusters}clusters.csv'
    csv_path = os.path.join(output_dir, csv_filename)
    
    # CSVファイルとして保存
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  Saved CSV: {csv_path}")
    
    return csv_path

def analyze_clusters(onomatopoeia_labels, data_matrix, linkage_matrix, n_clusters, person, session):
    """クラスタ分析結果を表示"""
    
    if len(data_matrix) < n_clusters:
        return
    
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    
    print(f"\n=== {person} Session {session} - {n_clusters}クラスタ分析結果 ===")
    
    # クラスタごとの統計情報
    for cluster_id in range(1, n_clusters + 1):
        mask = cluster_labels == cluster_id
        cluster_onomatopoeia = np.array(onomatopoeia_labels)[mask]
        
        if len(cluster_onomatopoeia) > 0:
            print(f"\n【クラスタ {cluster_id}】 ({len(cluster_onomatopoeia)}個)")
            print(f"オノマトペ: {', '.join(cluster_onomatopoeia)}")
            
            # クラスタ内の平均値を計算
            cluster_data = data_matrix[mask]
            if len(cluster_data) > 0:
                print(f"特徴量平均:")
                print(f"  Power: {np.mean(cluster_data[:, 0]):.3f} (±{np.std(cluster_data[:, 0]):.3f})")
                print(f"  Duration: {np.mean(cluster_data[:, 1]):.3f} (±{np.std(cluster_data[:, 1]):.3f})")
                print(f"  Speed: {np.mean(cluster_data[:, 2]):.3f} (±{np.std(cluster_data[:, 2]):.3f})")

def process_person_session_clustering(sensor_data, person, session, output_dir):
    """特定の個人・セッションのクラスタリングを実行"""
    
    print(f"\n" + "="*60)
    print(f"Processing {person} Session {session}")
    print("="*60)
    
    # データ準備
    onomatopoeia_labels, data_matrix = prepare_person_session_data(sensor_data, person, session)
    
    if onomatopoeia_labels is None:
        return
    
    # 最小サンプル数チェック
    if len(data_matrix) < 5:
        print(f"Warning: Only {len(data_matrix)} samples available. Need at least 5 for meaningful clustering.")
        if len(data_matrix) < 2:
            return
    
    # 階層型クラスタリング実行
    linkage_matrix, data_scaled = perform_hierarchical_clustering(data_matrix)
    
    if linkage_matrix is None:
        return
    
    # 5, 6, 7クラスタでの結果を作成
    cluster_numbers = [5, 6, 7]
    
    for n_clusters in cluster_numbers:
        if len(data_matrix) >= n_clusters:
            print(f"\n--- {n_clusters}クラスタでの分析 ---")
            
            # デンドログラム付きクラスタ結果を可視化
            plot_clusters_with_dendrogram(data_matrix, onomatopoeia_labels, linkage_matrix, 
                                        n_clusters, person, session, output_dir)
            
            # 従来の2次元散布図も保持
            plot_clusters_2d(data_matrix, onomatopoeia_labels, linkage_matrix, 
                            n_clusters, person, session, output_dir)
            
            # クラスタ分析結果を表示
            analyze_clusters(onomatopoeia_labels, data_matrix, linkage_matrix, 
                           n_clusters, person, session)
            
            # クラスタリング結果をCSVで保存
            save_clustering_results_csv(onomatopoeia_labels, data_matrix, linkage_matrix, 
                                      n_clusters, person, session, output_dir)
        else:
            print(f"Skipping {n_clusters} clusters (only {len(data_matrix)} samples available)")

def main():
    """メイン処理"""
    print("=== センサーデータ階層型クラスタリング分析 ===")
    
    # 正規化済みセンサーデータを読み込み
    sensor_data = load_normalized_sensor_data()
    if sensor_data is None:
        return
    
    # 出力ディレクトリを作成
    output_dir = 'result'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # 全ての個人・セッション組み合わせを取得
    unique_persons = np.unique(sensor_data['person_ids'])
    unique_sessions = np.unique(sensor_data['sessions'])
    
    print(f"\nAvailable persons: {list(unique_persons)}")
    print(f"Available sessions: {list(unique_sessions)}")
    
    # 各個人・セッションの組み合わせについて処理
    total_combinations = 0
    processed_combinations = 0
    
    for person in unique_persons:
        for session in unique_sessions:
            total_combinations += 1
            
            # データが存在するかチェック
            person_session_mask = (sensor_data['person_ids'] == person) & (sensor_data['sessions'] == session)
            if np.any(person_session_mask):
                process_person_session_clustering(sensor_data, person, session, output_dir)
                processed_combinations += 1
    
    print(f"\n" + "="*60)
    print(f"センサーデータクラスタリング分析完了!")
    print(f"処理済み: {processed_combinations}/{total_combinations} 組み合わせ")
    print(f"結果ファイル: {output_dir}/")
    print("="*60)

if __name__ == "__main__":
    main()