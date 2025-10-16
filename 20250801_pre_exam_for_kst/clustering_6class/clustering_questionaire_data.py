#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
オノマトペの階層型クラスタリング分析 アンケート（個人別版）
Power, Speed, Duration(持続性)の3次元データを用いて各参加者ごとに階層型クラスタリングを実行し、
デンドログラムとクラスタ結果の可視化を行う
"""

import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
plt.rcParams['font.family'] = ['Times New Roman', 'IPAexGothic']  # 英数字はTimes New Roman、日本語はIPAexGothic
plt.rcParams['font.sans-serif'] = ['IPAexGothic']  # 日本語フォント
plt.rcParams['font.serif'] = ['Times New Roman']  #

plt.rcParams['axes.labelsize'] = 30  # 軸ラベルのフォントサイズ
plt.rcParams['xtick.labelsize'] = 30  # x軸の目盛りラベルのフォントサイズ
plt.rcParams['ytick.labelsize'] = 30  # y軸の目盛りラベルのフォントサイズ
plt.rcParams['axes.titlesize'] = 16   # タイトルのフォントサイズ

def load_json_data():
    """全てのJSONファイルを読み込む"""
    json_files = glob.glob("../questionaire_result/*.json")
    data = []
    
    for file in json_files:
        with open(file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            data.append(json_data)
    
    return data

def extract_onomatopoeia_data(data):
    """オノマトペデータを抽出し整理"""
    # 被験者番号と名前のマッピング
    participant_name_mapping = {
        "1": "Iさん",
        "2": "Uさん", 
        "3": "Kさん",
        "4": "Hさん",
        "5": "Wさん",
        "6": "菊池さん",
        "7": "吉永",
        "8": "Oさん"
    }
    
    participants = []
    
    for json_data in data:
        participant_id = json_data["responses"]["【あなたについて】_被験者番号"]
        participant_name = participant_name_mapping.get(participant_id, f"参加者{participant_id}")
        responses = json_data["responses"]
        
        # オノマトペデータを抽出
        onomatopoeia_data = {}
        
        # パワー、スピード、時間の値を抽出
        for key, value in responses.items():
            if "【オノマトペについて】" in key:
                try:
                    # キーを解析してオノマトペ名と特徴量を抽出
                    if 'パワー' in key and '： 「' in key:
                        onomatopoeia = key.split('： 「')[1].split('」')[0]
                        if onomatopoeia not in onomatopoeia_data:
                            onomatopoeia_data[onomatopoeia] = {}
                        onomatopoeia_data[onomatopoeia]['power'] = float(value) / 100.0
                    
                    elif 'スピード' in key and '： 「' in key:
                        onomatopoeia = key.split('： 「')[1].split('」')[0]
                        if onomatopoeia not in onomatopoeia_data:
                            onomatopoeia_data[onomatopoeia] = {}
                        onomatopoeia_data[onomatopoeia]['speed'] = float(value) / 100.0
                    
                    elif '時間の長さ（持続性）' in key and '： 「' in key:
                        onomatopoeia = key.split('： 「')[1].split('」')[0]
                        if onomatopoeia not in onomatopoeia_data:
                            onomatopoeia_data[onomatopoeia] = {}
                        onomatopoeia_data[onomatopoeia]['time'] = float(value) / 100.0
                except (IndexError, ValueError) as e:
                    print(f"Warning: Failed to parse key '{key}' with value '{value}': {e}")
                    continue
        
        participants.append({
            'id': participant_id,
            'name': participant_name,
            'onomatopoeia': onomatopoeia_data
        })
    
    return participants

def prepare_clustering_data_for_participant(participant):
    """個人のクラスタリング用データを準備"""
    onomatopoeia_data = participant['onomatopoeia']
    
    # 完全なデータ（power, speed, time全てある）のオノマトペのみを使用
    valid_onomatopoeia = []
    data_matrix = []
    onomatopoeia_values = {}
    
    for onomatopoeia, values in onomatopoeia_data.items():
        if 'power' in values and 'speed' in values and 'time' in values:
            valid_onomatopoeia.append(onomatopoeia)
            data_matrix.append([values['power'], values['speed'], values['time']])
            onomatopoeia_values[onomatopoeia] = {
                'power': values['power'],
                'speed': values['speed'], 
                'time': values['time'],
                'count': 1
            }
    
    if len(valid_onomatopoeia) < 3:
        return None, None, None  # クラスタリングに必要な最小データ数がない
        
    data_matrix = np.array(data_matrix)
    return valid_onomatopoeia, data_matrix, onomatopoeia_values

def perform_hierarchical_clustering(data_matrix, method='ward', metric='euclidean'):
    """階層型クラスタリングを実行"""
    # データの標準化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_matrix)
    
    # 階層型クラスタリング実行
    if method == 'ward':
        # Ward法の場合はユークリッド距離を使用
        linkage_matrix = linkage(data_scaled, method='ward')
    else:
        # その他の手法の場合は距離行列を先に計算
        distances = pdist(data_scaled, metric=metric)
        linkage_matrix = linkage(distances, method=method)
    
    return linkage_matrix, data_scaled, scaler

def plot_dendrogram(linkage_matrix, onomatopoeia_names, filename, method='ward', participant=None, n_clusters=6):
    """デンドログラムを描画"""
    plt.figure(figsize=(10, 10))
    
    # 6クラスで色分けするためのthresholdを計算
    from scipy.cluster.hierarchy import fcluster
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    
    # デンドログラムを6色で色分けして描画
    dendrogram(linkage_matrix, 
               labels=onomatopoeia_names,
               orientation='top',
               distance_sort='descending',
               show_leaf_counts=True,
               leaf_font_size=20,
               color_threshold=linkage_matrix[-n_clusters+1, 2],  # 6クラスでの閾値
               above_threshold_color='gray')
    
    person_name = participant['name'] if participant else 'Unknown'
    #  plt.title(f'オノマトペの階層型クラスタリング - {person_name}\n手法: {method.upper()}')
    plt.xlabel('Onomatopoeia ')
    plt.ylabel('Distance')
    plt.xticks(rotation=80, ha='right')
    plt.grid(True, alpha=0.3)
    
    # 6クラス分割線を追加
    threshold = linkage_matrix[-n_clusters+1, 2]
    plt.axhline(y=threshold, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    plt.tight_layout()
    
    # ディレクトリが存在しない場合は作成
    output_dir = 'clustering_dend'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

def plot_clusters_2d(data_matrix, onomatopoeia_names, linkage_matrix, n_clusters=5, method='ward', participant=None):
    """クラスタ結果を2次元散布図で可視化"""
    # クラスタラベルを取得
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    
    # 参加者情報を取得（ファイル名用）
    if participant:
        person_suffix = participant['name']
    else:
        person_suffix = 'unknown_participant'
    
    # 3つの2次元プロットを作成
    combinations = [
        (1, 2, 'Speed', 'Duration'),
        (1, 0, 'Speed', 'Power'),
        (0, 2, 'Power', 'Duration')
    ]
    
    # カラーマップの設定（6クラス固定）
    colors = plt.cm.tab10(np.linspace(0, 1, 6))
    
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
                cluster_names = np.array(onomatopoeia_names)[mask]
                for i, name in enumerate(cluster_names):
                    ax.annotate(name, 
                               (cluster_data[i, x_idx], cluster_data[i, y_idx]),
                               xytext=(5, 5), textcoords='offset points',
                              alpha=0.8)
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_title(f'{x_label} vs {y_label} - {person_suffix}')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
        plt.tight_layout()
        
        # ディレクトリが存在しない場合は作成
        output_dir = 'clustering_dend'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # ファイル名にperson名を含める
        filename = f'onomatopoeia_clusters_2d_{method}_{n_clusters}clusters_{x_label}_{y_label}_{person_suffix}.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

def plot_3d_clusters(data_matrix, onomatopoeia_names, linkage_matrix, n_clusters=5, method='ward', participant=None):
    """3次元でクラスタ結果を可視化"""
    from mpl_toolkits.mplot3d import Axes3D
    
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    
    # 参加者情報を取得（ファイル名用）
    if participant:
        person_suffix = participant['name']
    else:
        person_suffix = 'unknown_participant'
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # カラーマップの設定（6クラス固定）
    colors = plt.cm.tab10(np.linspace(0, 1, 6))
    
    # クラスタごとに色分けしてプロット
    for cluster_id in range(1, n_clusters + 1):
        mask = cluster_labels == cluster_id
        if np.any(mask):
            ax.scatter(data_matrix[mask, 1], data_matrix[mask, 2], data_matrix[mask, 0],
                      c=[colors[cluster_id-1]], s=100, alpha=0.7,
                      label=f'クラスタ {cluster_id}')
            
            # オノマトペ名を表示
            for i, name in enumerate(np.array(onomatopoeia_names)[mask]):
                mask_indices = np.where(mask)[0]
                ax.text(data_matrix[mask_indices[i], 1], 
                       data_matrix[mask_indices[i], 2], 
                       data_matrix[mask_indices[i], 0],
                       name)
    
    ax.set_xlabel('Speed')
    ax.set_ylabel('Duration')
    ax.set_zlabel('Power')
    ax.set_title(f'オノマトペ3次元クラスタリング ({n_clusters}クラスタ) - {person_suffix}')
    ax.legend()
    
    plt.tight_layout()
    
    # ディレクトリが存在しない場合は作成
    output_dir = 'clustering_dend'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # ファイル名にperson名を含める
    filename = f'onomatopoeia_clusters_3d_{method}_{n_clusters}clusters_{person_suffix}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

def save_clustering_results_csv(onomatopoeia_names, onomatopoeia_means, linkage_matrix, n_clusters=5, method='ward', participant=None):
    """クラスタリング結果をCSVファイルとして保存"""
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    
    # ディレクトリ作成
    result_dir = 'clustering_dend/result'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # 参加者情報を取得（ファイル名用）
    if participant:
        person_suffix = participant['name']
    else:
        person_suffix = 'unknown_participant'
    
    # クラスタリング結果をデータフレームに変換
    results_data = []
    for i, onomatopoeia in enumerate(onomatopoeia_names):
        results_data.append({
            'オノマトペ': onomatopoeia,
            'クラスターID': cluster_labels[i],
            'パワー': onomatopoeia_means[onomatopoeia]['power'],
            'スピード': onomatopoeia_means[onomatopoeia]['speed'],
            '持続性': onomatopoeia_means[onomatopoeia]['time'],
            'データ数': onomatopoeia_means[onomatopoeia]['count']
        })
    
    df = pd.DataFrame(results_data)
    
    # CSVファイル名を生成
    csv_filename = f'clustering_results_{method}_{n_clusters}clusters_{person_suffix}.csv'
    csv_path = os.path.join(result_dir, csv_filename)
    
    # CSVファイルとして保存
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\nクラスタリング結果を保存: {csv_path}")
    
    # クラスタ別統計を保存
    cluster_stats = []
    for cluster_id in range(1, n_clusters + 1):
        mask = cluster_labels == cluster_id
        cluster_onomatopoeia = np.array(onomatopoeia_names)[mask]
        
        if len(cluster_onomatopoeia) > 0:
            cluster_powers = [onomatopoeia_means[name]['power'] for name in cluster_onomatopoeia]
            cluster_speeds = [onomatopoeia_means[name]['speed'] for name in cluster_onomatopoeia]
            cluster_times = [onomatopoeia_means[name]['time'] for name in cluster_onomatopoeia]
            
            cluster_stats.append({
                'クラスターID': cluster_id,
                'オノマトペ数': len(cluster_onomatopoeia),
                'オノマトペリスト': ', '.join(cluster_onomatopoeia),
                'パワー_平均': np.mean(cluster_powers),
                'パワー_標準偏差': np.std(cluster_powers),
                'スピード_平均': np.mean(cluster_speeds),
                'スピード_標準偏差': np.std(cluster_speeds),
                '持続性_平均': np.mean(cluster_times),
                '持続性_標準偏差': np.std(cluster_times)
            })
    
    stats_df = pd.DataFrame(cluster_stats)
    stats_filename = f'cluster_statistics_{method}_{n_clusters}clusters_{person_suffix}.csv'
    stats_path = os.path.join(result_dir, stats_filename)
    stats_df.to_csv(stats_path, index=False, encoding='utf-8-sig')
    print(f"クラスタ統計を保存: {stats_path}")
    
    return csv_path, stats_path

def analyze_clusters(onomatopoeia_names, onomatopoeia_means, linkage_matrix, n_clusters=5):
    """クラスタ分析結果を詳細に表示"""
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    
    print("=" * 80)
    print(f"階層型クラスタリング分析結果 ({n_clusters}クラスタ)")
    print("=" * 80)
    
    # クラスタごとの統計情報
    for cluster_id in range(1, n_clusters + 1):
        mask = cluster_labels == cluster_id
        cluster_onomatopoeia = np.array(onomatopoeia_names)[mask]
        
        if len(cluster_onomatopoeia) > 0:
            print(f"\n【クラスタ {cluster_id}】 ({len(cluster_onomatopoeia)}個)")
            print(f"オノマトペ: {', '.join(cluster_onomatopoeia)}")
            
            # クラスタ内の平均値を計算
            cluster_powers = [onomatopoeia_means[name]['power'] for name in cluster_onomatopoeia]
            cluster_speeds = [onomatopoeia_means[name]['speed'] for name in cluster_onomatopoeia]
            cluster_times = [onomatopoeia_means[name]['time'] for name in cluster_onomatopoeia]
            
            print(f"特徴量平均:")
            print(f"  パワー: {np.mean(cluster_powers):.3f} (±{np.std(cluster_powers):.3f})")
            print(f"  スピード: {np.mean(cluster_speeds):.3f} (±{np.std(cluster_speeds):.3f})")
            print(f"  持続性: {np.mean(cluster_times):.3f} (±{np.std(cluster_times):.3f})")

def main():
    """メイン処理"""
    # データ読み込み
    data = load_json_data()
    print(f"読み込んだファイル数: {len(data)}")
    
    # オノマトペデータを抽出
    participants = extract_onomatopoeia_data(data)
    print(f"参加者数: {len(participants)}")
    
    # 複数の手法で階層型クラスタリングを実行
    methods = ['ward', 'complete', 'average']
    
    # 各参加者について個別にクラスタリングを実行
    for participant in participants:
        print(f"\n" + "=" * 80)
        print(f"参加者: {participant['name']} (ID: {participant['id']}) のクラスタリング分析")
        print("=" * 80)
        
        # 個人のクラスタリング用データを準備
        onomatopoeia_names, data_matrix, onomatopoeia_means = prepare_clustering_data_for_participant(participant)
        
        if onomatopoeia_names is None:
            print(f"参加者 {participant['name']} のデータが不十分です（有効なオノマトペ数: < 3）")
            continue
            
        print(f"クラスタリング対象オノマトペ数: {len(onomatopoeia_names)}")
        print(f"オノマトペリスト: {', '.join(onomatopoeia_names)}")
        
        # データの基本統計を表示
        print(f"\nデータ統計:")
        print(f"Power 範囲: {data_matrix[:, 0].min():.3f} - {data_matrix[:, 0].max():.3f}")
        print(f"Speed 範囲: {data_matrix[:, 1].min():.3f} - {data_matrix[:, 1].max():.3f}")
        print(f"Duration 範囲: {data_matrix[:, 2].min():.3f} - {data_matrix[:, 2].max():.3f}")
        
        for method in methods:
            print(f"\n" + "-" * 60)
            print(f"{method.upper()}法による階層型クラスタリング - {participant['name']}")
            print("-" * 60)
            
            # クラスタリング実行
            linkage_matrix, data_scaled, scaler = perform_hierarchical_clustering(data_matrix, method=method)
            
            # デンドログラム描画（6クラス色分け）
            dendrogram_filename = f'onomatopoeia_dendrogram_{method}_{participant["name"]}.png'
            plot_dendrogram(linkage_matrix, onomatopoeia_names, dendrogram_filename, method, participant, n_clusters=6)
            print(f"デンドログラム保存: clustering_dend/{dendrogram_filename}")
            
            # 6クラスタでの結果を表示
            for n_clusters in [6]:
                # データ数がクラスタ数より少ない場合はスキップ
                if len(onomatopoeia_names) < n_clusters:
                    print(f"\n--- {n_clusters}クラスタ: データ数不足のためスキップ ---")
                    continue
                    
                print(f"\n--- {n_clusters}クラスタでの分析 ---")
                
                # 2次元散布図
                plot_clusters_2d(data_matrix, onomatopoeia_names, linkage_matrix, n_clusters, method, participant)
                
                # 3次元散布図
                if n_clusters == 6:  # 6クラスタの場合のみ3次元表示
                    plot_3d_clusters(data_matrix, onomatopoeia_names, linkage_matrix, n_clusters, method, participant)
                
                # クラスタ分析
                analyze_clusters(onomatopoeia_names, onomatopoeia_means, linkage_matrix, n_clusters)
                
                # クラスタリング結果をCSVで保存
                save_clustering_results_csv(onomatopoeia_names, onomatopoeia_means, linkage_matrix, n_clusters, method, participant)
    
    print(f"\n" + "=" * 80)
    print("すべての個人別階層型クラスタリング分析が完了しました!")
    print("=" * 80)

if __name__ == "__main__":
    main()