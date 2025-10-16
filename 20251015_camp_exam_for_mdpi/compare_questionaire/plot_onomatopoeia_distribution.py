#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
オノマトペの分散可視化＋個別データ点分布表示
分散楕円の中に各被験者の個別データ点を表示することで、
実際のデータがどのように分布しているかを詳細に可視化
"""

import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import os
from scipy.spatial.distance import mahalanobis

plt.rcParams['font.family'] = ['Times New Roman', 'IPAexGothic']  # 英数字はTimes New Roman、日本語はIPAexGothic
plt.rcParams['font.sans-serif'] = ['IPAexGothic']  # 日本語フォント
plt.rcParams['font.serif'] = ['Times New Roman']  #

plt.rcParams['axes.labelsize'] = 30  # 軸ラベルのフォントサイズ
plt.rcParams['xtick.labelsize'] = 30  # x軸の目盛りラベルのフォントサイズ
plt.rcParams['ytick.labelsize'] = 30  # y軸の目盛りラベルのフォントサイズ
plt.rcParams['axes.titlesize'] = 16   # タイトルのフォントサイズ

ID_table = [
    "A", "B", "C", "D","E", "F"
]

# 日本語フォント設定
#plt.rcParams['font.family'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

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
    participants = []
    
    for json_data in data:
        participant_id = json_data["responses"]["【あなたについて】_被験者番号"]
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
            'onomatopoeia': onomatopoeia_data
        })
    
    return participants

def calculate_onomatopoeia_statistics_with_points(participants):
    """各オノマトペの統計量と個別データ点を計算"""
    # オノマトペごとのデータを集約
    onomatopoeia_stats = {}
    
    # 全オノマトペを収集
    all_onomatopoeia = set()
    for participant in participants:
        all_onomatopoeia.update(participant['onomatopoeia'].keys())
    
    for onomatopoeia in all_onomatopoeia:
        # 各被験者のデータを格納
        participant_data = []
        
        # 被験者間でのデータを収集
        for participant in participants:
            if onomatopoeia in participant['onomatopoeia']:
                values = participant['onomatopoeia'][onomatopoeia]
                # 3つの値すべてが存在する場合のみ追加
                if 'power' in values and 'speed' in values and 'time' in values:
                    participant_data.append({
                        'participant_id': participant['id'],
                        'power': values['power'],
                        'speed': values['speed'],
                        'time': values['time']
                    })
        
        if len(participant_data) >= 2:  # 最低2人のデータが必要
            # 各変数の統計量を計算
            stats = {}
            
            for var_name in ['power', 'speed', 'time']:
                var_values = [data[var_name] for data in participant_data]
                stats[var_name] = {
                    'mean': np.mean(var_values),
                    'std': np.std(var_values, ddof=1),  # 標本標準偏差
                    'var': np.var(var_values, ddof=1),  # 標本分散
                    'count': len(var_values),
                    'values': var_values
                }
            
            # 2次元組み合わせの統計量も計算
            for var_pair in [('power', 'speed'), ('power', 'time'), ('speed', 'time')]:
                x_var, y_var = var_pair
                pair_key = f"{x_var}_{y_var}"
                
                # 2次元座標として扱う
                x_values = [data[x_var] for data in participant_data]
                y_values = [data[y_var] for data in participant_data]
                participant_ids = [data['participant_id'] for data in participant_data]
                
                stats[pair_key] = {
                    'x_mean': np.mean(x_values),
                    'y_mean': np.mean(y_values),
                    'x_std': np.std(x_values, ddof=1),
                    'y_std': np.std(y_values, ddof=1),
                    'x_var': np.var(x_values, ddof=1),
                    'y_var': np.var(y_values, ddof=1),
                    'count': len(x_values),
                    'x_values': x_values,
                    'y_values': y_values,
                    'participant_ids': participant_ids
                }
            
            onomatopoeia_stats[onomatopoeia] = stats
    
    return onomatopoeia_stats

def create_distribution_plot(onomatopoeia_stats, x_var, y_var, filename):
    """分散楕円と個別データ点の分布を表示"""
    fig, ax = plt.subplots(figsize=(24, 16))
    
    # カラーマップの設定（オノマトペ用）
    onomatopoeia_colors = plt.cm.tab20(np.linspace(0, 1, len(onomatopoeia_stats)))
    
    # 被験者用のマーカー設定
    participant_markers = ['o', 's', '^', 'v', '<', '>', 'D', 'P', '*', 'X', 'h', 'H', '+', 'x', 'd']
    participant_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    
    # 全被験者IDを収集
    all_participant_ids = set()
    for stats in onomatopoeia_stats.values():
        pair_key = f"{x_var}_{y_var}"
        if pair_key in stats:
            all_participant_ids.update(stats[pair_key]['participant_ids'])
    
    # 被験者IDとマーカーのマッピング
    participant_marker_map = {}
    participant_color_map = {}
    for i, pid in enumerate(sorted(all_participant_ids)):
        participant_marker_map[pid] = participant_markers[i % len(participant_markers)]
        participant_color_map[pid] = participant_colors[i % len(participant_colors)]
    
    legend_elements_onomatopoeia = []
    legend_elements_participants = []
    pair_key = f"{x_var}_{y_var}"
    
    for i, (onomatopoeia, stats) in enumerate(sorted(onomatopoeia_stats.items())):
        # 2次元組み合わせのデータが存在するかチェック
        if pair_key in stats and stats[pair_key]['count'] >= 2:
            # 2次元での重心（平均値）を取得
            x_mean = stats[pair_key]['x_mean']
            y_mean = stats[pair_key]['y_mean']
            x_variance = stats[pair_key]['x_var']
            y_variance = stats[pair_key]['y_var']
            
            # 個別データ点を取得
            x_values = stats[pair_key]['x_values']
            y_values = stats[pair_key]['y_values']
            participant_ids = stats[pair_key]['participant_ids']
            
            color = onomatopoeia_colors[i]
            
            # 分散楕円を描画
            ellipse = Ellipse((x_mean, y_mean), 
                            width=np.sqrt(x_variance) * 2,   # x軸方向の分散に基づく広がり
                            height=np.sqrt(y_variance) * 2,  # y軸方向の分散に基づく広がり
                            facecolor=color, alpha=0.2, 
                            edgecolor=color, linewidth=2)
            ax.add_patch(ellipse)
            
            # 重心点をプロット（大きい黒い点）
            ax.scatter(x_mean, y_mean, c='black', s=200, marker='*', 
                      edgecolors='white', linewidth=2, alpha=0.9, zorder=5)
            
            # 個別データ点をプロット（被験者別にマーカーを変える）
            for x_val, y_val, pid in zip(x_values, y_values, participant_ids):
                marker = participant_marker_map[pid]
                ax.scatter(x_val, y_val, c=[color], s=80, marker=marker, 
                          alpha=0.8, edgecolors='black', linewidth=0.5, zorder=3)
                
                # 重心と個別データ点を線で繋ぐ
                ax.plot([x_mean, x_val], [y_mean, y_val], color=color, alpha=0.4, linewidth=1, zorder=2)
            
            # オノマトペ名をテキストとして表示
            ax.text(x_mean, y_mean, onomatopoeia, 
                   fontsize=20, alpha=0.9, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                   zorder=6)
            
            # オノマトペ凡例用要素を作成
            legend_elements_onomatopoeia.append(plt.Line2D([0], [0], marker='o', color='w', 
                                           markerfacecolor=color, markersize=10, 
                                           label=f'{onomatopoeia} (n={stats[pair_key]["count"]})'))
    
    # 被験者凡例用要素を作成
    for pid in sorted(all_participant_ids):
        marker = participant_marker_map[pid]
        legend_elements_participants.append(plt.Line2D([0], [0], marker=marker, color='gray', 
                                                      markerfacecolor='gray', markersize=8, 
                                                      linestyle='None',
                                                      label=f'被験者{pid}'))
    
    # 変数名のマッピング
    var_names = {
        'power': 'Power',
        'speed': 'Speed', 
        'time': 'Duration'
    }
    
    # 軸ラベル設定
    ax.set_xlabel(var_names[x_var])
    ax.set_ylabel(var_names[y_var])
    #ax.set_title(f'オノマトペの分散と個別データ分布\n({var_names[x_var]} vs {var_names[y_var]})\n★重心、楕円は分散範囲、線は重心と個別データの関係', fontsize=16)
    
    # 軸の範囲を設定
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    
    # グリッドを追加
    ax.grid(True, alpha=0.3)
    
    # オノマトペ凡例を追加
    if legend_elements_onomatopoeia:
        legend1 = ax.legend(handles=legend_elements_onomatopoeia, 
                          title='オノマトペ (被験者数)', 
                          bbox_to_anchor=(1.01, 1), 
                          loc='upper left',
                          fontsize=20,
                          ncol=1)
        ax.add_artist(legend1)
    
    # 被験者凡例を追加
    if legend_elements_participants:
        # 被験者数に応じて列数を調整
        ncol = min(3, max(1, len(legend_elements_participants) // 6))
        legend2 = ax.legend(handles=legend_elements_participants, 
                          title='被験者マーカー', 
                          bbox_to_anchor=(1.01, 0.45), 
                          loc='upper left',
                          fontsize=20,
                          ncol=ncol,
                          columnspacing=0.5,
                          handletextpad=0.3)
    
    # 出力ディレクトリを作成
    output_dir = 'distribution_plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 凡例用のスペースを確保するために余白を調整
    plt.subplots_adjust(right=0.75)  # 右側に25%の余白を確保
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    #plt.show()
    
    return fig

def create_detailed_onomatopoeia_plot(onomatopoeia_stats, target_onomatopoeia, filename_prefix):
    """特定のオノマトペに絞った詳細分析プロット"""
    if target_onomatopoeia not in onomatopoeia_stats:
        print(f"オノマトペ '{target_onomatopoeia}' のデータが見つかりません")
        return
    
    stats = onomatopoeia_stats[target_onomatopoeia]
    
    # 被験者用のマーカーとカラー設定
    participant_ids = list(set(stats['power_speed']['participant_ids']))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    markers = ['o']
    
    participant_style_map = {}
    for i, pid in enumerate(sorted(participant_ids)):
        participant_style_map[pid] = {
            'color': colors[i % len(colors)],
            'marker': markers[i % len(markers)]
        }
    
    combinations = [
        ('power', 'speed', 'Power', 'Speed', 0),
        ('power', 'time', 'Power', 'Duration', 1),
        ('speed', 'time', 'Speed', 'Duration', 2)
    ]
    
    for x_var, y_var, x_label, y_label, idx in combinations:
        fig, axes = plt.subplots(1, 1, figsize=(9, 8))
    
        ax = axes
        pair_key = f"{x_var}_{y_var}"
        
        if pair_key in stats:
            # 統計値を取得
            x_mean = stats[pair_key]['x_mean']
            y_mean = stats[pair_key]['y_mean']
            x_variance = stats[pair_key]['x_var']
            y_variance = stats[pair_key]['y_var']
            x_values = stats[pair_key]['x_values']
            y_values = stats[pair_key]['y_values']
            participant_ids_pair = stats[pair_key]['participant_ids']
            
            # 複数の信頼区間楕円を描画（1σ、2σ、3σ）- 共分散行列を使用
            colors_ellipse = ['lightblue', 'lightgreen', 'lightyellow']
            
            # データ点の共分散行列を計算
            data_points = np.column_stack((x_values, y_values))
            if len(data_points) > 1:
                cov_matrix = np.cov(data_points.T)
                
                # 固有値・固有ベクトルを計算（楕円の向きと大きさを決定）
                eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
                
                # 楕円の回転角度を計算
                angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
                
                for scale, alpha, lw, ec in [(1, 0.4, 2, 'blue'), (2, 0.25, 1.5, 'green'), (3, 0.15, 1, 'orange')]:
                    # 固有値に基づいて楕円の幅・高さを計算
                    width = np.sqrt(eigenvals[0]) * scale * 2
                    height = np.sqrt(eigenvals[1]) * scale * 2
                    
                    ellipse = Ellipse((x_mean, y_mean), 
                                    width=width, height=height, angle=angle,
                                    facecolor=colors_ellipse[(scale-1) % len(colors_ellipse)], alpha=alpha, 
                                    edgecolor=ec, linewidth=lw,
                                    label=f'{scale} σ Range' if scale == 1 else None)
                    ax.add_patch(ellipse)
            else:
                # データ点が1つの場合は点のみ表示
                pass
            
            # 平均値点をプロット
            ax.scatter(x_mean, y_mean, c='red', s=200, marker='*', 
                      edgecolors='black', linewidth=2, alpha=0.9, zorder=5,
                      label='Mean')
            
            # 個別データ点をプロット（凡例は後で統一して作成）
            for x_val, y_val, pid in zip(x_values, y_values, participant_ids_pair):
                style = participant_style_map[pid]
                
                ax.scatter(x_val, y_val, c=[style['color']], s=100, marker="o", 
                          alpha=0.8, edgecolors='black', linewidth=0.5, zorder=3)
            
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            # ax.set_title(f'{target_onomatopoeia}\n{x_label} vs {y_label}', fontsize=14)
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 1.1)
            ax.set_aspect('equal', adjustable='box')  # 軸の長さを揃える
            ax.grid(True, alpha=0.3)
            
            # データ点が各楕円内に含まれる割合を計算
            data_points = np.column_stack((x_values, y_values))
            center = np.array([x_mean, y_mean])
            
            # 共分散行列を計算
            if len(data_points) > 1:
                cov_matrix = np.cov(data_points.T)
                if np.linalg.det(cov_matrix) > 1e-10:  # 特異行列でない場合
                    try:
                        cov_inv = np.linalg.inv(cov_matrix)
                        # 各データ点のマハラノビス距離を計算
                        mahal_distances = []
                        for point in data_points:
                            dist = mahalanobis(point, center, cov_inv)
                            mahal_distances.append(dist)
                        
                        mahal_distances = np.array(mahal_distances)
                        
                        # 各σ範囲内の点の割合を計算
                        within_1sigma = np.sum(mahal_distances <= 1) / len(mahal_distances) * 100
                        within_2sigma = np.sum(mahal_distances <= 2) / len(mahal_distances) * 100
                        within_3sigma = np.sum(mahal_distances <= 3) / len(mahal_distances) * 100
                        
                        # 統計情報をテキストで表示（被験者数を削除）
                        stats_text = f'平均値: ({x_mean:.3f}, {y_mean:.3f})\n'
                        stats_text += f'分散: ({x_variance:.4f}, {y_variance:.4f})\n'
                        stats_text += f'1 σ: {within_1sigma:.1f}%\n'
                        stats_text += f'2 σ: {within_2sigma:.1f}%\n'
                        stats_text += f'3 σ: {within_3sigma:.1f}%'
                    except:
                        # マハラノビス距離計算に失敗した場合
                        stats_text = f'平均値: ({x_mean:.3f}, {y_mean:.3f})\n'
                        stats_text += f'分散: ({x_variance:.4f}, {y_variance:.4f})'
                else:
                    stats_text = f'平均値: ({x_mean:.3f}, {y_mean:.3f})\n'
                    stats_text += f'分散: ({x_variance:.4f}, {y_variance:.4f})'
            else:
                stats_text = f'平均値: ({x_mean:.3f}, {y_mean:.3f})\n'
                stats_text += f'分散: ({x_variance:.4f}, {y_variance:.4f})'
            
            # データポイントの分布に基づいて最適なテキスト位置を決定
            def find_best_text_position(x_values, y_values, ax_xlim, ax_ylim):
                """データポイントと重ならない最適なテキスト位置を見つける"""
                # 候補位置（軸座標系での相対位置）
                candidate_positions = [
                    (0.02, 0.98, 'top'),     # 左上
                    (0.02, 0.02, 'bottom'),  # 左下
                    (0.65, 0.98, 'top'),     # 右上（凡例を避ける）
                    (0.65, 0.02, 'bottom'),  # 右下
                ]
                
                # 各候補位置でのデータポイントとの距離を計算
                best_position = candidate_positions[0]
                max_min_distance = 0
                
                for pos_x, pos_y, valign in candidate_positions:
                    # 軸座標系での位置を実際の座標に変換
                    actual_x = ax_xlim[0] + (ax_xlim[1] - ax_xlim[0]) * pos_x
                    actual_y = ax_ylim[0] + (ax_ylim[1] - ax_ylim[0]) * pos_y
                    
                    # 全データポイントとの最小距離を計算
                    distances = []
                    for x_val, y_val in zip(x_values, y_values):
                        dist = np.sqrt((actual_x - x_val)**2 + (actual_y - y_val)**2)
                        distances.append(dist)
                    
                    min_distance = min(distances) if distances else float('inf')
                    
                    # 最も離れた位置を選択
                    if min_distance > max_min_distance:
                        max_min_distance = min_distance
                        best_position = (pos_x, pos_y, valign)
                
                return best_position
            
            # 最適な位置を決定
            best_x, best_y, valign = find_best_text_position(x_values, y_values, 
                                                           ax.get_xlim(), ax.get_ylim())
            
            #ax.text(best_x, best_y, stats_text, fontsize=plt.rcParams['legend.fontsize'], 
            #       transform=ax.transAxes, verticalalignment=valign,
            #       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
        # 実験参加者凡例を追加
        participant_legend_elements = []
        for pid in sorted(participant_ids):
            style = participant_style_map[pid]
            print(int(pid) - 1, ID_table)
            participant_legend_elements.append(plt.Line2D([0], [0], marker=style['marker'], 
                                            color='w', markerfacecolor=style['color'], 
                                            markersize=8, label=f'Person {ID_table[int(pid)-1]}'))
        
        # 範囲・平均値の凡例を追加
        ellipse_legend_elements = [
            plt.Line2D([0], [0], marker='*', color='w', 
                    markerfacecolor='red', markersize=12, label='Mean'),
            plt.Line2D([0], [0], marker='s', color='w', 
                    markerfacecolor='lightblue', markersize=10, label='1 σ'),
            plt.Line2D([0], [0], marker='s', color='w', 
                    markerfacecolor='lightgreen', markersize=10, label='2 σ'),
            plt.Line2D([0], [0], marker='s', color='w', 
                    markerfacecolor='lightyellow', markersize=10, label='3 σ')
        ]
        
        # 2つの凡例を作成（右側に配置、グラフにより近づける）
        legend1 = fig.legend(handles=participant_legend_elements, title='Person', 
                            bbox_to_anchor=(0.81, 0.8), loc='upper left',
                            ncol=1, frameon=True, fancybox=True, shadow=False, fontsize=18,title_fontsize=18)
        legend2 = fig.legend(handles=ellipse_legend_elements, title='Range', 
                            bbox_to_anchor=(0.81, 0.1), loc='lower left',
                            frameon=True, fancybox=True, shadow=False, fontsize=18, title_fontsize=18)
        
        #plt.suptitle(f'オノマトペ「{target_onomatopoeia}」の分布', fontsize=18)
        
        # 凡例用のスペースを確保するために余白を調整（凡例とグラフを近づける）
        plt.subplots_adjust(right=0.78, top=0.85)  # 右側の余白を減らしてグラフと凡例を近づける
    
        # 出力ディレクトリを作成
        output_dir = 'distribution_plots/onomatope'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        filename = f"{filename_prefix}_{target_onomatopoeia}_detailed_{y_label}_{x_label}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()  # showの代わりにcloseで画面表示せずにファイル保存のみ
        
    return fig

def print_data_summary(onomatopoeia_stats):
    """データサマリーを表示"""
    print("=" * 80)
    print("オノマトペ個別データ分布分析サマリー")
    print("=" * 80)
    
    for onomatopoeia, stats in sorted(onomatopoeia_stats.items()):
        print(f"\n【{onomatopoeia}】")
        if 'power_speed' in stats:
            count = stats['power_speed']['count']
            participant_ids = stats['power_speed']['participant_ids']
            print(f"  被験者数: {count}")
            print(f"  被験者ID: {', '.join(map(str, sorted(participant_ids)))}")
            
            for var_pair in ['power_speed', 'power_time', 'speed_time']:
                if var_pair in stats:
                    x_var, y_var = var_pair.split('_')
                    x_mean = stats[var_pair]['x_mean']
                    y_mean = stats[var_pair]['y_mean']
                    x_var_val = stats[var_pair]['x_var']
                    y_var_val = stats[var_pair]['y_var']
                    print(f"  {var_pair}: 重心({x_mean:.3f}, {y_mean:.3f}), 分散({x_var_val:.4f}, {y_var_val:.4f})")

def main():
    """メイン処理"""
    # データ読み込み
    data = load_json_data()
    print(f"読み込んだファイル数: {len(data)}")
    
    # オノマトペデータを抽出
    participants = extract_onomatopoeia_data(data)
    print(f"参加者数: {len(participants)}")
    
    # 統計量と個別データ点を計算
    onomatopoeia_stats = calculate_onomatopoeia_statistics_with_points(participants)
    print(f"分析対象オノマトペ数: {len(onomatopoeia_stats)}")
    
    # データサマリー表示
    print_data_summary(onomatopoeia_stats)
    
    # 2次元分布プロット（全オノマトペ）を作成
    combinations = [
        ('power', 'speed', 'onomatopoeia_distribution_power_vs_speed.png'),
        ('power', 'time', 'onomatopoeia_distribution_power_vs_time.png'),
        ('speed', 'time', 'onomatopoeia_distribution_speed_vs_time.png')
    ]
    
    print("\n" + "=" * 60)
    print("分散＋個別データ分布プロット作成中...")
    print("=" * 60)
    
    for x_var, y_var, filename in combinations:
        print(f"\n{filename}を作成中...")
        fig = create_distribution_plot(onomatopoeia_stats, x_var, y_var, filename)
        print(f"distribution_plots/{filename}が作成されました")
    
    # 全オノマトペの詳細分析
    print("\n" + "=" * 60)
    print("全オノマトペの詳細分析を実行中...")
    print("=" * 60)
    
    # 被験者数順のオノマトペリスト作成
    onomatopoeia_counts = []
    for onomatopoeia, stats in onomatopoeia_stats.items():
        if 'power_speed' in stats:
            count = stats['power_speed']['count']
            onomatopoeia_counts.append((onomatopoeia, count))
    
    onomatopoeia_counts.sort(key=lambda x: x[1], reverse=True)
    
    print("被験者数順のオノマトペ:")
    for onomatopoeia, count in onomatopoeia_counts:
        print(f"  {onomatopoeia}: {count}人")
    
    # 全オノマトペの詳細分析
    print(f"\n全{len(onomatopoeia_counts)}個のオノマトペの詳細分析を実行中...")
    for i, (onomatopoeia, count) in enumerate(onomatopoeia_counts, 1):
        print(f"\n[{i}/{len(onomatopoeia_counts)}] {onomatopoeia} (被験者数: {count}) の詳細分析中...")
        create_detailed_onomatopoeia_plot(onomatopoeia_stats, onomatopoeia, "detail")
        print(f"distribution_plots/onomatope/detail_{onomatopoeia}_detailed.pngが作成されました")
    
    print("\n" + "=" * 60)
    print("すべての分布プロット作成が完了しました!")
    print("=" * 60)

if __name__ == "__main__":
    main()