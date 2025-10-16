#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hungarianアルゴリズムによるクラスタ対応分析
- session1/session2/session3全てとの比較
- 個人別混同行列を1枚ずつ保存
- 全員分をまとめた混同行列も保存
- 個人別・全員別でAcc/F1等のスコアを算出
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    normalized_mutual_info_score, 
    adjusted_rand_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from scipy.optimize import linear_sum_assignment
import os
import japanize_matplotlib

plt.rcParams['font.family'] = ['Times New Roman', 'IPAexGothic']
plt.rcParams['font.sans-serif'] = ['IPAexGothic']
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.labelsize'] = 17
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['axes.titlesize'] = 16

def load_sensor_clustering_results(sensor_person_id, session, n_clusters=6):
    """センサーデータのクラスタリング結果を読み込み"""
    sensor_result_path = f"result/sensor_clustering_results_{sensor_person_id}_session{session}_{n_clusters}clusters.csv"
    
    if not os.path.exists(sensor_result_path):
        print(f"センサーデータ結果が見つかりません: {sensor_result_path}")
        return None
    
    try:
        df = pd.read_csv(sensor_result_path, encoding='utf-8-sig')
        return df
    except Exception as e:
        print(f"センサーデータ読み込みエラー: {e}")
        return None

def load_questionnaire_clustering_results(person_name, n_clusters=6, method='ward'):
    """アンケートデータのクラスタリング結果を読み込み"""
    questionnaire_result_path = f"clustering_dend/result/clustering_results_{method}_{n_clusters}clusters_{person_name}.csv"
    
    if not os.path.exists(questionnaire_result_path):
        print(f"アンケートデータ結果が見つかりません: {questionnaire_result_path}")
        return None
    
    try:
        df = pd.read_csv(questionnaire_result_path, encoding='utf-8-sig')
        return df
    except Exception as e:
        print(f"アンケートデータ読み込みエラー: {e}")
        return None

def get_person_name_mapping():
    """アンケートデータとセンサーデータの人名マッピングを取得"""
    return {
        "Iさん": "inamura",
        "Uさん": "utsumi", 
        "Kさん": "kawano",
        "Hさん": "hatanaka",
        "Wさん": "watanabe",
        "菊池さん": "kikuchi",
       # "Oさん": "okamoto"
    }
def get_person_id_mapping():
    """アンケートデータとセンサーデータの人名マッピングを取得"""
    return {
        "Iさん": "1",
        "Uさん": "2", 
        "Kさん": "3",
        "Hさん": "4",
        "Wさん": "5",
        "菊池さん": "6",
       # "Oさん": "8"
    }


def normalize_onomatopoeia_name(name):
    """オノマトペ名を正規化"""
    import re
    normalized = re.sub(r'[（(].*?[）)]', '', name)
    normalized = normalized.strip()
    return normalized

def align_clustering_results(sensor_df, questionnaire_df):
    """二つのクラスタリング結果を共通のオノマトペで整列"""
    # オノマトペ名を正規化
    sensor_df_normalized = sensor_df.copy()
    questionnaire_df_normalized = questionnaire_df.copy()
    
    sensor_df_normalized['オノマトペ_正規化'] = sensor_df_normalized['オノマトペ'].apply(normalize_onomatopoeia_name)
    questionnaire_df_normalized['オノマトペ_正規化'] = questionnaire_df_normalized['オノマトペ'].apply(normalize_onomatopoeia_name)
    
    # 共通のオノマトペを特定
    sensor_onomatopoeia = set(sensor_df_normalized['オノマトペ_正規化'])
    questionnaire_onomatopoeia = set(questionnaire_df_normalized['オノマトペ_正規化'])
    common_onomatopoeia = sensor_onomatopoeia.intersection(questionnaire_onomatopoeia)
    
    if len(common_onomatopoeia) < 3:
        return None, None, None
    
    # 共通オノマトペでフィルタリングしてソート
    sensor_common = sensor_df_normalized[sensor_df_normalized['オノマトペ_正規化'].isin(common_onomatopoeia)]
    questionnaire_common = questionnaire_df_normalized[questionnaire_df_normalized['オノマトペ_正規化'].isin(common_onomatopoeia)]
    
    sensor_common = sensor_common.sort_values('オノマトペ_正規化').reset_index(drop=True)
    questionnaire_common = questionnaire_common.sort_values('オノマトペ_正規化').reset_index(drop=True)
    
    return sensor_common, questionnaire_common, list(common_onomatopoeia)

def get_cluster_labels(df):
    """データフレームからクラスタラベルを取得"""
    if 'クラスターID' in df.columns:
        return df['クラスターID'].values
    elif 'クラスタID' in df.columns:
        return df['クラスタID'].values
    else:
        raise ValueError(f"クラスタラベルの列が見つかりません: {df.columns.tolist()}")

def hungarian_cluster_alignment(sensor_labels, questionnaire_labels):
    """ハンガリアンアルゴリズムによる最適クラスタ対応"""
    
    # 混同行列を作成
    cm = confusion_matrix(sensor_labels, questionnaire_labels)
    
    # コスト行列（最大化問題を最小化問題に変換）
    cost_matrix = cm.max() - cm
    
    # ハンガリアンアルゴリズムで最適対応を計算
    sensor_indices, questionnaire_indices = linear_sum_assignment(cost_matrix)
    
    # 対応関係を辞書で保存
    alignment = dict(zip(sensor_indices + 1, questionnaire_indices + 1))  # 1-indexedに変換
    
    # アンケートラベルを対応に基づいて再マッピング
    aligned_questionnaire_labels = questionnaire_labels.copy()
    reverse_alignment = {v: k for k, v in alignment.items()}
    
    for original, aligned in reverse_alignment.items():
        mask = questionnaire_labels == original
        aligned_questionnaire_labels[mask] = aligned
    
    # 対応後の混同行列
    aligned_cm = confusion_matrix(sensor_labels, aligned_questionnaire_labels)
    
    return alignment, aligned_questionnaire_labels, aligned_cm

def calculate_classification_metrics(y_true, y_pred):
    """分類指標を計算"""
    metrics = {}
    
    # 基本指標
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # マクロ平均
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # ミクロ平均
    metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    # 重み付き平均
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # その他の指標
    metrics['nmi'] = normalized_mutual_info_score(y_true, y_pred)
    metrics['ari'] = adjusted_rand_score(y_true, y_pred)
    
    return metrics

def plot_confusion_matrix(cm, title, person_name, session, n_clusters, output_dir):
    """混同行列を可視化して保存"""
    plt.figure(figsize=(8, 8))  # 正方形にする
    
    # ヒートマップを作成
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f'A{i}' for i in range(1, n_clusters+1)],
                yticklabels=[f'S{i}' for i in range(1, n_clusters+1)],
                square=True,  # 正方形のセルにする
                annot_kws={'size': 20})  # 数字のサイズを大きくする
    
    plt.title(f'{title}\n{person_name} - Session {session}')
    plt.xlabel('Questionnaire Class')
    plt.ylabel('Sensor Data Class')
    plt.tight_layout()
    
    # 保存
    filename = f'confusion_matrix_{person_name}_session{session}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath

def plot_combined_confusion_matrix(combined_cm, session, n_clusters, output_dir):
    """全員分を合計した混同行列を可視化"""
    plt.figure(figsize=(10, 10))  # 正方形にする
    
    # ヒートマップを作成
    sns.heatmap(combined_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f'A{i}' for i in range(1, n_clusters+1)],
                yticklabels=[f'S{i}' for i in range(1, n_clusters+1)],
                square=True,  # 正方形のセルにする
                annot_kws={'size': 20})  # 数字のサイズを大きくする
    
    plt.xlabel('Questionnaire Class')
    plt.ylabel('Sensor Data Class')
    plt.tight_layout()
    
    # 保存
    filename = f'confusion_matrix_combined_session{session}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath

def save_cluster_onomatopoeia_details(sensor_df, questionnaire_df, aligned_questionnaire_labels, person_name, session, output_dir):
    """クラスタ別オノマトペ詳細をCSVに保存"""
    
    # センサークラスタとアンケートクラスタ（Hungarian対応後）の情報を整理
    sensor_labels = get_cluster_labels(sensor_df)
    original_questionnaire_labels = get_cluster_labels(questionnaire_df)
    
    cluster_details = []
    
    for i in range(len(sensor_df)):
        cluster_details.append({
            'オノマトペ': sensor_df.iloc[i]['オノマトペ_正規化'],
            'センサークラスタ': sensor_labels[i],
            'アンケートクラスタ_元': original_questionnaire_labels[i],
            'アンケートクラスタ_Hungarian対応後': aligned_questionnaire_labels[i],
            'センサー_パワー': sensor_df.iloc[i]['パワー'],
            'センサー_持続性': sensor_df.iloc[i]['持続性'],
            'センサー_スピード': sensor_df.iloc[i]['スピード'],
            'アンケート_パワー': questionnaire_df.iloc[i]['パワー'],
            'アンケート_持続性': questionnaire_df.iloc[i]['持続性'],
            'アンケート_スピード': questionnaire_df.iloc[i]['スピード'],
            'クラスタ一致': sensor_labels[i] == aligned_questionnaire_labels[i]
        })
    
    # DataFrameに変換
    details_df = pd.DataFrame(cluster_details)
    
    # 保存
    filename = f'cluster_onomatopoeia_details_{person_name}_session{session}.csv'
    filepath = os.path.join(output_dir, filename)
    details_df.to_csv(filepath, index=False, encoding='utf-8-sig')
    
    return filepath

def save_cluster_summary_table(sensor_df, questionnaire_df, aligned_questionnaire_labels, person_name, session, output_dir):
    """見やすい形式のクラスタ別オノマトペ表をCSVに保存（センサーとアンケートを統合）"""
    
    sensor_labels = get_cluster_labels(sensor_df)
    
    # センサークラスタ別にオノマトペをグループ化
    sensor_clusters = {}
    for i in range(len(sensor_df)):
        cluster_id = sensor_labels[i]
        onomatopoeia = sensor_df.iloc[i]['オノマトペ_正規化']
        
        if cluster_id not in sensor_clusters:
            sensor_clusters[cluster_id] = []
        sensor_clusters[cluster_id].append(onomatopoeia)
    
    # アンケートクラスタ（Hungarian対応後）別にオノマトペをグループ化  
    questionnaire_clusters = {}
    for i in range(len(questionnaire_df)):
        cluster_id = aligned_questionnaire_labels[i]
        onomatopoeia = questionnaire_df.iloc[i]['オノマトペ_正規化']
        
        if cluster_id not in questionnaire_clusters:
            questionnaire_clusters[cluster_id] = []
        questionnaire_clusters[cluster_id].append(onomatopoeia)
    
    # 統合された表を作成（センサーとアンケート両方を含む）
    combined_summary_data = []
    
    # センサークラスタを追加
    for cluster_id in sorted(sensor_clusters.keys()):
        row = {'データ種別': 'センサー', 'class': cluster_id}
        for i, onomatopoeia in enumerate(sensor_clusters[cluster_id]):
            row[f'オノマトペ{i+1}'] = onomatopoeia
        combined_summary_data.append(row)
    
    # 空行を追加
    combined_summary_data.append({})
    
    # アンケートクラスタ（Hungarian対応後）を追加
    for cluster_id in sorted(questionnaire_clusters.keys()):
        row = {'データ種別': 'アンケート(Hungarian対応後)', 'class': cluster_id}
        for i, onomatopoeia in enumerate(questionnaire_clusters[cluster_id]):
            row[f'オノマトペ{i+1}'] = onomatopoeia
        combined_summary_data.append(row)
    
    combined_summary_df = pd.DataFrame(combined_summary_data)
    
    # 統合クラスタ表を保存
    combined_filename = f'cluster_summary_{person_name}_session{session}.csv'
    combined_filepath = os.path.join(output_dir, combined_filename)
    combined_summary_df.to_csv(combined_filepath, index=False, encoding='utf-8-sig')
    
    return combined_filepath

def analyze_session(session, person_mapping, person_id_mapping, n_clusters=6, method='ward', output_dir='session_analysis'):
    """特定のセッションを分析"""
    
    print(f"\n{'='*80}")
    print(f"Session {session} 分析開始")
    print(f"{'='*80}")
    
    session_results = []
    all_sensor_labels = []
    all_questionnaire_labels = []
    combined_cm = np.zeros((n_clusters, n_clusters), dtype=int)
    
    session_output_dir = os.path.join(output_dir, f'session{session}')
    os.makedirs(session_output_dir, exist_ok=True)
    
    # 各被験者について分析

    
    for questionnaire_name, sensor_name in person_mapping.items():
        try:
            print(f"\n分析中: {questionnaire_name} ({sensor_name})")
            
            # データ読み込み
            sensor_person_id = person_id_mapping.get(questionnaire_name)
            if not sensor_person_id:
                print(f"  警告: {questionnaire_name}に対応するIDが見つかりません。スキップします。")
                continue

            sensor_df = load_sensor_clustering_results(sensor_person_id, session, n_clusters)
            questionnaire_df = load_questionnaire_clustering_results(questionnaire_name, n_clusters, method)
            
            if sensor_df is None or questionnaire_df is None:
                print(f"  データ読み込み失敗: {questionnaire_name}")
                continue
            
            # データ整列
            sensor_aligned, questionnaire_aligned, common_onomatopoeia = align_clustering_results(sensor_df, questionnaire_df)
            
            if sensor_aligned is None:
                print(f"  データ整列失敗: {questionnaire_name}")
                continue
            
            # クラスタラベル取得
            sensor_labels = get_cluster_labels(sensor_aligned)
            questionnaire_labels = get_cluster_labels(questionnaire_aligned)
            
            # Hungarianアルゴリズムによる対応
            alignment, aligned_questionnaire_labels, aligned_cm = hungarian_cluster_alignment(sensor_labels, questionnaire_labels)
            
            # 個人別指標を計算
            individual_metrics = calculate_classification_metrics(sensor_labels, aligned_questionnaire_labels)
            
            # 個人別混同行列を保存
            cm_path = plot_confusion_matrix(
                aligned_cm, 
                f'Hungarian対応後混同行列', 
                questionnaire_name, 
                session, 
                n_clusters, 
                session_output_dir
            )
            
            # クラスタ別オノマトペリストを保存
            cluster_onomatopoeia_path = save_cluster_onomatopoeia_details(
                sensor_aligned, questionnaire_aligned, aligned_questionnaire_labels,
                questionnaire_name, session, session_output_dir
            )
            
            # 見やすい形式のクラスタ別オノマトペ表を保存
            cluster_summary_path = save_cluster_summary_table(
                sensor_aligned, questionnaire_aligned, aligned_questionnaire_labels,
                questionnaire_name, session, session_output_dir
            )
            
            print(f"  混同行列保存: {cm_path}")
            print(f"  クラスタ詳細保存: {cluster_onomatopoeia_path}")
            print(f"  クラスタ要約保存: {cluster_summary_path}")
            print(f"  Accuracy: {individual_metrics['accuracy']:.4f}")
            print(f"  F1-macro: {individual_metrics['f1_macro']:.4f}")
            print(f"  NMI: {individual_metrics['nmi']:.4f}")
            
            # 結果を保存
            result_data = {
                'session': session,
                'questionnaire_person_name': questionnaire_name,
                'sensor_person_name': sensor_name,
                'common_onomatopoeia_count': len(common_onomatopoeia),
                'alignment': str(alignment),
                **individual_metrics
            }
            session_results.append(result_data)
            
            # 全員分のデータを蓄積
            all_sensor_labels.extend(sensor_labels.tolist())
            all_questionnaire_labels.extend(aligned_questionnaire_labels.tolist())
            combined_cm += aligned_cm
            
        except Exception as e:
            print(f"  エラー（{questionnaire_name}）: {e}")
            continue
    
    # 全員分の混同行列を保存
    if len(session_results) > 0:
        combined_cm_path = plot_combined_confusion_matrix(
            combined_cm, 
            session, 
            n_clusters, 
            session_output_dir
        )
        print(f"\n全員合計混同行列保存: {combined_cm_path}")
        
        # 全員分の指標を計算
        combined_metrics = calculate_classification_metrics(
            np.array(all_sensor_labels), 
            np.array(all_questionnaire_labels)
        )
        
        print(f"\n--- Session {session} 全員合計指標 ---")
        print(f"Accuracy: {combined_metrics['accuracy']:.4f}")
        print(f"F1-macro: {combined_metrics['f1_macro']:.4f}")
        print(f"F1-micro: {combined_metrics['f1_micro']:.4f}")
        print(f"F1-weighted: {combined_metrics['f1_weighted']:.4f}")
        print(f"Precision-macro: {combined_metrics['precision_macro']:.4f}")
        print(f"Recall-macro: {combined_metrics['recall_macro']:.4f}")
        print(f"NMI: {combined_metrics['nmi']:.4f}")
        print(f"ARI: {combined_metrics['ari']:.4f}")
        
        # 全員分指標を結果に追加
        combined_result = {
            'session': session,
            'analysis_type': 'combined_all_participants',
            'total_participants': len(session_results),
            'total_samples': len(all_sensor_labels),
            **{f'combined_{k}': v for k, v in combined_metrics.items()}
        }
        session_results.append(combined_result)
    
    return session_results

def create_session_comparison_summary(all_results, output_dir):
    """セッション間比較サマリーを作成"""
    
    # 個人別結果とまとめ結果を分離
    individual_results = [r for r in all_results if 'analysis_type' not in r]
    combined_results = [r for r in all_results if r.get('analysis_type') == 'combined_all_participants']
    
    # 個人別結果をデータフレーム化
    individual_df = pd.DataFrame(individual_results)
    combined_df = pd.DataFrame(combined_results)
    
    if len(individual_df) > 0:
        # 個人別結果の保存
        individual_csv_path = os.path.join(output_dir, 'individual_results_all_sessions.csv')
        individual_df.to_csv(individual_csv_path, index=False, encoding='utf-8-sig')
        print(f"\n個人別結果保存: {individual_csv_path}")
        
        # セッション別個人平均の表示
        print(f"\n{'='*80}")
        print("=== セッション別個人平均指標 ===")
        print(f"{'='*80}")
        
        metrics_to_show = ['accuracy', 'f1_macro', 'f1_micro', 'precision_macro', 'recall_macro', 'nmi', 'ari']
        
        for session in sorted(individual_df['session'].unique()):
            session_data = individual_df[individual_df['session'] == session]
            print(f"\nSession {session} (参加者数: {len(session_data)}):")
            
            for metric in metrics_to_show:
                if metric in session_data.columns:
                    mean_val = session_data[metric].mean()
                    std_val = session_data[metric].std()
                    print(f"  {metric}: {mean_val:.4f} (±{std_val:.4f})")
    
    if len(combined_df) > 0:
        # 全員合計結果の保存
        combined_csv_path = os.path.join(output_dir, 'combined_results_all_sessions.csv')
        combined_df.to_csv(combined_csv_path, index=False, encoding='utf-8-sig')
        print(f"\n全員合計結果保存: {combined_csv_path}")
        
        # セッション別全員合計指標の表示
        print(f"\n{'='*80}")
        print("=== セッション別全員合計指標 ===")
        print(f"{'='*80}")
        
        combined_metrics = [col for col in combined_df.columns if col.startswith('combined_')]
        
        for session in sorted(combined_df['session'].unique()):
            session_data = combined_df[combined_df['session'] == session].iloc[0]
            print(f"\nSession {session} (総サンプル数: {session_data['total_samples']}):")
            
            for metric in combined_metrics:
                if metric in combined_df.columns:
                    value = session_data[metric]
                    clean_metric = metric.replace('combined_', '')
                    print(f"  {clean_metric}: {value:.4f}")

def main():
    """メイン処理"""
    print("=== Session1/2/3 全てとの Hungarian対応分析 ===")
    
    # 設定
    sessions = [1, 2, 3]
    n_clusters = 6
    method = 'ward'
    output_dir = 'hungarian_analysis'
    
    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 人名マッピング
    person_mapping = get_person_name_mapping()
    
    person_id = get_person_id_mapping()

    # 全セッションの結果を保存
    all_results = []
    
    # 各セッションを分析
    for session in sessions:
        session_results = analyze_session(session, person_mapping, person_id, n_clusters, method, output_dir)
        all_results.extend(session_results)
    
    # セッション間比較サマリーを作成
    create_session_comparison_summary(all_results, output_dir)
    
    print(f"\n{'='*80}")
    print("=== 全セッション分析完了 ===")
    print(f"結果保存ディレクトリ: {output_dir}")
    print("- 個人別混同行列: session1/session2/session3/")
    print("- 全員合計混同行列: session1/session2/session3/")
    print("- 統合結果CSV: individual_results_all_sessions.csv")
    print("- 全員合計CSV: combined_results_all_sessions.csv")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()