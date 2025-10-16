#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare onomatopoeia distribution between first and second experiments
First experiment data is marked with 'o', second experiment data is marked with 'X'
"""

import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import os
from scipy.spatial.distance import mahalanobis

plt.rcParams['font.family'] = ['Times New Roman', 'IPAexGothic']
plt.rcParams['font.sans-serif'] = ['IPAexGothic']
plt.rcParams['font.serif'] = ['Times New Roman']

plt.rcParams['axes.labelsize'] = 30
plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['ytick.labelsize'] = 30
plt.rcParams['axes.titlesize'] = 16

ID_table = [
    "A", "B", "C", "D", "E", "F"
]

def load_json_data(directory):
    """Load all JSON files from specified directory"""
    json_files = glob.glob(f"{directory}/*.json")
    data = []
    
    for file in json_files:
        if '.bak' in file:  # Skip backup files
            continue
        with open(file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            data.append(json_data)
    
    return data

def extract_onomatopoeia_data(data):
    """Extract and organize onomatopoeia data"""
    participants = []
    
    for json_data in data:
        participant_id = json_data["responses"]["【あなたについて】_被験者番号"]
        responses = json_data["responses"]
        
        # Extract onomatopoeia data
        onomatopoeia_data = {}
        
        # Extract power, speed, time values
        for key, value in responses.items():
            if "【オノマトペについて】" in key:
                try:
                    # Parse key to extract onomatopoeia name and feature
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
    """Calculate statistics and individual data points for each onomatopoeia"""
    # Aggregate data for each onomatopoeia
    onomatopoeia_stats = {}
    
    # Collect all onomatopoeia
    all_onomatopoeia = set()
    for participant in participants:
        all_onomatopoeia.update(participant['onomatopoeia'].keys())
    
    for onomatopoeia in all_onomatopoeia:
        # Store data for each participant
        participant_data = []
        
        # Collect data across participants
        for participant in participants:
            if onomatopoeia in participant['onomatopoeia']:
                values = participant['onomatopoeia'][onomatopoeia]
                # Add only if all three values exist
                if 'power' in values and 'speed' in values and 'time' in values:
                    participant_data.append({
                        'participant_id': participant['id'],
                        'power': values['power'],
                        'speed': values['speed'],
                        'time': values['time']
                    })
        
        if len(participant_data) >= 2:  # Need at least 2 participants
            # Calculate statistics for each variable
            stats = {}
            
            for var_name in ['power', 'speed', 'time']:
                var_values = [data[var_name] for data in participant_data]
                stats[var_name] = {
                    'mean': np.mean(var_values),
                    'std': np.std(var_values, ddof=1),
                    'var': np.var(var_values, ddof=1),
                    'count': len(var_values),
                    'values': var_values
                }
            
            # Calculate 2D combination statistics
            for var_pair in [('power', 'speed'), ('power', 'time'), ('speed', 'time')]:
                x_var, y_var = var_pair
                pair_key = f"{x_var}_{y_var}"
                
                # Treat as 2D coordinates
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

def create_detailed_comparison_plot(onomatopoeia_stats_first, onomatopoeia_stats_second, target_onomatopoeia, filename_prefix):
    """Create detailed comparison plot for specific onomatopoeia between first and second experiments"""
    
    # Check if data exists in both experiments
    if target_onomatopoeia not in onomatopoeia_stats_first and target_onomatopoeia not in onomatopoeia_stats_second:
        print(f"Onomatopoeia '{target_onomatopoeia}' not found in either experiment")
        return
    
    # Get stats for both experiments (use empty dict if not available)
    stats_first = onomatopoeia_stats_first.get(target_onomatopoeia, {})
    stats_second = onomatopoeia_stats_second.get(target_onomatopoeia, {})
    
    # Collect all participant IDs from both experiments
    all_participant_ids = set()
    if stats_first:
        all_participant_ids.update(stats_first.get('power_speed', {}).get('participant_ids', []))
    if stats_second:
        all_participant_ids.update(stats_second.get('power_speed', {}).get('participant_ids', []))
    
    # Participant style mapping - same colors for same participants across experiments
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    participant_style_map = {}
    for i, pid in enumerate(sorted(all_participant_ids)):
        participant_style_map[pid] = {
            'color': colors[i % len(colors)]
        }
    
    combinations = [
        ('power', 'speed', 'Power', 'Speed', 0),
        ('power', 'time', 'Power', 'Duration', 1),
        ('speed', 'time', 'Speed', 'Duration', 2)
    ]
    
    for x_var, y_var, x_label, y_label, idx in combinations:
        fig, ax = plt.subplots(1, 1, figsize=(9, 8))
        
        pair_key = f"{x_var}_{y_var}"
        
        # Plot first experiment data (circle markers)
        if pair_key in stats_first:
            x_values_first = stats_first[pair_key]['x_values']
            y_values_first = stats_first[pair_key]['y_values']
            participant_ids_first = stats_first[pair_key]['participant_ids']
            
            for x_val, y_val, pid in zip(x_values_first, y_values_first, participant_ids_first):
                style = participant_style_map[pid]
                ax.scatter(x_val, y_val, c=[style['color']], s=100, marker='o', 
                          alpha=0.8, edgecolors='black', linewidth=0.5, zorder=3)
        
        # Plot second experiment data (simple x markers)
        if pair_key in stats_second:
            x_values_second = stats_second[pair_key]['x_values']
            y_values_second = stats_second[pair_key]['y_values']
            participant_ids_second = stats_second[pair_key]['participant_ids']
            
            for x_val, y_val, pid in zip(x_values_second, y_values_second, participant_ids_second):
                style = participant_style_map[pid]
                ax.scatter(x_val, y_val, c=[style['color']], s=100, marker='D', 
                          alpha=0.8, edgecolors='black', linewidth=0.5, zorder=3)
        
        # Connect first and second experiment data points with lines
        if pair_key in stats_first and pair_key in stats_second:
            # Create dictionaries for quick lookup
            first_data = {}
            for x_val, y_val, pid in zip(stats_first[pair_key]['x_values'], 
                                       stats_first[pair_key]['y_values'], 
                                       stats_first[pair_key]['participant_ids']):
                first_data[pid] = (x_val, y_val)
            
            second_data = {}
            for x_val, y_val, pid in zip(stats_second[pair_key]['x_values'], 
                                        stats_second[pair_key]['y_values'], 
                                        stats_second[pair_key]['participant_ids']):
                second_data[pid] = (x_val, y_val)
            
            # Draw connecting lines for participants present in both experiments
            for pid in first_data.keys():
                if pid in second_data:
                    style = participant_style_map[pid]
                    x1, y1 = first_data[pid]
                    x2, y2 = second_data[pid]
                    ax.plot([x1, x2], [y1, y2], color=style['color'], alpha=0.7, 
                           linewidth=1.5, zorder=2)
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        
        # Create legend elements
        participant_legend_elements = []
        for pid in sorted(all_participant_ids):
            style = participant_style_map[pid]
            participant_legend_elements.append(plt.Line2D([0], [0], marker='o', 
                                            color='w', markerfacecolor=style['color'], 
                                            markersize=8, label=f'Person {ID_table[int(pid)-1]}'))
        
        # Experiment legend elements
        experiment_legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', 
                    markerfacecolor='gray', markersize=10, label='First Experiment'),
            plt.Line2D([0], [0], marker='D', color='w', 
                    markerfacecolor='gray', markersize=10, label='Second Experiment')
        ]
        
        # Create two legends
        legend1 = fig.legend(handles=participant_legend_elements, title='Person', 
                            bbox_to_anchor=(0.81, 0.8), loc='upper left',
                            ncol=1, frameon=True, fancybox=True, shadow=False, 
                            fontsize=18, title_fontsize=18)
        legend2 = fig.legend(handles=experiment_legend_elements, title='Experiment', 
                            bbox_to_anchor=(0.81, 0.1), loc='lower left',
                            frameon=True, fancybox=True, shadow=False, 
                            fontsize=18, title_fontsize=18)
        
        plt.subplots_adjust(right=0.78, top=0.85)
    
        # Create output directory
        output_dir = 'compare_detail_onomatope'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        filename = f"{filename_prefix}_{target_onomatopoeia}_compare_{y_label}_{x_label}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main processing"""
    # Load data from both experiments
    print("Loading data from second experiment (../questionaire_result)...")
    data_second = load_json_data("../questionaire_result")
    print(f"Loaded {len(data_second)} files from second experiment")
    
    print("Loading data from first experiment (../../20250615_pre_exam_questionare/raw_data)...")
    data_first = load_json_data("../../20250615_pre_exam_questionare/raw_data")
    print(f"Loaded {len(data_first)} files from first experiment")
    
    # Extract onomatopoeia data
    participants_first = extract_onomatopoeia_data(data_first)
    participants_second = extract_onomatopoeia_data(data_second)
    print(f"First experiment participants: {len(participants_first)}")
    print(f"Second experiment participants: {len(participants_second)}")
    
    # Calculate statistics
    onomatopoeia_stats_first = calculate_onomatopoeia_statistics_with_points(participants_first)
    onomatopoeia_stats_second = calculate_onomatopoeia_statistics_with_points(participants_second)
    print(f"First experiment onomatopoeia count: {len(onomatopoeia_stats_first)}")
    print(f"Second experiment onomatopoeia count: {len(onomatopoeia_stats_second)}")
    
    # Get all unique onomatopoeia from both experiments
    all_onomatopoeia = set(onomatopoeia_stats_first.keys()) | set(onomatopoeia_stats_second.keys())
    print(f"Total unique onomatopoeia: {len(all_onomatopoeia)}")
    
    # Create comparison plots for all onomatopoeia
    print("\n" + "=" * 60)
    print("Creating comparison plots...")
    print("=" * 60)
    
    for i, onomatopoeia in enumerate(sorted(all_onomatopoeia), 1):
        print(f"[{i}/{len(all_onomatopoeia)}] Creating comparison plot for '{onomatopoeia}'...")
        create_detailed_comparison_plot(onomatopoeia_stats_first, onomatopoeia_stats_second, 
                                      onomatopoeia, "detail")
        print(f"  compare_detail_onomatope/detail_{onomatopoeia}_compare_*.png created")
    
    print("\n" + "=" * 60)
    print("All comparison plots created successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()