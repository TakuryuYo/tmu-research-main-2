#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate differences in Speed, Power, Duration for each participant between first and second experiments
"""

import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

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

def extract_participant_onomatopoeia_data(data):
    """Extract onomatopoeia data organized by participant"""
    participant_data = {}
    
    for json_data in data:
        participant_id = json_data["responses"]["【あなたについて】_被験者番号"]
        responses = json_data["responses"]
        
        # Initialize participant data
        if participant_id not in participant_data:
            participant_data[participant_id] = {}
        
        # Extract onomatopoeia data
        for key, value in responses.items():
            if "【オノマトペについて】" in key:
                try:
                    # Parse key to extract onomatopoeia name and feature
                    if 'パワー' in key and '： 「' in key:
                        onomatopoeia = key.split('： 「')[1].split('」')[0]
                        if onomatopoeia not in participant_data[participant_id]:
                            participant_data[participant_id][onomatopoeia] = {}
                        participant_data[participant_id][onomatopoeia]['power'] = float(value) / 100.0
                    
                    elif 'スピード' in key and '： 「' in key:
                        onomatopoeia = key.split('： 「')[1].split('」')[0]
                        if onomatopoeia not in participant_data[participant_id]:
                            participant_data[participant_id][onomatopoeia] = {}
                        participant_data[participant_id][onomatopoeia]['speed'] = float(value) / 100.0
                    
                    elif '時間の長さ（持続性）' in key and '： 「' in key:
                        onomatopoeia = key.split('： 「')[1].split('」')[0]
                        if onomatopoeia not in participant_data[participant_id]:
                            participant_data[participant_id][onomatopoeia] = {}
                        participant_data[participant_id][onomatopoeia]['time'] = float(value) / 100.0
                except (IndexError, ValueError) as e:
                    print(f"Warning: Failed to parse key '{key}' with value '{value}': {e}")
                    continue
    
    return participant_data

def calculate_individual_differences(first_data, second_data):
    """Calculate differences for each participant for each onomatopoeia"""
    differences = {}
    
    # Find common participants
    common_participants = set(first_data.keys()) & set(second_data.keys())
    print(f"Common participants: {sorted(common_participants)}")
    
    for participant_id in common_participants:
        differences[participant_id] = {}
        
        # Find common onomatopoeia for this participant
        first_onomatopoeia = set(first_data[participant_id].keys())
        second_onomatopoeia = set(second_data[participant_id].keys())
        common_onomatopoeia = first_onomatopoeia & second_onomatopoeia
        
        for onomatopoeia in common_onomatopoeia:
            first_vals = first_data[participant_id][onomatopoeia]
            second_vals = second_data[participant_id][onomatopoeia]
            
            # Calculate differences only if all three values exist in both experiments
            if all(key in first_vals for key in ['power', 'speed', 'time']) and \
               all(key in second_vals for key in ['power', 'speed', 'time']):
                
                differences[participant_id][onomatopoeia] = {
                    'power_diff': abs(second_vals['power'] - first_vals['power']),
                    'speed_diff': abs(second_vals['speed'] - first_vals['speed']),
                    'time_diff': abs(second_vals['time'] - first_vals['time'])
                }
    
    return differences

def save_individual_differences_csv(differences):
    """Save individual differences to CSV files for each participant"""
    output_dir = 'difference_analysis'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for participant_id, participant_diffs in differences.items():
        if not participant_diffs:
            continue
            
        # Create DataFrame for this participant
        rows = []
        for onomatopoeia, diffs in participant_diffs.items():
            rows.append({
                'onomatopoeia': onomatopoeia,
                'power_diff': diffs['power_diff'],
                'speed_diff': diffs['speed_diff'],
                'time_diff': diffs['time_diff']
            })
        
        df = pd.DataFrame(rows)
        filename = f"participant_{participant_id}_differences.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"Saved individual differences for participant {participant_id} to {filepath}")

def calculate_average_differences(differences):
    """Calculate average differences for each participant across all onomatopoeia"""
    average_diffs = {}
    
    for participant_id, participant_diffs in differences.items():
        if not participant_diffs:
            continue
            
        # Collect all differences for averaging
        power_diffs = []
        speed_diffs = []
        time_diffs = []
        
        for onomatopoeia, diffs in participant_diffs.items():
            power_diffs.append(diffs['power_diff'])
            speed_diffs.append(diffs['speed_diff'])
            time_diffs.append(diffs['time_diff'])
        
        if power_diffs:  # If we have any data
            average_diffs[participant_id] = {
                'avg_power_diff': np.mean(power_diffs),
                'avg_speed_diff': np.mean(speed_diffs),
                'avg_time_diff': np.mean(time_diffs),
                'onomatopoeia_count': len(power_diffs)
            }
    
    return average_diffs

def save_average_differences_csv(average_diffs):
    """Save average differences to CSV"""
    output_dir = 'difference_analysis'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    rows = []
    for participant_id, diffs in average_diffs.items():
        rows.append({
            'participant_id': participant_id,
            'avg_power_diff': diffs['avg_power_diff'],
            'avg_speed_diff': diffs['avg_speed_diff'],
            'avg_time_diff': diffs['avg_time_diff'],
            'onomatopoeia_count': diffs['onomatopoeia_count']
        })
    
    df = pd.DataFrame(rows)
    filepath = os.path.join(output_dir, "average_differences_summary.csv")
    df.to_csv(filepath, index=False, encoding='utf-8')
    print(f"Saved average differences summary to {filepath}")
    
    return df

def create_difference_plots(average_diffs):
    """Create plots showing participant differences sorted in descending order"""
    output_dir = 'difference_analysis'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    features = ['avg_power_diff', 'avg_speed_diff', 'avg_time_diff']
    feature_labels = ['Average Power Difference', 'Average Speed Difference', 'Average Duration Difference']
    
    # Create a single figure with 3 subplots arranged vertically - square overall shape
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    # Define simplified y-labels for each subplot
    simple_labels = ['Power', 'Speed', 'Duration']
    
    for i, (feature, label, simple_label) in enumerate(zip(features, feature_labels, simple_labels)):
        ax = axes[i]
        
        # Sort participants by this feature in descending order
        sorted_participants = sorted(average_diffs.items(), 
                                   key=lambda x: x[1][feature], 
                                   reverse=True)
        
        participant_ids = [item[0] for item in sorted_participants]
        values = [item[1][feature] for item in sorted_participants]
        
        # Create participant labels using ID_table (short form)
        participant_labels = [ID_table[int(pid)-1] for pid in participant_ids]
        
        # Create plot - since we're using absolute values, all bars are positive (blue)
        bars = ax.bar(range(len(participant_labels)), values, width=0.6,
                     color='blue', alpha=0.7, edgecolor='black', linewidth=1)
        
        # Customize plot
        if i == 2:  # Only add xlabel to bottom plot
            ax.set_xlabel('Participant')
        ax.set_ylabel(simple_label)
        ax.set_xticks(range(len(participant_labels)))
        ax.set_xticklabels(participant_labels)
        ax.grid(True, alpha=0.3)
        
        # Set y-axis limits to show the maximum value clearly
        max_val = max(values) if values else 1
        y_max = max_val * 1.1  # Add 10% padding above max value
        ax.set_ylim(0, y_max)
        
        # Explicitly set y-axis ticks to ensure 0 and max are shown
        import numpy as np
        # Create ticks that include 0 and the upper limit
        n_ticks = 5
        tick_values = np.linspace(0, y_max, n_ticks)
        ax.set_yticks(tick_values)
        # Format to show reasonable number of decimal places
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))
    
    # Add overall y-label for the entire figure
    fig.text(0.04, 0.5, 'Average Difference', va='center', rotation='vertical', fontsize=30)
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.20)  # Make room for the overall y-label and y-axis values
    
    # Save combined plot
    filename = "combined_differences_plot.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined differences plot to {filepath}")
    
    # Also create individual plots for compatibility
    for feature, label in zip(features, feature_labels):
        # Sort participants by this feature in descending order
        sorted_participants = sorted(average_diffs.items(), 
                                   key=lambda x: x[1][feature], 
                                   reverse=True)
        
        participant_ids = [item[0] for item in sorted_participants]
        values = [item[1][feature] for item in sorted_participants]
        
        # Create participant labels using ID_table (short form)
        participant_labels = [ID_table[int(pid)-1] for pid in participant_ids]
        
        # Create individual plot
        fig, ax = plt.subplots(figsize=(8, 8))
        bars = ax.bar(range(len(participant_labels)), values, 
                     color='blue', alpha=0.7, edgecolor='black', linewidth=1)
        
        # Customize plot
        ax.set_xlabel('Participant')
        ax.set_ylabel(label)
        ax.set_xticks(range(len(participant_labels)))
        ax.set_xticklabels(participant_labels)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save individual plot
        filename = f"{feature}_sorted_plot.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {label} plot to {filepath}")

def print_summary_statistics(average_diffs):
    """Print summary statistics"""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    features = ['avg_power_diff', 'avg_speed_diff', 'avg_time_diff']
    feature_names = ['Power', 'Speed', 'Duration']
    
    for feature, name in zip(features, feature_names):
        values = [diffs[feature] for diffs in average_diffs.values()]
        print(f"\n{name} Differences:")
        print(f"  Mean: {np.mean(values):.4f}")
        print(f"  Std:  {np.std(values, ddof=1):.4f}")
        print(f"  Min:  {np.min(values):.4f}")
        print(f"  Max:  {np.max(values):.4f}")
        
        # Show participants with highest and lowest differences
        sorted_participants = sorted(average_diffs.items(), 
                                   key=lambda x: x[1][feature], 
                                   reverse=True)
        highest = sorted_participants[0]
        lowest = sorted_participants[-1]
        
        print(f"  Highest: Person {ID_table[int(highest[0])-1]} ({highest[1][feature]:.4f})")
        print(f"  Lowest:  Person {ID_table[int(lowest[0])-1]} ({lowest[1][feature]:.4f})")

def main():
    """Main processing"""
    print("=" * 80)
    print("ONOMATOPOEIA DIFFERENCE ANALYSIS")
    print("=" * 80)
    
    # Load data from both experiments
    print("\nLoading data...")
    data_first = load_json_data("../../20250615_pre_exam_questionare/raw_data")
    data_second = load_json_data("../questionaire_result")
    print(f"First experiment: {len(data_first)} files")
    print(f"Second experiment: {len(data_second)} files")
    
    # Extract participant data
    print("\nExtracting participant data...")
    first_participant_data = extract_participant_onomatopoeia_data(data_first)
    second_participant_data = extract_participant_onomatopoeia_data(data_second)
    print(f"First experiment participants: {list(first_participant_data.keys())}")
    print(f"Second experiment participants: {list(second_participant_data.keys())}")
    
    # Calculate individual differences
    print("\nCalculating individual differences...")
    differences = calculate_individual_differences(first_participant_data, second_participant_data)
    
    # Save individual differences to CSV
    print("\nSaving individual differences to CSV...")
    save_individual_differences_csv(differences)
    
    # Calculate average differences
    print("\nCalculating average differences...")
    average_diffs = calculate_average_differences(differences)
    
    # Save average differences to CSV
    print("\nSaving average differences summary...")
    avg_df = save_average_differences_csv(average_diffs)
    
    # Create plots
    print("\nCreating difference plots...")
    create_difference_plots(average_diffs)
    
    # Print summary statistics
    print_summary_statistics(average_diffs)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETED!")
    print("Results saved in 'difference_analysis' directory:")
    print("- Individual CSV files for each participant")
    print("- average_differences_summary.csv")
    print("- Sorted bar plots for each feature")
    print("=" * 80)

if __name__ == "__main__":
    main()