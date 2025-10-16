#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Max-speed専用アノテーションツール

`annotation/annotation_tool.py` の機能を再利用しつつ、
ファイル名に `max-speed` を含むサンプルのみを対象にします。
結果は `annotation_max-speed_results/` 以下に保存されます。
"""

import os
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
ANNOTATION_DIR = REPO_ROOT / "annotation"

if str(ANNOTATION_DIR) not in sys.path:
    sys.path.insert(0, str(ANNOTATION_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from annotation_tool import WaveformAnnotator


class MaxSpeedWaveformAnnotator(WaveformAnnotator):
    """max-speedサンプル専用のアノテーター"""

    RESULTS_DIR = "annotation_max-speed_results"

    def load_dataset(self):
        """データセットを読み込んだ後、max-speedサンプルだけに絞り込む"""
        super().load_dataset()

        if not self.dataset or "data" not in self.dataset:
            print("Warning: Dataset is empty — nothing to filter.")
            return

        all_samples = self.dataset["data"]
        filtered = [
            sample
            for sample in all_samples
            if "filename" in sample
            and isinstance(sample["filename"], str)
            and "max-speed" in sample["filename"].lower()
        ]

        print(
            f"Filtering dataset: {len(filtered)}/{len(all_samples)} samples contain 'max-speed' in filename."
        )

        if not filtered:
            print("Warning: No samples matched the 'max-speed' filter.")

        # max-speed対象だけを保持し、インデックスをリセット
        self.dataset["data"] = filtered
        self.current_index = 0
        self.start_pos = None
        self.end_pos = None

    def load_existing_annotations(self):
        """保存先ディレクトリを切り替えて既存アノテーションを読み込み"""
        # 保存先を max-speed 専用フォルダに変更
        self.results_dir = self.RESULTS_DIR
        self.individual_dir = os.path.join(self.results_dir, "individual")
        self.plots_dir = os.path.join(self.results_dir, "plots")
        self.plots_fig_dir = os.path.join(self.results_dir, "plots-fig")

        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.individual_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.plots_fig_dir, exist_ok=True)

        print("Checking for existing max-speed annotations...")
        return super().load_existing_annotations()

    def plot_current_data(self):
        """現在表示中のサンプル名を追記表示し、ピーク周辺をズーム"""
        super().plot_current_data()

        if not self.dataset or not self.dataset.get("data") or self.current_index >= len(self.dataset["data"]):
            return
        if self.fig is None or self.ax is None:
            return

        current_data = self.dataset["data"][self.current_index]
        filename = current_data.get("filename", "Unknown filename")
        signal = current_data.get("aligned_value")

        # コンソールにも表示
        print(f"Viewing file: {filename}")

        # プロット内に表示（右上）
        text = getattr(self, "_filename_text", None)
        text_value = f"File: {filename}"

        if text is None:
            self._filename_text = self.ax.text(
                0.98,
                0.95,
                text_value,
                transform=self.ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.6),
            )
        else:
            self._filename_text.set_text(text_value)

        self.fig.canvas.draw_idle()

    def _sanitize_component(self, value):
        text = str(value) if value is not None else "unknown"
        return re.sub(r'[^0-9A-Za-z._-]+', '_', text)

    def _get_sampling_rate(self, sample):
        sampling_rate = sample.get("sampling_rate_hz")
        if sampling_rate:
            return float(sampling_rate)
        metadata = sample.get("metadata") or {}
        sampling_rate = metadata.get("sampling_rate_hz")
        if sampling_rate:
            return float(sampling_rate)
        return None

    def _get_nyquist(self, sample, sampling_rate):
        nyquist = sample.get("nyquist_hz")
        if nyquist:
            return float(nyquist)
        metadata = sample.get("metadata") or {}
        nyquist = metadata.get("nyquist_hz")
        if nyquist:
            return float(nyquist)
        if sampling_rate:
            return float(sampling_rate) / 2.0
        return None

    def _save_segment_plot(self, sample, start_idx, end_idx):
        signal = sample.get("aligned_value")
        if signal is None or len(signal) == 0:
            print("Warning: No signal data available for plotting.")
            return

        start_idx = max(0, min(start_idx, len(signal) - 1))
        end_idx = max(0, min(end_idx, len(signal) - 1))
        if end_idx <= start_idx:
            print("Warning: Invalid annotation range; skipping plot export.")
            return

        segment = signal[start_idx:end_idx + 1]
        sampling_rate = self._get_sampling_rate(sample)
        # nyquist = self._get_nyquist(sample, sampling_rate)  # commented out per request

        max_pos = sample.get("metadata", {}).get("max_pos")
        if max_pos is None:
            max_pos = sample.get("target_peak_position")

        if sampling_rate and sampling_rate > 0:
            indices = np.arange(start_idx, end_idx + 1)
            time_axis = indices / sampling_rate * 1000.0  # ms
            x_label = "Time [ms]"
            start_coord = time_axis[0]
            end_coord = time_axis[-1]
            if max_pos is not None and start_idx <= max_pos <= end_idx:
                max_coord = max_pos / sampling_rate * 1000.0
            else:
                max_coord = None
        else:
            if len(segment) == 0:
                print("Warning: Empty segment encountered; skipping plot export.")
                return
            time_axis = np.arange(start_idx, end_idx + 1)
            x_label = "Sample Index"
            start_coord = time_axis[0]
            end_coord = time_axis[-1]
            if max_pos is not None and start_idx <= max_pos <= end_idx:
                max_coord = max_pos
            else:
                max_coord = None

        person_label = sample.get("person_display") or sample.get("person") or "unknown_person"
        label_name = sample.get("label_name") or "unknown_label"
        filename = sample.get("filename") or f"sample{self.current_index:04d}"
        session = sample.get("session", "unknown")

        safe_person = self._sanitize_component(person_label)
        safe_label = self._sanitize_component(label_name)
        safe_file = self._sanitize_component(Path(filename).stem)

        plot_filename = f"{safe_person}_session{session}_{safe_label}_{safe_file}_{start_idx}-{end_idx}.png"
        output_path = os.path.join(self.plots_fig_dir, plot_filename)

        # info_lines = []
        # if sampling_rate and sampling_rate > 0:
        #     info_lines.append(f"Sampling frequency: {sampling_rate:.2f} Hz")
        # else:
        #     info_lines.append("Sampling frequency: Unknown")
        # if nyquist and nyquist > 0:
        #     info_lines.append(f"Nyquist frequency: {nyquist:.2f} Hz")
        # else:
        #     info_lines.append("Nyquist frequency: Unknown")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(time_axis, segment, color='tab:blue', linewidth=1.5)
        ax.set_title(f"{person_label} | {label_name} | {filename}")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        margin = (time_axis[-1] - time_axis[0]) * 0.02 if len(time_axis) > 1 else 1.0
        ax.set_xlim(time_axis[0] - margin, time_axis[-1] + margin)

        ax.axvline(start_coord, color='green', linestyle='--', linewidth=1.2, label='Start')
        ax.axvline(end_coord, color='red', linestyle='--', linewidth=1.2, label='End')
        if max_coord is not None:
            ax.axvline(max_coord, color='orange', linestyle=':', linewidth=1.2, label='Max')

        ax.legend(loc='upper right')

        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"Segment plot saved to {output_path}")

    def save_current_annotation(self):
        """アノテーション保存後に該当区間のプロットをエクスポート"""
        if self.start_pos is not None and self.end_pos is not None:
            if self.dataset and self.current_index < len(self.dataset.get("data", [])):
                start_idx = int(self.start_pos)
                end_idx = int(self.end_pos)
                if end_idx < start_idx:
                    start_idx, end_idx = end_idx, start_idx
                current_data = self.dataset["data"][self.current_index]
                self._save_segment_plot(current_data, start_idx, end_idx)

        return super().save_current_annotation()


def main(dataset_path: Optional[str] = None):
    """CLIエントリーポイント"""
    path = dataset_path or "./aligned_dataset/dataset.bin"
    annotator = MaxSpeedWaveformAnnotator(dataset_path=path)

    if not annotator.dataset or len(annotator.dataset.get("data", [])) == 0:
        print("No max-speed samples available. Exiting.")
        return

    annotator.start_annotation()


if __name__ == "__main__":
    main()
