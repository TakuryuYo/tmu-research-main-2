#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Max-speedアノテーションツール (manual override 版)

- 通常は `annotation_tool_maxspeed.py` と同じ挙動
- 'm' キーで `max_pos` / `power_raw` を手動入力して上書き可能
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
ANNOTATION_MAX_DIR = THIS_DIR.parent
REPO_ROOT = ANNOTATION_MAX_DIR.parent
ANNOTATION_DIR = REPO_ROOT / "annotation"

for target in (ANNOTATION_MAX_DIR, ANNOTATION_DIR):
    path_str = str(target)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from annotation_tool_maxspeed import MaxSpeedWaveformAnnotator


class ManualOverrideMaxSpeedWaveformAnnotator(MaxSpeedWaveformAnnotator):
    """'m' キーで max_pos / power_raw を手動更新できるアノテーター"""

    def __init__(self, dataset_path: Optional[str] = None):
        self.manual_overrides: Dict[int, Dict[str, float]] = {}
        self._manual_hint_text = None
        self._manual_follow_enabled = False
        self._manual_follow_sample: Optional[int] = None
        path = dataset_path or "./aligned_dataset/dataset.bin"
        super().__init__(dataset_path=path)

    # ------------------------------------------------------------------
    # Dataset / annotation loading
    # ------------------------------------------------------------------
    def load_existing_annotations(self):
        result = super().load_existing_annotations()
        self._refresh_manual_overrides()
        return result

    def _refresh_manual_overrides(self):
        """既存アノテーションから現在の max_pos / power_raw を初期化"""
        self.manual_overrides = {}
        for annotation in getattr(self, "annotations", []):
            sample_index = annotation.get("sample_index")
            if sample_index is None:
                continue
            self.manual_overrides[int(sample_index)] = {
                "max_pos": annotation.get("max_pos"),
                "power_raw": annotation.get("power_raw"),
            }

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------
    def plot_current_data(self):
        if getattr(self, "_manual_follow_sample", None) != self.current_index:
            self._manual_follow_enabled = False
            self._manual_follow_sample = self.current_index

        super().plot_current_data()
        self._apply_manual_max_line()
        self._update_manual_hint_text()

    def _apply_manual_max_line(self):
        if not self.dataset or self.fig is None or self.ax is None:
            return
        overrides = self.manual_overrides.get(self.current_index)
        if not overrides or overrides.get("max_pos") is None:
            return

        override_pos = int(overrides["max_pos"])

        # 既存の Max ラインを探して上書き
        target_line = None
        for line in getattr(self.ax, "lines", []):
            label = line.get_label()
            if isinstance(label, str) and label.startswith("Max"):
                target_line = line
                break

        if target_line is not None:
            target_line.set_xdata([override_pos, override_pos])
            target_line.set_label(f"Max (manual): {override_pos}")
            self.ax.legend(loc="upper right")
            self.fig.canvas.draw_idle()

    def _update_manual_hint_text(self):
        if self.fig is None or self.ax is None:
            return

        overrides = self.manual_overrides.get(self.current_index)
        if overrides:
            hint = "Press 'm': set max_pos to cursor (manual override active)"
        else:
            hint = "Press 'm' to set max_pos to cursor"

        if self._manual_hint_text is None:
            self._manual_hint_text = self.ax.text(
                0.02,
                0.82,
                hint,
                transform=self.ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.6),
            )
        else:
            self._manual_hint_text.set_text(hint)

    # ------------------------------------------------------------------
    # Feature calculation overrides
    # ------------------------------------------------------------------
    def calculate_features(self, signal, start_pos, end_pos, sample_index: Optional[int] = None):
        features = super().calculate_features(signal, start_pos, end_pos)
        if not features:
            return features

        index = sample_index if sample_index is not None else self.current_index
        overrides = self.manual_overrides.get(index)
        if not overrides:
            return features

        if overrides.get("max_pos") is not None:
            max_pos = int(overrides["max_pos"])
            max_pos = max(0, min(len(signal) - 1, max_pos))
            features["max_pos"] = max_pos
            features["speed_raw"] = abs(max_pos - start_pos)

        if overrides.get("power_raw") is not None:
            features["power_raw"] = float(overrides["power_raw"])

        return features

    # ------------------------------------------------------------------
    # Key handling / manual edit
    # ------------------------------------------------------------------
    def on_key_press(self, event):
        if event.key == "m":
            self.edit_manual_features()
            return

        super().on_key_press(event)

        if event.key == "r":
            removed = self.manual_overrides.pop(self.current_index, None)
            if removed:
                print("Manual overrides cleared for current sample.")
                self.plot_current_data()
            self._manual_follow_enabled = False
            self._manual_follow_sample = None

    def on_mouse_move(self, event):
        super().on_mouse_move(event)
        if (
            self._manual_follow_enabled
            and self._manual_follow_sample == self.current_index
            and event.inaxes == self.ax
            and event.xdata is not None
        ):
            self._update_manual_override_from_cursor(cursor_x=event.xdata, announce=False)

    def edit_manual_features(self):
        """'m' キー押下時に呼び出される処理（カーソル位置で max_pos を更新）"""
        if not self.dataset or not self.dataset.get("data"):
            print("Dataset not available for manual edit.")
            return
        if self.current_index >= len(self.dataset["data"]):
            print("Invalid index for manual edit.")
            return

        current_data = self.dataset["data"][self.current_index]
        signal = current_data.get("aligned_value")
        if signal is None or len(signal) == 0:
            print("No signal data available for manual edit.")
            return

        default_max_pos = int(np.argmax(np.abs(signal)))
        default_power_raw = float(np.max(np.abs(signal)))
        overrides = self.manual_overrides.get(self.current_index, {})

        current_max_pos = overrides.get("max_pos", default_max_pos)
        current_power = overrides.get("power_raw", default_power_raw)

        if self._manual_follow_enabled:
            self._manual_follow_enabled = False
            self._manual_follow_sample = None
            print("Manual follow disabled. Max line locked at current position.")
            return

        if not self._update_manual_override_from_cursor(announce=True):
            print("Move the cursor over the waveform before pressing 'm'.")
            return

        self._manual_follow_enabled = True
        self._manual_follow_sample = self.current_index
        print("Manual follow enabled. Move the cursor to adjust the max line; press 'm' again to lock.")

    def _update_manual_override_from_cursor(self, cursor_x=None, announce=False):
        if not self.dataset or self.current_index >= len(self.dataset.get("data", [])):
            return False

        current_data = self.dataset["data"][self.current_index]
        signal = current_data.get("aligned_value")
        if signal is None or len(signal) == 0:
            return False

        if cursor_x is None:
            if self.mouse_x is None:
                return False
            cursor_x = self.mouse_x

        candidate = int(round(cursor_x))
        candidate = max(0, min(len(signal) - 1, candidate))

        amplitude = float(np.abs(signal[candidate]))
        overrides = dict(self.manual_overrides.get(self.current_index, {}))
        overrides["max_pos"] = candidate
        overrides["power_raw"] = amplitude
        self.manual_overrides[self.current_index] = overrides

        if announce:
            print(
                f"Manual max_pos set to cursor ({candidate}); power_raw updated to {amplitude:.4f}."
            )

        self._apply_manual_max_line()
        if self.fig:
            self.fig.canvas.draw_idle()
        return True


def main(dataset_path: Optional[str] = None):
    """CLI entry point"""
    path = dataset_path or "./aligned_dataset/dataset.bin"
    annotator = ManualOverrideMaxSpeedWaveformAnnotator(dataset_path=path)

    if not annotator.dataset or len(annotator.dataset.get("data", [])) == 0:
        print("No max-speed samples available. Exiting.")
        return

    annotator.start_annotation()


if __name__ == "__main__":
    main()
