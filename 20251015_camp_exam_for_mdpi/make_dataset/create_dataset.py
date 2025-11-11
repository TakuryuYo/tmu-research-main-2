import numpy as np
import os
import sys
import pickle
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from reposition import reposition
from load import parse_filename


def _load_font(size):
    """Try to load a system font that can render Japanese text; fall back to default."""
    font_candidates = [
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/System/Library/Fonts/Supplemental/AppleSDGothicNeo.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
    ]
    for path in font_candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _text_height(font, text="Ag"):
    """Return text height for the given font."""
    try:
        bbox = font.getbbox(text)
        return bbox[3] - bbox[1]
    except AttributeError:
        return font.getsize(text)[1]


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DEFAULT_RAW_DATA_DIR = PROJECT_DIR / "raw_data"
DEFAULT_ALIGNED_DIR = PROJECT_DIR / "aligned_dataset"
DEFAULT_PLOT_DIR = PROJECT_DIR / "input_datas"


def create_aligned_dataset(repositioned_data, target_length=None, output_dir='aligned_dataset'):
    """
    位置合わせ済みデータから1次元の振幅データセットを作成
    
    Args:
        repositioned_data: 位置合わせ済みデータ
        target_length: 出力データの長さ（指定しない場合は最長データに合わせる）
        output_dir: 出力ディレクトリ
        
    Returns:
        dict: 作成されたデータセット情報
    """
    print(f"Creating aligned dataset in {output_dir}/...")
    
    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    
    # target_lengthが指定されていない場合、最長データを調べる
    if target_length is None:
        max_length = 0
        for person in repositioned_data:
            for entry in person["data"]:
                max_length = max(max_length, len(entry["value"]))
        target_length = max_length
        print(f"Target length set to maximum data length: {target_length}")
    
    dataset_info = {
        "target_length": target_length,
        "total_samples": 0,
        "samples_by_person": {},
        "samples_by_label": {},
        "data": []
    }
    
    total_samples = sum(len(person["data"]) for person in repositioned_data)
    processed_count = 0
    
    for person in repositioned_data:
        person_id = person["person"]
        print(f"Processing {person_id}...")
        
        person_samples = []
        
        for entry in person["data"]:
            # 元のデータを取得
            original_value = entry["value"]
            label_name = entry.get("label_name")
            label_slug = entry.get("label_slug")
            filename = entry.get("filename", "")
            metadata = entry.get("metadata") or {}
            sampling_rate = metadata.get("sampling_rate_hz")
            nyquist_hz = metadata.get("nyquist_hz")

            if not label_slug:
                _, _, _, inferred_slug = parse_filename(filename)
                label_slug = inferred_slug

            if not label_name or str(label_name).startswith("Unknown"):
                if label_slug:
                    label_name = label_slug
                elif filename:
                    label_name = Path(filename).stem
                else:
                    label_name = "Unknown"
            
            # エントリにも反映
            entry["label_name"] = label_name
            entry["label_slug"] = label_slug
            entry["metadata"] = metadata
            
            # 最大値の位置を検出
            peak_position = np.argmax(np.abs(original_value))
            
            # target_lengthの中央に最大値が来るようにシフト量を計算
            target_center = target_length // 2
            shift = target_center - peak_position
            
            # 新しい配列を作成
            aligned_value = np.zeros(target_length)
            
            # データの配置範囲を計算
            start_src = max(0, -shift)
            end_src = min(len(original_value), target_length - shift)
            start_dst = max(0, shift)
            end_dst = min(target_length, shift + len(original_value))
            
            # データをコピー
            if start_src < end_src and start_dst < end_dst:
                copy_length = min(end_src - start_src, end_dst - start_dst)
                aligned_value[start_dst:start_dst + copy_length] = original_value[start_src:start_src + copy_length]
            
            # エッジパディング
            if start_dst > 0:
                # 前方のパディング（最初の値で埋める）
                first_value = original_value[start_src] if start_src < len(original_value) else original_value[0]
                aligned_value[:start_dst] = first_value
            
            if end_dst < target_length:
                # 後方のパディング（最後の値で埋める）
                last_value = original_value[end_src-1] if end_src > 0 else original_value[-1]
                aligned_value[end_dst:] = last_value
            
            # データ情報を作成
            sample_info = {
                "person": person_id,
                "session": entry["session"],
                "label_id": entry["label_id"],
                "label_name": label_name,
                "label_slug": label_slug,
                "filename": filename,
                "sampling_rate_hz": sampling_rate,
                "nyquist_hz": nyquist_hz,
                "metadata": metadata,
                "original_length": len(original_value),
                "original_peak_position": peak_position,
                "target_peak_position": target_center,
                "shift": shift,
                "aligned_value": aligned_value
            }
            
            person_samples.append(sample_info)
            dataset_info["data"].append(sample_info)
            
            # 統計情報を更新
            label_name = entry["label_name"]
            if label_name not in dataset_info["samples_by_label"]:
                dataset_info["samples_by_label"][label_name] = 0
            dataset_info["samples_by_label"][label_name] += 1
            
            processed_count += 1
            if processed_count % 50 == 0:
                print(f"  Progress: {processed_count}/{total_samples} samples processed ({100*processed_count/total_samples:.1f}%)")
        
        dataset_info["samples_by_person"][person_id] = len(person_samples)
    
    dataset_info["total_samples"] = len(dataset_info["data"])
    
    # 統計情報を表示
    print(f"\nDataset creation completed!")
    print(f"Total samples: {dataset_info['total_samples']}")
    print(f"Target length: {dataset_info['target_length']}")
    print(f"Samples by person:")
    for person, count in dataset_info["samples_by_person"].items():
        print(f"  {person}: {count}")
    print(f"Samples by label:")
    for label, count in sorted(dataset_info["samples_by_label"].items()):
        print(f"  {label}: {count}")
    
    return dataset_info


def save_dataset_as_pickle(dataset_info, output_dir='aligned_dataset'):
    """
    データセットをpickleファイルとして保存
    
    Args:
        dataset_info: create_aligned_datasetの戻り値
        output_dir: 出力ディレクトリ
    """
    print(f"Saving dataset as pickle to {output_dir}/dataset.bin...")
    
    # データと ラベルを配列に変換
    X = np.array([sample["aligned_value"] for sample in dataset_info["data"]])
    y_label_id = np.array([sample["label_id"] for sample in dataset_info["data"]])
    y_label_name = np.array([sample["label_name"] for sample in dataset_info["data"]])
    person_ids = np.array([sample["person"] for sample in dataset_info["data"]])
    sessions = np.array([sample["session"] for sample in dataset_info["data"]])
    
    # データセット辞書を作成
    dataset = {
        "X": X,
        "y_label_id": y_label_id,
        "y_label_name": y_label_name,
        "person_ids": person_ids,
        "sessions": sessions,
        "dataset_info": dataset_info
    }
    
    # pickleファイルとして保存
    with open(os.path.join(output_dir, "dataset.bin"), "wb") as f:
        pickle.dump(dataset, f)
    
    print(f"Dataset saved as dataset.bin!")
    print(f"  X: {X.shape}")
    print(f"  y_label_id: {y_label_id.shape}")
    print(f"  y_label_name: {y_label_name.shape}")
    print(f"  person_ids: {person_ids.shape}")
    print(f"  sessions: {sessions.shape}")


def plot_all_data(dataset_info, plot_dir='input_datas'):
    """
    Pillowベースで全データのプロットを作成して保存
    
    Args:
        dataset_info: create_aligned_datasetの戻り値
        plot_dir: プロット保存ディレクトリ
    """
    print(f"Creating plots for all data in {plot_dir}/...")
    
    os.makedirs(plot_dir, exist_ok=True)
    
    total_samples = len(dataset_info["data"])
    plot_count = 0
    
    img_width, img_height = 1200, 600
    plot_width = int(img_width * 0.65)
    text_area_left = plot_width + 30
    title_font = _load_font(22)
    body_font = _load_font(16)
    mono_font = _load_font(14)
    title_height = _text_height(title_font)
    body_line_height = _text_height(body_font) + 6
    
    for sample in dataset_info["data"]:
        values = np.asarray(sample["aligned_value"], dtype=np.float64)
        if values.size == 0:
            continue
        
        img = Image.new("RGB", (img_width, img_height), "white")
        draw = ImageDraw.Draw(img)
        
        plot_left = 40
        plot_right = plot_width - 40
        plot_top = 20 + title_height + 10
        plot_bottom = img_height - 40
        plot_height = plot_bottom - plot_top
        plot_mid = plot_top + plot_height / 2
        
        # タイトル
        title = f'Aligned Data: {sample["person"]} - {sample["label_name"]}'
        draw.text((plot_left, 20), title, fill=(0, 0, 0), font=title_font)
        
        # グリッド描画
        grid_color = (220, 220, 220)
        for i in range(5):
            y = plot_top + plot_height * i / 4
            draw.line([(plot_left, y), (plot_right, y)], fill=grid_color, width=1)
        for i in range(6):
            x = plot_left + (plot_right - plot_left) * i / 5
            draw.line([(x, plot_top), (x, plot_bottom)], fill=grid_color, width=1)
        draw.rectangle([plot_left, plot_top, plot_right, plot_bottom], outline=(120, 120, 120), width=1)
        
        max_abs = float(np.max(np.abs(values)))
        if max_abs == 0:
            max_abs = 1.0
        scale = (plot_bottom - plot_top) / (2 * max_abs)
        zero_y = plot_mid
        draw.line([(plot_left, zero_y), (plot_right, zero_y)], fill=(180, 180, 180), width=1)
        
        value_len = len(values)
        if value_len > 1:
            denom = value_len - 1
        else:
            denom = 1
        
        max_points = 2000
        if value_len > max_points:
            sample_idx = np.linspace(0, value_len - 1, max_points, dtype=int)
        else:
            sample_idx = np.arange(value_len)
        
        waveform_points = []
        for idx in sample_idx:
            ratio = idx / denom
            x = plot_left + (plot_right - plot_left) * ratio
            y = plot_mid - values[idx] * scale
            y = max(plot_top, min(plot_bottom, y))
            waveform_points.append((int(round(x)), int(round(y))))
        
        if len(waveform_points) > 1:
            draw.line(waveform_points, fill=(46, 117, 182), width=2)
        else:
            draw.point(waveform_points[0], fill=(46, 117, 182))
        
        # ピーク位置ライン
        peak_index = int(sample["target_peak_position"])
        peak_index = max(0, min(peak_index, value_len - 1))
        peak_ratio = peak_index / denom
        peak_x = plot_left + (plot_right - plot_left) * peak_ratio
        draw.line([(peak_x, plot_top), (peak_x, plot_bottom)], fill=(200, 60, 60), width=2)
        peak_label = f"Peak {peak_index}"
        label_y = plot_top + 5
        draw.text((min(peak_x + 6, plot_right - 80), label_y), peak_label, fill=(200, 60, 60), font=mono_font)
        
        # メタ情報
        sampling_rate = sample.get("sampling_rate_hz")
        nyquist_hz = sample.get("nyquist_hz")
        info_lines = [
            ("Person", sample["person"]),
            ("Label", f"{sample['label_name']} (ID: {sample['label_id']:02d})"),
            ("Session", sample["session"]),
            ("Original Length", sample["original_length"]),
            ("Target Length", len(values)),
            ("Original Peak", sample["original_peak_position"]),
            ("Aligned Peak", sample["target_peak_position"]),
            ("Shift", sample["shift"]),
            ("Sampling Rate", f"{sampling_rate:.2f} Hz" if sampling_rate is not None else "-"),
            ("Nyquist", f"{nyquist_hz:.2f} Hz" if nyquist_hz is not None else "-"),
            ("Filename", sample["filename"]),
        ]
        
        text_y = 30
        draw.text((text_area_left, text_y), "Data Information", fill=(0, 0, 0), font=title_font)
        text_y += title_height + 10
        
        for key, value in info_lines:
            text = f"{key:>15}: {value}"
            draw.text((text_area_left, text_y), text, fill=(20, 20, 20), font=body_font)
            text_y += body_line_height
        
        # ファイル名生成
        label_name_safe = (
            sample['label_name']
            .replace('（', '_').replace('）', '_').replace('/', '_')
            .replace('ー', '_').replace(' ', '_')
        )
        filename = f"{sample['person']}_{label_name_safe}_session{sample['session']}_label{sample['label_id']:02d}.png"
        filepath = os.path.join(plot_dir, filename)
        
        img.save(filepath, format="PNG", optimize=True)
        
        plot_count += 1
        if plot_count % 50 == 0:
            print(f"  Progress: {plot_count}/{total_samples} plots saved ({100 * plot_count / total_samples:.1f}%)")
    
    print(f"Successfully saved {plot_count} plots to {plot_dir}/")


# メイン処理
if __name__ == "__main__":
    print("=== Creating Aligned 1D Amplitude Dataset ===")
    
    # 位置合わせ済みデータを読み込み
    print("Loading repositioned data...")
    repositioned_data = reposition('./raw_data/')
    
    # 位置合わせデータセットを作成
    dataset_info = create_aligned_dataset(repositioned_data, output_dir='aligned_dataset')
    
    # pickleファイルとして保存
    save_dataset_as_pickle(dataset_info, output_dir='aligned_dataset')
    
    # 全データのプロットを作成
    plot_all_data(dataset_info, plot_dir='input_datas')
    
    print("Dataset creation completed!")
