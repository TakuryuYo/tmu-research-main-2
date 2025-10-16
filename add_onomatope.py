from PIL import Image, ImageDraw, ImageFont
import os

# 番号に対応する日本語テキストの辞書
text_map = {
    0: "ちょこん", 1: "ぴょこん", 2: "ぐいっ", 3: "ぐいぐい", 4: "きゅっ", 5: "ぎゅっ", 6: "ぎゅーぎゅー",
    7: "ぬっ", 8: "すっ", 9: "むぎゅっ", 10: "ふにゃ", 11: "へな", 12: "ささっ", 13: "ちゃっ", 14: "ぱぱっ", 15: "さっ", 
    16: "ちょこちょこ", 17: "ちょこっ", 18: "ぐん", 19: "ぐんぐん", 20: "ずんずん", 21: "じりじり", 22: "じわじわ", 
    23: "ぴん", 24: "しっかり", 25: "力いっぱい"
}

# Windows標準の日本語フォント
font_path = "C:\\Windows\\Fonts\\msgothic.ttc"
font_size = 48  # 必要に応じて調整

def add_text_to_image(image_path, text, output_path):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(font_path, font_size)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except AttributeError:
        font = ImageFont.truetype(font_path, font_size)
        text_width, text_height = draw.textsize(text, font=font)

    # 上部中央に配置
    padding_top = 10
    x = (image.width - text_width) // 2
    y = padding_top

    # 黒文字のみ
    draw.text((x, y), text, font=font, fill="black")

    image.save(output_path)

# カレントディレクトリ内のすべての画像を処理
for i in range(26):
    original_filename = f"{i}.png"
    new_filename = f"{i}_{text_map[i]}.png"
    if os.path.exists(original_filename):
        add_text_to_image(original_filename, text_map[i], new_filename)
        print(f"{original_filename} を処理して {new_filename} として保存しました。")
    else:
        print(f"ファイル {original_filename} が存在しません。")

print("処理が完了しました。")
