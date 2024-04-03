import argparse
import pickle
import os
import matplotlib.pyplot as plt
import pandas as pd
import json


def visualize_data(file_path):
    # pklファイルを読み込む
    with open(file_path, "rb") as file:
        data = pickle.load(file)

    # 銘柄のリストを取得
    symbols = [col.split("_")[-1] for col in data.columns if "_" in col]
    symbols = sorted(set(symbols))

    # 出力フォルダ名を取得
    folder_name = os.path.splitext(os.path.basename(file_path))[0]
    output_folder = os.path.join("visualized_data", folder_name)
    os.makedirs(output_folder, exist_ok=True)

    # データの要約を作成
    summary = {
        "銘柄数": len(symbols),
        "平均取引高": data[[col for col in data.columns if "volume" in col]]
        .mean()
        .mean(),
        "平均価格": data[[col for col in data.columns if "close" in col]].mean().mean(),
    }

    # 要約をテキストファイルに保存
    with open(os.path.join(output_folder, "summary.txt"), "w") as file:
        for key, value in summary.items():
            file.write(f"{key}: {value}\n")

    # 列名をテキストファイルに保存
    with open(os.path.join(output_folder, "column_names.txt"), "w") as file:
        file.write("\n".join(data.columns))

    # 各銘柄の終値の推移をグラフ化
    for symbol in symbols:
        close_col = f"close_{symbol}"
        if close_col in data.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(data.index, data[close_col])
            plt.title(f"{symbol} - Close Price")
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f"{symbol}_price_trend.png"))
            plt.close()

    # データの統計情報を計算して保存
    stats = {}
    for symbol in symbols:
        symbol_stats = {}
        for col in ["open", "high", "low", "close", "volume"]:
            col_name = f"{col}_{symbol}"
            if col_name in data.columns:
                symbol_stats[col] = {
                    "min": data[col_name].min(),
                    "max": data[col_name].max(),
                    "mean": data[col_name].mean(),
                    "std": data[col_name].std(),
                }
        stats[symbol] = symbol_stats

    with open(os.path.join(output_folder, "stats.json"), "w") as file:
        json.dump(stats, file, indent=4)

    print(f"{folder_name}の可視化が完了しました。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize data from a pickle file.")
    parser.add_argument("file_path", type=str, help="Path to the pickle file.")
    args = parser.parse_args()

    visualize_data(args.file_path)
