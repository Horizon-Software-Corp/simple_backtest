import argparse
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_histograms(data, output_folder):
    columns = data.columns

    for i in range(0, len(columns), 2):
        column1 = columns[i]
        column2 = columns[i + 1] if i + 1 < len(columns) else None

        if (
            column2 is not None
            and column1.split("_")[-1] == column2.split("_")[-1]
            and "long_short" in column1
            and "long_short" in column2
        ):
            # 同じ銘柄に対する2つの指標の場合、重ねて表示
            finite_data1 = data[column1][np.isfinite(data[column1])]
            finite_data2 = data[column2][np.isfinite(data[column2])]

            plt.figure(figsize=(8, 6))
            plt.hist(
                finite_data1,
                bins=30,
                edgecolor="black",
                alpha=0.4,
                label=column1,
                color="red",
            )
            plt.hist(
                finite_data2,
                bins=30,
                edgecolor="black",
                alpha=0.4,
                label=column2,
                color="blue",
            )
            plt.title(f'Histogram of {column1.split("_", maxsplit=1)[1]}')
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.legend()
            plt.tight_layout()

            output_path = os.path.join(
                output_folder, f'histogram_{column1.split("_", maxsplit=1)[1]}.png'
            )
            plt.savefig(output_path)
            plt.close()
        else:
            # 単独の指標の場合、個別にヒストグラムを描画
            finite_data = data[column1][np.isfinite(data[column1])]

            plt.figure(figsize=(8, 6))
            plt.hist(finite_data, bins=30, edgecolor="black")
            plt.title(f"Histogram of {column1}")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.tight_layout()

            output_path = os.path.join(output_folder, f"histogram_{column1}.png")
            plt.savefig(output_path)
            plt.close()


def main(file_path):
    # pklファイルからデータを読み込む
    with open(file_path, "rb") as file:
        data = pickle.load(file)

    # 出力フォルダを作成
    output_folder = "histograms"
    os.makedirs(output_folder, exist_ok=True)

    # ヒストグラムを描画
    plot_histograms(data, output_folder)

    print("Histograms have been saved in the 'histograms' folder.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot histograms from a pkl file.")
    parser.add_argument("file_path", type=str, help="Path to the pkl file")
    args = parser.parse_args()

    main(args.file_path)
