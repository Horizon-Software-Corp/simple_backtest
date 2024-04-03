import argparse
import pickle
import pandas as pd
import preprocess as pp
from sklearn.model_selection import train_test_split

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_and_preprocess_data(file_path):
    # pklファイルからデータを読み込む
    with open(file_path, "rb") as file:
        data = pickle.load(file)

    data = pp.add_feature(data)
    data = pp.preprocess(data)
    data = pp.lagging(data)
    data, target = pp.set_target(data)

    return data, target


def split_data(data, target):
    # データを特徴量Xとターゲットyに分割する
    X = data[data.columns - target]  # 特徴量を選択
    y = data[target]  # ターゲット変数を選択

    # データを学習用とテスト用に分割する
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


def train_and_evaluate(
    model, X_train, X_test, y_train, y_test, epochs=3, batch_size=16, learning_rate=2e-5
):
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # オプティマイザの設定
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # データローダの作成
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train.values), torch.tensor(y_train.values)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    # 学習ループ
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        # 評価
        model.eval()
        with torch.no_grad():
            input_ids = torch.tensor(X_test.values).to(device)
            labels = torch.tensor(y_test.values).to(device)
            outputs = model(input_ids)
            predictions = torch.argmax(outputs.logits, dim=1)

            accuracy = accuracy_score(labels.cpu(), predictions.cpu())
            precision = precision_score(
                labels.cpu(), predictions.cpu(), average="weighted"
            )
            recall = recall_score(labels.cpu(), predictions.cpu(), average="weighted")
            f1 = f1_score(labels.cpu(), predictions.cpu(), average="weighted")

            print(f"Epoch {epoch + 1}:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1:.4f}")

    return model


def main(file_path):
    # データの読み込みと前処理
    data, target = load_and_preprocess_data(file_path)
    print(target)

    # データの分割
    X_train, X_test, y_train, y_test = split_data(data)

    # モデルの学習と評価
    model = train_and_evaluate(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a model.")
    parser.add_argument("file_path", type=str, help="Path to the .pkl file")
    args = parser.parse_args()

    main(args.file_path)
