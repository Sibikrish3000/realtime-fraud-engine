#!/usr/bin/env -S uv run --script
#
# /// script
# dependencies = [
#   "pandas",
#   "rich",
# ]
# ///
import os
import zipfile

path = os.path.join(os.path.dirname(__file__), "dataset")
def download_data():
    os.system(f"curl -L -o {path}/fraud-detection.zip https://www.kaggle.com/api/v1/datasets/download/kartik2112/fraud-detection")

    with zipfile.ZipFile(f"{path}/fraud-detection.zip", 'r') as zip_ref:
        zip_ref.extractall(f"{path}/")
def combine_data():
    import pandas as pd
    train = pd.read_csv(f"{path}/fraud-detection/fraudTrain.csv")
    test = pd.read_csv(f"{path}/fraud-detection/fraudTest.csv")
    df = pd.concat([train, test], axis=0).reset_index(drop=True)
    df.to_csv(f"{path}/fraud.csv", index=False)
if __name__ == "__main__":
    download_data()
    combine_data()
    