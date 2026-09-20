"""
Downloads the Credit Card Fraud Detection dataset (~98 MB) from a public GitHub mirror.
Run this once before running the pipeline or notebook: python3 download_data.py

The raw CSV is NOT committed to this repository (see .gitignore) because it sits right at
GitHub's 100 MB hard file-size limit, which makes direct commits slow and fragile. Downloading
it on demand keeps the repository itself lightweight and fast to clone.
"""
import os
import urllib.request

DATA_URL = ("https://raw.githubusercontent.com/nsethi31/"
            "Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv")
DATA_PATH = os.path.join(os.path.dirname(__file__), "creditcard.csv")

if os.path.exists(DATA_PATH):
    print(f"Already downloaded: {DATA_PATH}")
else:
    print(f"Downloading dataset (~98 MB) to {DATA_PATH} ...")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    print("Done.")
