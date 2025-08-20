import os
import io
import zipfile
import requests
import pandas as pd

ML_SMALL_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
ML_SMALL_DIRNAME = "ml-latest-small"

def ensure_movielens_data(data_dir: str = "data") -> None:
    os.makedirs(data_dir, exist_ok=True)
    target_path = os.path.join(data_dir, ML_SMALL_DIRNAME)
    movies_csv = os.path.join(target_path, "movies.csv")
    ratings_csv = os.path.join(target_path, "ratings.csv")

    if os.path.exists(movies_csv) and os.path.exists(ratings_csv):
        return

    zip_path = os.path.join(data_dir, "ml-latest-small.zip")
    if not os.path.exists(zip_path):
        resp = requests.get(ML_SMALL_URL, timeout=60)
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            f.write(resp.content)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(path=data_dir)

def load_movielens(data_dir: str = "data"):
    base = os.path.join(data_dir, ML_SMALL_DIRNAME)
    movies = pd.read_csv(os.path.join(base, "movies.csv"))
    ratings = pd.read_csv(os.path.join(base, "ratings.csv"))
    # Normalize genres field to consistent list form
    movies["genres"] = movies["genres"].fillna("(no genres listed)")
    return movies, ratings