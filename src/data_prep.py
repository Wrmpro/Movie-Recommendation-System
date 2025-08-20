import os
import io
import zipfile
import shutil
import requests
import certifi
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ML_SMALL_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
MOVIES_CSV = "movies.csv"
RATINGS_CSV = "ratings.csv"


def ensure_movielens_data(data_dir: str = "data") -> None:
    """
    Ensure movies.csv and ratings.csv exist under data_dir.
    If missing, attempt to download ml-latest-small.zip and extract them.
    On SSL failures, raises a RuntimeError with actionable instructions.
    """
    os.makedirs(data_dir, exist_ok=True)
    movies_path = os.path.join(data_dir, MOVIES_CSV)
    ratings_path = os.path.join(data_dir, RATINGS_CSV)

    # If files already present, nothing to do
    if os.path.exists(movies_path) and os.path.exists(ratings_path):
        return

    # Attempt a robust download using certifi CA bundle and retries
    try:
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("https://", adapter)

        resp = session.get(ML_SMALL_URL, timeout=30, verify=certifi.where())
        resp.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            # extract only the two files we need
            for name in (MOVIES_CSV, RATINGS_CSV):
                member = f"ml-latest-small/{name}"
                if member in z.namelist():
                    z.extract(member, path=data_dir)
                    src = os.path.join(data_dir, member)
                    dst = os.path.join(data_dir, name)
                    shutil.move(src, dst)
            # cleanup extracted directory if present
            extracted_dir = os.path.join(data_dir, "ml-latest-small")
            if os.path.isdir(extracted_dir):
                shutil.rmtree(extracted_dir)

    except requests.exceptions.SSLError as e:
        # Raise a helpful error with clear next steps
        raise RuntimeError(
            "SSL error while downloading MovieLens dataset. This frequently happens on hosted "
            "runners when HTTPS certificate verification fails or outbound TLS is blocked.\n\n"
            "Recommended actions:\n"
            "1) Add the small MovieLens CSVs to your repo under the 'data/' folder (movies.csv and ratings.csv) "
            "so the app does not need to download anything at runtime. This is the most robust solution for Streamlit Cloud.\n"
            "   - Locally: download ml-latest-small.zip from https://files.grouplens.org/datasets/movielens/ and copy "
            "movies.csv and ratings.csv into data/\n"
            "   - Then git add/commit/push data/movies.csv data/ratings.csv and redeploy.\n\n"
            "2) (Temporary debug) If you want to confirm this is indeed an SSL verification issue, try setting "
            "verify=False in the request (not recommended long-term).\n\n"
            "Full error from requests was: " + str(e)
        ) from e

    except Exception as e:
        # Re-raise with context so logs show the failure reason
        raise RuntimeError("Failed to download/extract MovieLens data: " + str(e)) from e


def load_movielens(data_dir: str = "data"):
    movies_path = os.path.join(data_dir, MOVIES_CSV)
    ratings_path = os.path.join(data_dir, RATINGS_CSV)

    if not os.path.exists(movies_path) or not os.path.exists(ratings_path):
        raise FileNotFoundError(
            "MovieLens CSV files not found in data/. Ensure data/ contains movies.csv and ratings.csv, "
            "or that the automatic download succeeded."
        )

    movies_df = pd.read_csv(movies_path)
    ratings_df = pd.read_csv(ratings_path)
    # Normalize genres field to consistent list form
    movies_df["genres"] = movies_df["genres"].fillna("(no genres listed)")
    return movies_df, ratings_df