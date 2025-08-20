import os
import io
import zipfile
import requests
import pandas as pd
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3

# Allow overriding the ZIP URL via environment variable (e.g., in deployment settings)
ML_SMALL_URL = os.getenv(
    "MOVIELENS_ZIP_URL",
    "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
)
ML_SMALL_DIRNAME = "ml-latest-small"

def _download_with_retry(url: str, timeout: int = 60) -> bytes:
    """Download file with retry logic and SSL error handling."""
    # Configure retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)

    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    headers = {
        # Some hosts are picky about missing UA; provide a standard UA
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }

    # Respect an env override to disable SSL verification (as last resort)
    verify_pref = os.getenv("MOVIELENS_VERIFY_SSL", "true").lower() not in ("0", "false", "no")
    timeout_pref = int(os.getenv("MOVIELENS_TIMEOUT", str(timeout)))

    def _try_request(verify_flag: bool) -> bytes:
        resp = session.get(url, timeout=timeout_pref, verify=verify_flag, headers=headers)
        resp.raise_for_status()
        return resp.content

    # Try with SSL verification first (unless explicitly disabled)
    try:
        content = _try_request(verify_pref)
        return content
    except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
        print(f"Network/SSL error encountered: {e}")
        print("Retrying with SSL verification disabled...")
        try:
            # Suppress SSL warnings when bypassing verification
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            content = _try_request(False)
            return content
        except Exception as fallback_error:
            raise RuntimeError(
                f"Failed to download data from {url}. "
                f"Original error: {e}. "
                f"Fallback error: {fallback_error}. "
                f"Please check your internet connection or download the data manually."
            ) from fallback_error
    except Exception as e:
        raise RuntimeError(
            f"Failed to download data from {url}. Error: {e}"
        ) from e

def ensure_movielens_data(data_dir: str = "data") -> None:
    os.makedirs(data_dir, exist_ok=True)
    target_path = os.path.join(data_dir, ML_SMALL_DIRNAME)
    movies_csv = os.path.join(target_path, "movies.csv")
    ratings_csv = os.path.join(target_path, "ratings.csv")

    if os.path.exists(movies_csv) and os.path.exists(ratings_csv):
        return

    zip_path = os.path.join(data_dir, "ml-latest-small.zip")
    if not os.path.exists(zip_path):
        try:
            print(f"Downloading MovieLens data from {ML_SMALL_URL}...")
            content = _download_with_retry(ML_SMALL_URL, timeout=60)
            with open(zip_path, "wb") as f:
                f.write(content)
            print("Download completed successfully.")
        except Exception as e:
            raise RuntimeError(
                f"Could not download MovieLens data. Error: {e}\n\n"
                f"To fix this issue:\n"
                f"1. Check your internet connection\n"
                f"2. Or manually download the file from:\n"
                f"   {ML_SMALL_URL}\n"
                f"3. Save it as: {zip_path}\n"
                f"4. Then restart the application"
            ) from e

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(path=data_dir)
        print(f"Data extracted to {target_path}")
    except zipfile.BadZipFile as e:
        # Remove corrupted zip file so it can be re-downloaded
        if os.path.exists(zip_path):
            os.remove(zip_path)
        raise RuntimeError("Downloaded file appears to be corrupted. Please try again.") from e

def load_movielens(data_dir: str = "data"):
    base = os.path.join(data_dir, ML_SMALL_DIRNAME)
    movies = pd.read_csv(os.path.join(base, "movies.csv"))
    ratings = pd.read_csv(os.path.join(base, "ratings.csv"))
    # Normalize genres field to consistent list form
    movies["genres"] = movies["genres"].fillna("(no genres listed)")
    return movies, ratings
