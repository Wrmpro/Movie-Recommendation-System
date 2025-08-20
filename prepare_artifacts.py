import os
import joblib
import pandas as pd

# This script is intended to be run locally by the repository owner to precompute
# heavy artifacts (vectorizer/similarity) and save them to artifacts/recommender.joblib
# so that the Streamlit app can load them quickly at runtime.

from src.content_based import fit_content_model


def main():
    data_dir = "data"
    movies_path = os.path.join(data_dir, "movies.csv")
    if not os.path.exists(movies_path):
        raise SystemExit("data/movies.csv not found. Download ml-latest-small locally and place movies.csv in data/ before running this.")

    movies_df = pd.read_csv(movies_path)
    artifact = fit_content_model(movies_df)
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(artifact, "artifacts/recommender.joblib")
    print("Saved artifacts/recommender.joblib")


if __name__ == '__main__':
    main()