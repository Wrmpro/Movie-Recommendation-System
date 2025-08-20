import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

def _parse_genres(genre_str: str):
    if pd.isna(genre_str) or genre_str.strip() == "":
        return []
    if genre_str == "(no genres listed)":
        return []
    return genre_str.split("|")

def fit_content_model(movies_df: pd.DataFrame):
    movies = movies_df.copy()
    movies["genre_list"] = movies["genres"].apply(_parse_genres)
    mlb = MultiLabelBinarizer(sparse_output=False)
    genre_features = mlb.fit_transform(movies["genre_list"])
    # Cosine similarity over genre features
    sim_index = cosine_similarity(genre_features)
    # Map title to index for quick lookup
    title_to_index = pd.Series(movies.index.values, index=movies["title"]).to_dict()
    return sim_index, title_to_index

def recommend_by_title(title: str, sim_index, title_to_index, movies_df: pd.DataFrame, top_n: int = 10):
    if title not in title_to_index:
        raise ValueError(f"Title '{title}' not found.")
    idx = title_to_index[title]
    sims = sim_index[idx]
    # Exclude the same movie
    similar_indices = np.argsort(-sims)
    similar_indices = [i for i in similar_indices if i != idx][:top_n]
    recs = movies_df.iloc[similar_indices][["movieId", "title", "genres"]].reset_index(drop=True)
    recs.insert(0, "rank", range(1, len(recs) + 1))
    return recs