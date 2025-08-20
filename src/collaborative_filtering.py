import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

def fit_cf_model(ratings_df: pd.DataFrame):
    # Map IDs to continuous indices
    user_ids = ratings_df["userId"].unique()
    movie_ids = ratings_df["movieId"].unique()
    user_id_to_idx = {uid: i for i, uid in enumerate(sorted(user_ids))}
    movie_id_to_idx = {mid: i for i, mid in enumerate(sorted(movie_ids))}
    idx_to_movie_id = {i: mid for mid, i in movie_id_to_idx.items()}

    # Build sparse user-item matrix
    rows = ratings_df["userId"].map(user_id_to_idx).values
    cols = ratings_df["movieId"].map(movie_id_to_idx).values
    data = ratings_df["rating"].values.astype(np.float32)
    n_users = len(user_id_to_idx)
    n_items = len(movie_id_to_idx)
    mat = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))

    # Normalize per user (mean-center) for similarity
    user_means = np.array(mat.sum(axis=1)).reshape(-1) / np.maximum(1, (mat != 0).sum(axis=1).A1)
    user_means = np.nan_to_num(user_means, nan=0.0)
    mat_centered = mat.copy().astype(np.float32)
    # Subtract user mean from non-zero entries
    mat_centered = mat_centered.tolil()
    for u in range(mat_centered.shape[0]):
        start, end = mat_centered.rows[u][0:1], None  # noop; we loop anyway
        if len(mat_centered.data[u]) > 0:
            mat_centered.data[u] = [val - user_means[u] for val in mat_centered.data[u]]
    mat_centered = mat_centered.tocsr()

    return {
        "mat": mat,
        "mat_centered": mat_centered,
        "user_means": user_means,
        "user_id_to_idx": user_id_to_idx,
        "movie_id_to_idx": movie_id_to_idx,
        "idx_to_movie_id": idx_to_movie_id,
    }

def _top_k_similar_users(target_vec, user_matrix_centered, k: int = 40):
    # Compute cosine similarity between target user and all users
    sims = cosine_similarity(target_vec, user_matrix_centered).ravel()
    # Exclude the user itself (similarity of 1.0 at its index)
    return sims

def recommend_for_user(user_id: int, cf_bundle: dict, movies_df: pd.DataFrame, ratings_df: pd.DataFrame, top_n: int = 10, k: int = 40, min_ratings_per_item: int = 5):
    mat = cf_bundle["mat"]
    mat_c = cf_bundle["mat_centered"]
    user_means = cf_bundle["user_means"]
    user_id_to_idx = cf_bundle["user_id_to_idx"]
    idx_to_movie_id = cf_bundle["idx_to_movie_id"]

    if user_id not in user_id_to_idx:
        raise ValueError(f"User {user_id} not found.")

    u_idx = user_id_to_idx[user_id]
    target_row = mat_c.getrow(u_idx)

    # Compute similarities to all users
    sims_all = _top_k_similar_users(target_row, mat_c, k=k)
    sims_all[u_idx] = 0.0  # exclude self

    # Get top-k neighbors
    neighbor_idx = np.argsort(-sims_all)[:k]
    neighbor_sims = sims_all[neighbor_idx]

    # Items the target user has already rated
    rated_items = set(mat.getrow(u_idx).indices.tolist())

    # Predict scores for all items not yet rated
    # Weighted sum of neighbor deviations + user's mean
    # r_hat(u, i) = mu_u + sum_v(sim(u,v) * (r(v,i) - mu_v)) / sum_v(|sim(u,v)|) over neighbors v who rated i
    # Build candidate items from neighbors' rated items
    cand_items = set()
    for v in neighbor_idx:
        cand_items.update(mat.getrow(v).indices.tolist())
    cand_items = list(cand_items - rated_items)

    if not cand_items:
        # Fallback: popular items
        popular = (
            ratings_df.groupby("movieId")["rating"]
            .agg(["count", "mean"])
            .query("count >= @min_ratings_per_item")
            .sort_values(["mean", "count"], ascending=[False, False])
            .head(top_n)
            .reset_index()
        )
        merged = popular.merge(movies_df, on="movieId", how="left")
        merged = merged[["movieId", "title", "genres", "mean", "count"]]
        merged.insert(0, "rank", range(1, len(merged) + 1))
        merged = merged.rename(columns={"mean": "predicted_rating", "count": "num_ratings"})
        return merged

    # Prepare neighbor data
    neighbor_rows = mat[neighbor_idx]
    neighbor_means = user_means[neighbor_idx]
    # For each candidate item, compute weighted prediction
    preds = []
    denom_eps = 1e-8
    for i in cand_items:
        # Get neighbor ratings for item i
        col = neighbor_rows[:, i].toarray().ravel()
        mask = col > 0
        if not np.any(mask):
            continue
        sims = neighbor_sims[mask]
        r_vi = col[mask]
        mu_v = neighbor_means[mask]
        num = np.sum(sims * (r_vi - mu_v))
        den = np.sum(np.abs(sims)) + denom_eps
        r_hat = user_means[u_idx] + num / den
        preds.append((i, r_hat))

    if not preds:
        return pd.DataFrame(columns=["rank", "movieId", "title", "genres", "predicted_rating"])

    preds.sort(key=lambda x: x[1], reverse=True)
    top_preds = preds[:top_n]
    movie_ids = [idx_to_movie_id[i] for i, _ in top_preds]
    scores = [s for _, s in top_preds]

    recs = pd.DataFrame({"movieId": movie_ids, "predicted_rating": scores})
    recs = recs.merge(movies_df, on="movieId", how="left")
    recs.insert(0, "rank", range(1, len(recs) + 1))
    recs = recs[["rank", "movieId", "title", "genres", "predicted_rating"]]
    return recs