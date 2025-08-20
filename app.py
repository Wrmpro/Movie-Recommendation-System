import os
import time
import zipfile
import streamlit as st
import pandas as pd

from src.data_prep import ensure_movielens_data, load_movielens
from src.content_based import fit_content_model, recommend_by_title
from src.collaborative_filtering import fit_cf_model, recommend_for_user

st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬", layout="wide")

@st.cache_data(show_spinner=False)
def _load_data(data_dir: str):
    ensure_movielens_data(data_dir)  # If this downloads on cold start, it will slow the first run.
    movies_df, ratings_df = load_movielens(data_dir)
    return movies_df, ratings_df

@st.cache_resource(show_spinner=False)
def _fit_content(movies_df: pd.DataFrame):
    # Heavy step: vectorizer fit + similarity matrix. Cached after first build.
    return fit_content_model(movies_df)

@st.cache_resource(show_spinner=False)
def _fit_cf(ratings_df: pd.DataFrame):
    # Heavy step: CF model/NN index build. Cached after first build.
    return fit_cf_model(ratings_df)

def main():
    st.title("🎬 Movie Recommendation System")
    st.write("Simple mini project with Content-Based and Collaborative Filtering recommendations.")

    data_dir = "data"
    t0 = time.perf_counter()
    with st.spinner("Loading data..."):
        try:
            movies_df, ratings_df = _load_data(data_dir)
        except Exception as e:
            st.error("Automatic download of the MovieLens dataset failed due to a network/SSL issue.")
            with st.expander("Details"):
                st.code(str(e))
            st.info(
                "Option A: Upload the dataset zip (ml-latest-small.zip) below.\n"
                "Option B: Commit the dataset to the repository under data/ml-latest-small/.\n"
                "Option C: Set environment variable MOVIELENS_VERIFY_SSL=false (not always available on hosted platforms)."
            )
            uploaded = st.file_uploader("Upload ml-latest-small.zip", type=["zip"])
            if uploaded is not None:
                os.makedirs(data_dir, exist_ok=True)
                zip_path = os.path.join(data_dir, "ml-latest-small.zip")
                with open(zip_path, "wb") as f:
                    f.write(uploaded.read())
                try:
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        zf.extractall(path=data_dir)
                    st.success("Data uploaded and extracted. Click 'Reload data' to continue.")
                    if st.button("Reload data"):
                        st.cache_data.clear()
                        st.rerun()
                except zipfile.BadZipFile:
                    st.error("The uploaded file is not a valid zip. Please upload ml-latest-small.zip from GroupLens.")
            st.stop()
    st.caption(f"Data loaded in {time.perf_counter() - t0:.2f}s")

    tabs = st.tabs(["Content-Based (Genres)", "Collaborative Filtering (User-Based)", "About"])

    with tabs[0]:
        st.subheader("Find similar movies by genre")
        movie_list = movies_df["title"].dropna().drop_duplicates().sort_values().tolist()
        default_title = "Toy Story (1995)" if "Toy Story (1995)" in movie_list else (movie_list[0] if movie_list else "")
        default_index = movie_list.index(default_title) if default_title in movie_list else 0
        selected_title = st.selectbox("Pick a movie:", movie_list, index=default_index)
        top_n = st.slider("Number of recommendations:", min_value=5, max_value=20, value=10, step=1)

        if st.button("Recommend (Content-Based)"):
            t1 = time.perf_counter()
            with st.spinner("Building/Loading content model..."):
                sim_index, title_to_index = _fit_content(movies_df)  # Built only when needed (first click)
            with st.spinner("Computing recommendations..."):
                recs = recommend_by_title(selected_title, sim_index, title_to_index, movies_df, top_n=top_n)
            st.success(f"Top {top_n} movies similar to: {selected_title}")
            st.caption(f"Total time: {time.perf_counter() - t1:.2f}s (first run will be slower, next runs use cache)")
            st.dataframe(recs, use_container_width=True)

    with tabs[1]:
        st.subheader("Recommend for an existing user (MovieLens)")
        st.caption("Tip: Try user IDs between 1 and 610 (MovieLens small).")
        user_ids = ratings_df["userId"].dropna().astype(int).unique().tolist()
        user_ids.sort()
        default_user = user_ids[0] if user_ids else 1
        user_id = st.number_input("User ID", min_value=int(min(user_ids)), max_value=int(max(user_ids)), value=int(default_user), step=1)
        top_n_cf = st.slider("Number of recommendations:", min_value=5, max_value=20, value=10, step=1)
        k_neighbors = st.slider("Neighbors (k)", min_value=10, max_value=100, value=40, step=5)

        if st.button("Recommend (Collaborative Filtering)"):
            if user_id not in user_ids:
                st.error(f"User ID {user_id} not found in dataset.")
            else:
                t2 = time.perf_counter()
                with st.spinner("Building/Loading CF model..."):
                    cf_bundle = _fit_cf(ratings_df)  # Built only when needed (first click)
                with st.spinner("Computing recommendations..."):
                    recs_cf = recommend_for_user(
                        user_id=int(user_id),
                        cf_bundle=cf_bundle,
                        movies_df=movies_df,
                        ratings_df=ratings_df,
                        top_n=top_n_cf,
                        k=k_neighbors
                    )
                st.success(f"Top {top_n_cf} recommendations for user {int(user_id)}")
                st.caption(f"Total time: {time.perf_counter() - t2:.2f}s (first run will be slower, next runs use cache)")
                st.dataframe(recs_cf, use_container_width=True)

    with tabs[2]:
        st.markdown("""
        - Dataset: MovieLens Latest Small (ml-latest-small) by GroupLens
        - Methods:
          - Content-Based: Genre similarity using cosine similarity
          - Collaborative Filtering: User-based k-NN with cosine similarity and weighted ratings
        - This is a simple, educational mini project suitable for quick demos.
        """)

if __name__ == "__main__":
    main()
