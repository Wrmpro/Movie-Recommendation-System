import os
import streamlit as st
import pandas as pd

from src.data_prep import ensure_movielens_data, load_movielens
from src.content_based import fit_content_model, recommend_by_title
from src.collaborative_filtering import fit_cf_model, recommend_for_user

st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬", layout="wide")

@st.cache_data(show_spinner=False)
def _load_data(data_dir: str):
    ensure_movielens_data(data_dir)
    movies_df, ratings_df = load_movielens(data_dir)
    return movies_df, ratings_df

@st.cache_resource(show_spinner=False)
def _fit_content(movies_df: pd.DataFrame):
    return fit_content_model(movies_df)

@st.cache_resource(show_spinner=False)
def _fit_cf(ratings_df: pd.DataFrame):
    return fit_cf_model(ratings_df)

def main():
    st.title("🎬 Movie Recommendation System")
    st.write("Simple mini project with Content-Based and Collaborative Filtering recommendations.")

    data_dir = "data"
    with st.spinner("Loading data..."):
        movies_df, ratings_df = _load_data(data_dir)

    tabs = st.tabs(["Content-Based (Genres)", "Collaborative Filtering (User-Based)", "About"]) 

    with tabs[0]:
        st.subheader("Find similar movies by genre")
        movie_list = movies_df["title"].sort_values().unique().tolist()
        default_title = "Toy Story (1995)" if "Toy Story (1995)" in movie_list else movie_list[0]
        selected_title = st.selectbox("Pick a movie:", movie_list, index=movie_list.index(default_title))
        top_n = st.slider("Number of recommendations:", min_value=5, max_value=20, value=10, step=1)

        sim_index, title_to_index = _fit_content(movies_df)

        if st.button("Recommend (Content-Based)"):
            with st.spinner("Computing recommendations..."):
                recs = recommend_by_title(selected_title, sim_index, title_to_index, movies_df, top_n=top_n)
            st.success(f"Top {top_n} movies similar to: {selected_title}")
            st.dataframe(recs, use_container_width=True)

    with tabs[1]:
        st.subheader("Recommend for an existing user (MovieLens)")
        st.caption("Tip: Try user IDs between 1 and 610 (from the MovieLens small dataset).")
        user_ids = ratings_df["userId"].sort_values().unique().tolist()
        default_user = user_ids[0]
        user_id = st.number_input("User ID", min_value=int(min(user_ids)), max_value=int(max(user_ids)), value=int(default_user), step=1)
        top_n_cf = st.slider("Number of recommendations:", min_value=5, max_value=20, value=10, step=1)
        k_neighbors = st.slider("Neighbors (k)", min_value=10, max_value=100, value=40, step=5)

        cf_bundle = _fit_cf(ratings_df)

        if st.button("Recommend (Collaborative Filtering)"):
            if user_id not in user_ids:
                st.error(f"User ID {user_id} not found in dataset.")
            else:
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