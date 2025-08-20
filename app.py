import os
import time
import zipfile
import pandas as pd
import streamlit as st

# Local imports — these must exist in your repo
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
    st.write("Content-Based and Collaborative Filtering recommendations using the MovieLens dataset.")

    data_dir = "data"
    t0 = time.perf_counter()
    with st.spinner("Loading data..."):
        try:
            movies_df, ratings_df = _load_data(data_dir)
        except Exception as e:
            st.error("Automatic download of the MovieLens dataset failed due to a network/SSL issue.")
            with st.expander("Error details"):
                st.code(str(e))
            st.info(
                "Use one of the options below:\n"
                "- Upload the dataset zip (ml-latest-small.zip) and I will extract it.\n"
                "- Or commit the extracted dataset under data/ml-latest-small/ in the repo.\n"
                "- If you control env vars, you can set MOVIELENS_VERIFY_SSL=false for this download only."
            )

            uploaded = st.file_uploader("Upload ml-latest-small.zip", type=["zip"], key="upload_zip")
            if uploaded is not None:
                try:
                    os.makedirs(data_dir, exist_ok=True)
                    zip_path = os.path.join(data_dir, "ml-latest-small.zip")
                    with open(zip_path, "wb") as f:
                        f.write(uploaded.read())
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        zf.extractall(path=data_dir)
                    st.success("Data uploaded and extracted. Click 'Reload data' to continue.")
                except zipfile.BadZipFile:
                    st.error("The uploaded file is not a valid zip. Upload the official ml-latest-small.zip from GroupLens.")
                except Exception as ex:
                    st.error(f"Failed to process uploaded file: {ex}")

            if st.button("Reload data", key="reload_after_upload"):
                st.cache_data.clear()
                st.rerun()
            st.stop()

    st.caption(f"Data loaded in {time.perf_counter() - t0:.2f}s")

    tabs = st.tabs(["Content-Based (Genres)", "Collaborative Filtering (User-Based)", "About"])

    # Tab 1: Content-Based
    with tabs[0]:
        st.subheader("Find similar movies by genre")

        movie_list = movies_df["title"].dropna().drop_duplicates().sort_values().tolist()
        if not movie_list:
            st.warning("No movies found in dataset.")
        else:
            default_title = "Toy Story (1995)" if "Toy Story (1995)" in movie_list else movie_list[0]
            selected_title = st.selectbox("Pick a movie:", movie_list, index=movie_list.index(default_title), key="cb_pick_movie")

            # Unique key to avoid DuplicateWidgetID across tabs
            top_n = st.slider(
                "Number of recommendations:",
                min_value=5, max_value=20, value=10, step=1,
                key="cb_top_n"
            )

            if st.button("Recommend (Content-Based)", key="cb_recommend_btn"):
                t1 = time.perf_counter()
                with st.spinner("Building/Loading content model..."):
                    sim_index, title_to_index = _fit_content(movies_df)
                with st.spinner("Computing recommendations..."):
                    recs = recommend_by_title(
                        selected_title,
                        sim_index,
                        title_to_index,
                        movies_df,
                        top_n=top_n
                    )
                st.success(f"Top {top_n} movies similar to: {selected_title}")
                st.caption(f"Time: {time.perf_counter() - t1:.2f}s (first run can be slower; cached after) ")
                st.dataframe(recs, use_container_width=True)

    # Tab 2: Collaborative Filtering
    with tabs[1]:
        st.subheader("Recommend for an existing user (MovieLens)")
        st.caption("Tip: Try user IDs between 1 and 610 (MovieLens small).")

        user_ids = ratings_df["userId"].dropna().astype(int).unique().tolist()
        user_ids.sort()

        if not user_ids:
            st.warning("No user IDs available in ratings.")
        else:
            min_uid = int(min(user_ids))
            max_uid = int(max(user_ids))
            default_user = int(user_ids[0])

            user_id = st.number_input(
                "User ID",
                min_value=min_uid,
                max_value=max_uid,
                value=default_user,
                step=1,
                key="cf_user_id"
            )

            # Unique keys to avoid DuplicateWidgetID
            top_n_cf = st.slider(
                "Number of recommendations:",
                min_value=5, max_value=20, value=10, step=1,
                key="cf_top_n"
            )
            k_neighbors = st.slider(
                "Neighbors (k)",
                min_value=10, max_value=100, value=40, step=5,
                key="cf_k_neighbors"
            )

            if st.button("Recommend (Collaborative Filtering)", key="cf_recommend_btn"):
                if int(user_id) not in user_ids:
                    st.error(f"User ID {int(user_id)} not found in dataset.")
                else:
                    t2 = time.perf_counter()
                    with st.spinner("Building/Loading CF model..."):
                        cf_bundle = _fit_cf(ratings_df)
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
                    st.caption(f"Time: {time.perf_counter() - t2:.2f}s (first run can be slower; cached after) ")
                    st.dataframe(recs_cf, use_container_width=True)

    # Tab 3: About
    with tabs[2]:
        st.markdown(
            "- Dataset: MovieLens Latest Small (ml-latest-small) by GroupLens\n"
            "- Methods:\n"
            "  - Content-Based: Genre similarity using cosine similarity\n"
            "  - Collaborative Filtering: User-based k-NN with cosine similarity and weighted ratings\n"
            "- This is a simple, educational mini project suitable for quick demos.\n"
        )

if __name__ == "__main__":
    main()
