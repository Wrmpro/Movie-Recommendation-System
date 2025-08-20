# 🎬 Movie Recommendation System

A simple mini project that recommends movies using:
- Content-Based Filtering (by genres)
- Collaborative Filtering (user-based k-NN)

This project is designed to be small, easy to run, and consistent with a clean repo structure and README style.

## ✨ Features
- Content-based recommendations: find movies similar to a selected title using genre similarity.
- Collaborative filtering: recommend movies for an existing user (from the MovieLens dataset) using user-based cosine similarity.
- Streamlit app for quick interactive demos.
- Automatic dataset download (MovieLens Latest Small).

## 🗂 Project Structure
```
.
├─ app.py                    # Streamlit app entry point
├─ requirements.txt
├─ src/
│  ├─ __init__.py
│  ├─ data_prep.py           # Data download + loading utilities
│  ├─ content_based.py       # Genre-based content model
│  └─ collaborative_filtering.py  # User-based CF model
├─ data/                     # Auto-populated (MovieLens) on first run
│  └─ ml-latest-small/
├─ README.md
└─ .gitignore
```

## 🚀 Quick Start

1) Create and activate a virtual environment (recommended)
```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

2) Install dependencies
```bash
pip install -r requirements.txt
```

3) Run the app
```bash
streamlit run app.py
```

The app will:
- Download the MovieLens small dataset on first run.
- Let you pick between Content-Based and Collaborative Filtering recommendations.

## 🧠 How It Works

- Content-Based
  - Parses movie genres (e.g., "Action|Adventure").
  - Uses MultiLabelBinarizer to build multi-hot vectors.
  - Computes cosine similarity to find movies similar to the selected title.

- Collaborative Filtering (User-Based)
  - Builds a user-item rating matrix from MovieLens ratings.
  - Computes cosine similarity between the target user and other users.
  - Predicts scores for unseen movies using a weighted average of neighbor deviations (mean-centered ratings).

## 🧪 Example Usage

- Content-Based:
  - Select "Toy Story (1995)" and get top-10 similar movies by genre.

- Collaborative Filtering:
  - Enter a user ID between 1 and 610 (MovieLens small dataset).
  - Get top-N recommendations predicted for that user.

## 📦 Dataset
- MovieLens Latest Small (ml-latest-small)
- Source: https://files.grouplens.org/datasets/movielens/
- Automatically downloaded to data/ml-latest-small/

## 🔧 Configuration
You can adjust:
- Number of recommendations (top-N)
- Number of neighbors (k) in collaborative filtering
- Minimum ratings per item for popularity fallback (in code)

## 🛣️ Roadmap / Future Improvements
- Hybrid recommendations (combine content and CF scores)
- Improved content features (e.g., TF-IDF of plot overviews)
- Model persistence and caching strategies
- Evaluation scripts (precision@k, MAP, NDCG)
- Dockerfile for containerized runs

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to submit a PR or open an issue.

## 📄 License
This project is for educational purposes. You can adapt it as needed. Add a LICENSE file (e.g., MIT) if you plan to distribute.

## 🙌 Acknowledgments
- MovieLens dataset by GroupLens
- Streamlit for rapid app development