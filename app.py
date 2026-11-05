"""Streamlit demo: `streamlit run app.py`."""
from pathlib import Path

import pandas as pd
import streamlit as st

from movierec.data import load_data, search_title, split_per_user
from movierec.evaluate import evaluate_all
from movierec.models import build_models

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")
st.title("🎬 Movie Recommendation System")


@st.cache_resource(show_spinner="Training models...")
def load():
    movies, ratings = load_data()
    return movies, ratings, build_models(ratings, movies)


movies, ratings, models = load()
model_name = st.sidebar.selectbox("Model", list(models), index=list(models).index("Hybrid"))
k = st.sidebar.slider("Number of recommendations", 5, 20, 10)
model = models[model_name]
tab1, tab2, tab3 = st.tabs(["For a user", "Similar movies", "Model comparison"])

with tab1:
    uid = st.selectbox("User ID", model.user_ids)
    liked = ratings[(ratings.userId == uid) & (ratings.rating >= 4.5)].merge(movies, on="movieId")
    c1, c2 = st.columns(2)
    c1.subheader("Movies this user loved")
    c1.dataframe(liked[["title", "genres", "rating"]].head(10), hide_index=True)
    c2.subheader(f"Recommended ({model_name})")
    c2.dataframe(model.describe(model.recommend(uid, k)), hide_index=True)

with tab2:
    query = st.text_input("Movie title", movies.iloc[0]["title"].split(" (")[0])
    found = search_title(movies, query)
    if found.empty:
        st.warning("No match.")
    else:
        pick = st.selectbox("Did you mean", found["title"].head(20))
        mid = int(found[found.title == pick].iloc[0]["movieId"])
        st.dataframe(models["ItemKNN"].describe(models["ItemKNN"].similar(mid, k)), hide_index=True)

with tab3:
    path = Path("results/metrics.csv")
    if st.button("Run evaluation (temporal split)") or path.exists():
        if not path.exists() or st.session_state.get("rerun"):
            train, test = split_per_user(ratings)
            evaluate_all(build_models(train, movies), test).round(4).to_csv(path)
        st.dataframe(pd.read_csv(path, index_col=0))
