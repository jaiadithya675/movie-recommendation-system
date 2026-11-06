"""Streamlit demo: `streamlit run app.py`."""
import json

import numpy as np
import pandas as pd
import streamlit as st

from movierec.data import search_title
from movierec.pipeline import RESULTS, load_models

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")
st.title("🎬 Movie Recommendation System")


@st.cache_resource(show_spinner="Loading models (the first run trains them, ~20 s)...")
def get_models():
    return load_models()


models = get_models()
names = list(models)
name = st.sidebar.selectbox("Model", names, index=names.index("Hybrid"))
k = st.sidebar.slider("Number of recommendations", 5, 20, 10)
model, knn, content = models[name], models["ItemKNN"], models["ContentBased"]
movies = model.movies.reset_index()
st.sidebar.caption(f"{len(model.user_ids)} users · {len(model.item_ids)} movies")
tab1, tab2, tab3, tab4 = st.tabs(["👤 Existing user", "🆕 New user (live)", "🎞 Similar movies", "📊 Model comparison"])

with tab1:
    uid = st.selectbox("User ID", model.user_ids)
    row = model.R[model.uidx[uid]]
    order = np.argsort(-row.data, kind="stable")[:8]
    fav = model.movies.loc[model.item_ids[row.indices[order]], ["title", "genres"]].assign(rating=row.data[order])
    c1, c2 = st.columns(2)
    c1.subheader("Their highest-rated movies")
    c1.dataframe(fav.reset_index(drop=True), hide_index=True)
    c2.subheader(f"Recommended by {name}")
    c2.dataframe(model.describe(model.recommend(uid, k)), hide_index=True)

with tab2:
    st.write("Rate a few movies and get recommendations instantly, with no retraining (item-based CF).")
    titles = model.movies.loc[model.item_ids[np.argsort(-model.pop_count)], "title"].tolist()
    id_of = dict(zip(movies["title"], movies["movieId"]))
    defaults = [t for t in titles if t.startswith(("Toy Story (", "Matrix, The", "Pulp Fiction"))][:3] or titles[:3]
    picked = st.multiselect("Movies you have seen", titles, default=defaults, max_selections=10)
    ratings = {id_of[t]: st.slider(t, 0.5, 5.0, 4.5, 0.5, key=f"rate_{t}") for t in picked}
    if ratings:
        st.dataframe(knn.describe(knn.recommend_for_ratings(ratings, k)), hide_index=True)
    else:
        st.info("Pick at least one movie.")

with tab3:
    query = st.text_input("Movie title", movies.iloc[0]["title"].split(" (")[0])
    found = search_title(movies, query)
    if found.empty:
        st.warning("No match.")
    else:
        pick = st.selectbox("Did you mean", found["title"].head(20))
        mid = int(found[found.title == pick].iloc[0]["movieId"])
        a, b = st.columns(2)
        a.subheader("By viewing patterns (ItemKNN)")
        a.dataframe(knn.describe(knn.similar(mid, k)), hide_index=True)
        b.subheader("By genres and tags (content)")
        b.dataframe(content.describe(content.similar(mid, k)), hide_index=True)

with tab4:
    metrics = RESULTS / "metrics.csv"
    if metrics.exists():
        st.dataframe(pd.read_csv(metrics, index_col=0))
        for img in ("model_comparison.png", "eda_overview.png"):
            if (RESULTS / img).exists():
                st.image(str(RESULTS / img))
        if (RESULTS / "best_params.json").exists():
            st.caption("Tuned hyperparameters")
            st.json(json.loads((RESULTS / "best_params.json").read_text()))
    else:
        st.info("Run `make all` to generate results.")
