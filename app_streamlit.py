"""Small demo UI: pick a user, see their history next to recommendations.

Calls src/pipeline.py directly (no need to run the FastAPI server).
Run with: make demo   (streamlit run app_streamlit.py)
"""

import pandas as pd
import streamlit as st

from src.pipeline import RecommenderPipeline

st.set_page_config(page_title="Movie Recommender Demo", layout="wide")


@st.cache_resource
def load_pipeline() -> RecommenderPipeline:
    return RecommenderPipeline()


pipeline = load_pipeline()

st.title("Movie Recommender Demo")

user_id = st.number_input("User ID", min_value=1, value=1, step=1)
k = st.slider("Number of recommendations", min_value=5, max_value=20, value=10)

is_known = pipeline.is_known_user(int(user_id))
if not is_known:
    st.warning(f"User {user_id} has no training history — showing popularity-based cold-start recommendations.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("History (movies they liked)")
    history = pipeline.get_history(int(user_id), k=20)
    if history:
        st.dataframe(pd.DataFrame(history)[["title", "genres"]], hide_index=True)
    else:
        st.write("No history on record for this user.")

with col2:
    st.subheader(f"Top-{k} Recommendations")
    recs = pipeline.recommend(int(user_id), k=k)
    st.dataframe(pd.DataFrame(recs)[["title", "genres", "score"]], hide_index=True)
