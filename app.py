# app.py
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import datetime

# --- Carregar os dados ---
netflix = pd.read_csv("netflix_titles.csv")
prime = pd.read_csv("amazon_prime_titles.csv")
disney = pd.read_csv("disney_plus_titles.csv")

netflix["platform"] = "Netflix"
prime["platform"] = "Amazon Prime"
disney["platform"] = "Disney+"

df = pd.concat([netflix, prime, disney], ignore_index=True)
df.dropna(subset=["title", "description"], inplace=True)

# --- Vetorização TF-IDF ---
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["description"])

# --- Similaridade ---
cosine_sim = cosine_similarity(tfidf_matrix)

# --- Popularidade temporal fictícia (substituir por real, se possível) ---
df["year_added"] = pd.to_datetime(df["date_added"], errors="coerce").dt.year
df["popularity_score"] = np.exp(-(2025 - df["year_added"].fillna(2020)))  # decaimento exponencial

# --- Interface ---
st.title("Sistema de Recomendação Híbrido")

selected_title = st.selectbox("Escolha um título:", df["title"].dropna().unique())

# --- Recomendações ---
def get_recommendations(title, top_n=10):
    idx = df[df["title"] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:100]  # ignora ele mesmo

    # Score híbrido
    results = []
    for i, sim in sim_scores:
        score = 0.7 * sim + 0.3 * df.iloc[i]["popularity_score"]
        results.append((i, score))

    top = sorted(results, key=lambda x: x[1], reverse=True)[:top_n]
    return df.iloc[[i for i, _ in top]]

# --- Abas por plataforma ---
tab1, tab2, tab3 = st.tabs(["Netflix", "Amazon Prime", "Disney+"])
for tab, platform in zip([tab1, tab2, tab3], ["Netflix", "Amazon Prime", "Disney+"]):
    with tab:
        st.subheader(f"Top 10 recomendados na {platform}")
        recs = get_recommendations(selected_title)
        recs = recs[recs["platform"] == platform].head(10)
        for _, row in recs.iterrows():
            st.markdown(f"- **{row['title']}** ({row.get('release_year', 'N/A')})")

