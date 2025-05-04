#!/usr/bin/env python
# coding: utf-8

# #
# ### Streamlit_app.py
# - Run Main code first to get file - final_custom_df.csv
# #

# In[1]:


#pip install wordcloud


# In[2]:


# streamlined_streamlit_app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
from itertools import product

# -----------------------------
# Page Setup & Styling
# -----------------------------
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto&display=swap');
html, body, [class*="css"] {
    font-family: 'Roboto', sans-serif;
}
h1, h2, h3, h4 {
    color: #333333;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("final_custom_df.csv", parse_dates=["datetime"])
    df.dropna(subset=["title", "rating"], inplace=True)
    return df

df = load_data()

# -----------------------------
# Dashboard Title
# -----------------------------
st.title("Movie Recommender Dashboard for Ages 18–35")
st.write(f"Dataset Loaded: {df.shape[0]:,} ratings")

# -----------------------------
# Age Group Filter
# -----------------------------
age_group = st.selectbox("Select Age Group", sorted(df["sim_age_group_pca"].dropna().unique()))
df_filtered = df[df["sim_age_group_pca"] == age_group]
st.write(f"Showing {df_filtered.shape[0]:,} ratings for age group: {age_group}")

# -----------------------------
# KPI Cards
# -----------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Total Ratings", f"{len(df_filtered):,}")
col2.metric("Unique Movies", f"{df_filtered['movieId'].nunique():,}")
col3.metric("Unique Users", f"{df_filtered['userId'].nunique():,}")

# -----------------------------
# Animated Genre Popularity
# -----------------------------
df_filtered['year'] = df_filtered['datetime'].dt.year
df_filtered['genres_list'] = df_filtered['genres'].str.split('|')
df_exploded = df_filtered.explode('genres_list')

# Year range filter
min_year = int(df_filtered['year'].min())
max_year = int(df_filtered['year'].max())
year_range = st.slider("Select Year Range", min_value=min_year, max_value=max_year,
                       value=(min_year, max_year), step=1)
df_exploded = df_exploded[df_exploded['year'].between(year_range[0], year_range[1])]

# Full index to ensure consistency
all_years = df_exploded['year'].unique()
all_genres = df_exploded['genres_list'].unique()
all_ages = df_exploded['sim_age_group_pca'].unique()
full_index = pd.DataFrame(product(all_ages, all_years, all_genres),
                          columns=['sim_age_group_pca', 'year', 'genres_list'])

# Group and merge
genre_year_counts = (
    df_exploded.groupby(['sim_age_group_pca', 'year', 'genres_list'])
    .size().reset_index(name='count')
)
merged = pd.merge(full_index, genre_year_counts,
                  on=['sim_age_group_pca', 'year', 'genres_list'],
                  how='left').fillna(0)

age_data = merged[merged['sim_age_group_pca'] == age_group]
fig_genre_animated = px.bar(
    age_data,
    x='genres_list', y='count', color='genres_list',
    animation_frame='year',
    title=f"Genre Popularity Over Time – Age {age_group}",
    color_discrete_sequence=px.colors.qualitative.Set3
)
st.plotly_chart(fig_genre_animated)

# -----------------------------
# Genre Distribution
# -----------------------------
st.subheader("Genre Popularity")
genre_counts = df_filtered["genres"].str.split("|").explode().value_counts().reset_index()
genre_counts.columns = ["Genre", "Count"]
fig_genre = px.bar(genre_counts, x="Genre", y="Count", title="Genre Distribution",
                   color='Count', color_continuous_scale='Viridis')
st.plotly_chart(fig_genre)

# -----------------------------
# Word Cloud for Tags
# -----------------------------
st.subheader("Common Tags Word Cloud")
tags = df_filtered['tag'].dropna().str.cat(sep=' ')
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(tags)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
st.pyplot(plt)

# -----------------------------
# Content-Based Recommendations
# -----------------------------
st.subheader("Content-Based Movie Recommendations")
df_content = df_filtered.copy()
df_content["content"] = df_content["genres"].fillna("") + " " + df_content["tag"].fillna("")
df_content = df_content[df_content["content"].str.strip() != ""]
df_content = df_content.drop_duplicates(subset=["title"]).reset_index(drop=True)

# Fit TF-IDF model
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df_content["content"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df_content.index, index=df_content["title"]).drop_duplicates()

# Movie dropdown
movie_titles = sorted(df_content["title"].dropna().unique())
selected_title = st.selectbox("Choose a movie", movie_titles)

# Recommendations
if selected_title:
    idx = indices[selected_title]
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    recommendations = df_content.iloc[movie_indices][["title", "genres"]]
    st.table(recommendations)


# #
# ## Download as app.py 
# ### Anaconda Prompt
# - conda activate base
# - cd "C:\Users\laris\Desktop\CCT\CA2-integrated Data ML"
# - streamlit run app.py
# 
# #

# In[ ]:





# In[ ]:




