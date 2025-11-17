

import pickle
import streamlit as st
import requests
import pandas as pd
import numpy as np

# -----------------------------
# Function to fetch movie poster
# -----------------------------
def fetch_poster(movie_id):
    """Fetches the movie poster URL from TMDB API."""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
    except requests.exceptions.RequestException:
        pass
    # Return placeholder if poster not found or request fails
    return "https://placehold.co/500x750/333/FFFFFF?text=No+Poster"

# -----------------------------
# Function to recommend movies
# -----------------------------
@st.cache_data(show_spinner=False)
def recommend(movie):
    """Recommends 5 similar movies based on the selected movie."""
    try:
        index = movies[movies['title'] == movie].index[0]
    except IndexError:
        st.error("Movie not found in the dataset. Please select another one.")
        return [], [], [], []
    
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    
    recommended_names = []
    recommended_posters = []
    recommended_years = []
    recommended_ratings = []

    for i in distances[1:6]:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_posters.append(fetch_poster(movie_id))
        recommended_names.append(movies.iloc[i[0]].title)
        recommended_years.append(movies.iloc[i[0]].year)
        recommended_ratings.append(movies.iloc[i[0]].vote_average)

    return recommended_names, recommended_posters, recommended_years, recommended_ratings

# -----------------------------
# Streamlit page setup
# -----------------------------
st.set_page_config(layout="wide")
st.title('🎬 Movie Recommender System Using Machine Learning')

# -----------------------------
# Load model files
# -----------------------------
try:
    movies_dict = pickle.load(open('artifacts/movie_dict.pkl', 'rb'))
    movies = pd.DataFrame(movies_dict)
    similarity = pickle.load(open('artifacts/similarity.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model files not found. Please run the data processing notebook first.")
    st.stop()

# -----------------------------
# Movie selection
# -----------------------------
movie_list = movies['title'].values
selected_movie = st.selectbox("Type or select a movie from the dropdown", movie_list)

# -----------------------------
# Show recommendations
# -----------------------------
if st.button('Show Recommendation'):
    with st.spinner('Finding recommendations...'):
        names, posters, years, ratings = recommend(selected_movie)
    
    if names:
        cols = st.columns(5)
        for i, col in enumerate(cols):
            with col:
                st.text(names[i])
                st.image(posters[i])
                st.caption(f"Year: {int(years[i]) if pd.notna(years[i]) else 'N/A'}")
                st.caption(f"Rating: {ratings[i]:.1f} ⭐")
    else:
        st.warning("No recommendations found.")