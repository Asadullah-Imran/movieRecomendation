# app.py
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
import os

# Initialize FastAPI app
app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load datasets (in a real app, you'd load from CSV files)
def load_data():
    # Movies data
    movies = pd.DataFrame({
        'movieId': [1, 2, 3, 4, 5],
        'title': [
            'Toy Story (1995)',
            'Jumanji (1995)',
            'Grumpier Old Men (1995)',
            'Waiting to Exhale (1995)',
            'Father of the Bride Part II (1995)'
        ],
        'genres': [
            'Animation|Adventure|Children|Fantasy|Comedy',
            'Adventure|Children|Fantasy|Comedy',
            'Comedy|Romance|Children|Fantasy',
            'Comedy|Drama|Romance',
            'Comedy|Fantasy'
        ]
    })
    
    # Ratings data
    ratings = pd.DataFrame({
        'userId': [1,1,1,1,1,1,1,4,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,7,7],
        'movieId': [1,3,6,47,50,70,101,4765,4881,4896,4902,4967,1,21,34,36,1064,1073,1082,1084,1,50,58,150,165,260],
        'rating': [4.0,4.0,4.0,5.0,5.0,3.0,5.0,5.0,3.0,4.0,4.0,5.0,4.0,4.0,4.0,4.0,4.0,3.0,4.0,3.0,4.5,4.5,3.0,4.5,4.0,5.0]
    })
    
    # Links data
    links = pd.DataFrame({
        'movieId': [1, 2, 3, 4, 5],
        'imdbId': ['0114709', '0113497', '0113228', '0114885', '0113041'],
        'tmdbId': [862, 8844, 15602, 31357, 11862]
    })
    
    return movies, ratings, links

# Initialize data and models
movies, ratings, links = load_data()

# Create hybrid features
tfidf = TfidfVectorizer(analyzer=lambda s: s.split('|'))
content_features = tfidf.fit_transform(movies['genres']).toarray()

# Create collaborative features
valid_movies = movies['movieId'].unique()
ratings_filtered = ratings[ratings['movieId'].isin(valid_movies)]
rating_matrix = ratings_filtered.pivot_table(
    index='movieId', 
    columns='userId', 
    values='rating',
    fill_value=0
).reindex(valid_movies, fill_value=0).values

# Combine features
content_weight = 0.7
collab_weight = 0.3
hybrid_features = np.hstack((
    content_weight * content_features,
    collab_weight * rating_matrix
))

# Create mapping between movie titles and indices
movie_title_to_idx = {title: idx for idx, title in enumerate(movies['title'])}
movie_idx_to_id = {idx: movie_id for idx, movie_id in enumerate(movies['movieId'])}

# Euclidean distance calculation
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Find k nearest neighbors
def knn_find_neighbors(X, x_query, k):
    distances = [euclidean_distance(x, x_query) for x in X]
    k_indices = np.argsort(distances)[:k]
    return k_indices

# Get movie recommendations
def get_recommendations(movie_title, k=3):
    if movie_title not in movie_title_to_idx:
        return []
    
    query_idx = movie_title_to_idx[movie_title]
    neighbor_indices = knn_find_neighbors(hybrid_features, hybrid_features[query_idx], k+1)
    
    recommendations = []
    for idx in neighbor_indices:
        if idx != query_idx:  # Exclude the query movie itself
            movie_id = movie_idx_to_id[idx]
            movie_title = movies[movies['movieId'] == movie_id]['title'].values[0]
            
            # Get TMDB ID for poster
            tmdb_id = links[links['movieId'] == movie_id]['tmdbId'].values[0]
            recommendations.append({
                'title': movie_title,
                'tmdb_id': tmdb_id
            })
    
    return recommendations[:k]

# Get movie poster URL from TMDB
def get_poster_url(tmdb_id):
    try:
        # Get poster path from TMDB API
        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{tmdb_id}",
            params={
                "api_key": "741a40962c8a0dee565jfff2eacbe",  # Replace with your TMDB API key
                "language": "en-US"
            }
        )
        data = response.json()
        poster_path = data.get('poster_path')
        
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
        return "https://via.placeholder.com/200x300?text=No+Poster"
    except:
        return "https://via.placeholder.com/200x300?text=No+Poster"

# API endpoint for recommendations
@app.get("/recommend/{movie_title}")
async def recommend(movie_title: str):
    recommendations = get_recommendations(movie_title)
    
    # Add poster URLs to recommendations
    for movie in recommendations:
        movie['poster'] = get_poster_url(movie['tmdb_id'])
    
    return {
        "query": movie_title,
        "recommendations": recommendations
    }

# Home page with movie list
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # Add poster URLs to movies
    movie_list = []
    for _, row in movies.iterrows():
        movie_id = row['movieId']
        tmdb_id = links[links['movieId'] == movie_id]['tmdbId'].values[0]
        poster = get_poster_url(tmdb_id)
        movie_list.append({
            'title': row['title'],
            'genres': row['genres'],
            'poster': poster
        })
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "movies": movie_list
    })

# Recommendation page
@app.get("/recommendation/{movie_title}", response_class=HTMLResponse)
async def recommendation_page(request: Request, movie_title: str):
    recommendations = get_recommendations(movie_title)
    
    # Get poster for query movie
    if movie_title in movie_title_to_idx:
        query_idx = movie_title_to_idx[movie_title]
        movie_id = movie_idx_to_id[query_idx]
        tmdb_id = links[links['movieId'] == movie_id]['tmdbId'].values[0]
        query_poster = get_poster_url(tmdb_id)
    else:
        query_poster = "https://via.placeholder.com/200x300?text=No+Poster"
    
    # Add poster URLs to recommendations
    for movie in recommendations:
        movie['poster'] = get_poster_url(movie['tmdb_id'])
    
    return templates.TemplateResponse("recommendation.html", {
        "request": request,
        "query_movie": movie_title,
        "query_poster": query_poster,
        "recommendations": recommendations
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)