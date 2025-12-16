
from fastapi import FastAPI
from recommender import recommend_songs

app = FastAPI(
    title="Song Recommendation System",
    description="Content-Based Song Recommendation using TF-IDF & Cosine Similarity",
    version="1.0"
)

@app.get("/")
def home():
    return {"message": "Song Recommendation API is running"}

@app.get("/recommend")
def recommend(song: str):
    clean_song = song.strip()   # 👈 IMPORTANT
    recommendations = recommend_songs(clean_song)

    return {
        "selected_song": clean_song,
        "recommended_songs": recommendations
    }
