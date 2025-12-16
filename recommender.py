import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("data/songs.csv")
df.fillna("", inplace=True)

# Clean song names
df["song"] = df["song"].str.strip()

df["combined_features"] = (
    df["artist"] + " " +
    df["song"] + " " +
    df["text"]
)

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = vectorizer.fit_transform(df["combined_features"])


def recommend_songs(song_name, num_recommendations=5):
    song_name = song_name.strip()

    if song_name not in df["song"].values:
        return ["Song not found"]

    idx = df[df["song"] == song_name].index[0]

    cosine_scores = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    similar_indices = cosine_scores.argsort()[::-1][1:num_recommendations + 1]

    songs = df.iloc[similar_indices]["song"].tolist()

    # Remove duplicate song names
    unique_songs = list(dict.fromkeys(songs))

    return unique_songs

