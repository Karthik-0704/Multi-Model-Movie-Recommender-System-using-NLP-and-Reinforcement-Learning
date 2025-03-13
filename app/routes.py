from flask import Blueprint, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import random

main = Blueprint('main', __name__)

# Load interaction data (user_id, item_id, rating, timestamp)
interactions = pd.read_csv(
    "data/u.data", sep="\t", header=None, names=["user_id", "item_id", "rating", "timestamp"]
)

# Load item metadata (movie_id | movie_title | genres)
items = pd.read_csv(
    "data/u.item", sep="|", header=None, encoding="latin-1",
    names=[
        "item_id", "title", "release_date", "video_release_date", "IMDb_URL",
        "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
        "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
        "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]
)

# Keep only relevant columns for metadata
items = items[["item_id", "title", "Action", "Adventure", "Drama", "Sci-Fi", "Comedy"]]

# Content-Based Filtering
items["description"] = items[["Action", "Adventure", "Drama", "Sci-Fi", "Comedy"]].apply(
    lambda x: " ".join(x.index[x == 1]), axis=1
)

tfidf_vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf_vectorizer.fit_transform(items["description"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_content_recommendations(title, cosine_sim=cosine_sim, df=items):
    idx = df[df["title"] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_indices = [i[0] for i in sim_scores[1:6]]
    return df["title"].iloc[sim_indices].tolist()

# Collaborative Filtering
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(interactions[["user_id", "item_id", "rating"]], reader)
trainset, testset = train_test_split(data, test_size=0.2)
model = SVD()
model.fit(trainset)

def get_collaborative_recommendations(user_id, model=model, df_items=items):
    all_items = df_items["item_id"].unique()
    rated_items = interactions[interactions["user_id"] == user_id]["item_id"].unique()
    unrated_items = [item for item in all_items if item not in rated_items]
    predictions = [(item, model.predict(user_id, item).est) for item in unrated_items]
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:5]
    return df_items[df_items["item_id"].isin([p[0] for p in predictions])]["title"].tolist()

# Reinforcement Learning Simulation
def rl_simulation():
    actions = items["title"].tolist()
    q_values = {action: 0 for action in actions}
    
    for _ in range(100):  # Simulate 100 iterations
        action = random.choice(actions)
        reward = random.choice([0, 1])  # Simulated reward
        q_values[action] += reward

    sorted_actions = sorted(q_values.items(), key=lambda x: x[1], reverse=True)
    return [action for action, value in sorted_actions[:5]]

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/content-recommendations', methods=['GET'])
def content_recommendations():
    title = request.args.get('title')
    recommendations = get_content_recommendations(title)
    return jsonify({'recommendations': recommendations})

@main.route('/collaborative-recommendations', methods=['GET'])
def collaborative_recommendations():
    user_id = int(request.args.get('user_id'))
    recommendations = get_collaborative_recommendations(user_id)
    return jsonify({'recommendations': recommendations})

@main.route('/rl-recommendations', methods=['GET'])
def rl_recommendations():
    recommendations = rl_simulation()
    return jsonify({'recommendations': recommendations})
