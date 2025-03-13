# Movie Recommendation System with NLP and Reinforcement Learning

This **Movie Recommendation System** is a comprehensive platform designed to provide personalized movie recommendations using a combination of **Content-Based Filtering**, **Collaborative Filtering**, and **Reinforcement Learning (RL)** techniques. The system helps users find movies based on their preferences or past interactions and provides an intuitive and dynamic user interface to make movie discovery a seamless experience.

### 1. Content-Based Filtering (NLP-based):
- **TF-IDF Vectorizer**: The system uses the **Term Frequency-Inverse Document Frequency (TF-IDF)** algorithm to analyze movie descriptions based on their genres (Action, Drama, Comedy, etc.). This technique helps represent the genre information as vectors, capturing the importance of each genre's role in describing the movie.
- **Cosine Similarity**: To generate movie recommendations based on content, the system computes the cosine similarity between movies' genre-based vectors. This helps suggest movies that are similar in genre to the one the user inputs.

### 2. Collaborative Filtering (Matrix Factorization using SVD):
- The system uses **Singular Value Decomposition (SVD)**, a popular matrix factorization technique, to perform **Collaborative Filtering**. The method works by analyzing the user-item interaction matrix to discover latent factors that explain users' preferences. It predicts missing ratings for movies that a user hasn't watched yet, helping recommend movies based on the preferences of similar users.

### 3. Reinforcement Learning (Simulated Agent):
- In the **Reinforcement Learning (RL)** component, a simulated agent is used to dynamically recommend movies based on user interaction and feedback. The agent receives rewards (positive feedback) when it recommends movies that the user likes and adjusts its recommendations based on the cumulative rewards. This helps the agent improve its recommendation strategy over time. The **Q-Learning** algorithm is employed to update the action-value function (Q-values) based on the rewards received.

## Tech Stack:

### 1. Backend:
- **Flask**: A lightweight Python web framework used to create the REST API for serving movie recommendations and handling user requests.
- **Python**: The core programming language used for implementing algorithms such as TF-IDF, cosine similarity, SVD, and Reinforcement Learning.
- **pandas**: For handling and manipulating datasets (user interactions, movie metadata).
- **scikit-learn**: Used for machine learning algorithms such as TF-IDF vectorization, cosine similarity, and implementing SVD for collaborative filtering.
- **Surprise**: A Python library for building recommendation systems, which is used here for implementing the SVD-based collaborative filtering algorithm.

### 2. Frontend:
- **HTML**: To structure the content on the web page, providing an interactive interface for users to input data and view recommendations.
- **CSS**: For styling the user interface to make it visually appealing and responsive. Flexbox is used for centering elements.
- **JavaScript**: Handles asynchronous requests to the backend via **Fetch API** for retrieving movie recommendations dynamically based on user input.

### 3. Additional Libraries and Tools:
- **NLP Techniques**: TF-IDF Vectorizer from **scikit-learn** is used for converting movie genre data into numerical vectors for similarity calculation.
- **Q-Learning for RL**: A basic implementation of **Q-learning** is employed to simulate the learning agent for movie recommendation based on user feedback.

This system uses a dynamic combination of advanced recommendation algorithms and user-friendly UI to help users discover movies that fit their preferences.
