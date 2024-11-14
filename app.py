from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

# Load preprocessed datasets
books = pd.read_csv('Books_Cleaned.csv')
ratings = pd.read_csv('Ratings_Cleaned.csv')

# Merge ratings with books data
ratings_books = pd.merge(ratings, books, left_on='ISBN', right_on='book_id', how='inner')
ratings_books = ratings_books[['UserID', 'book_title', 'Rating']].dropna()

# Define the recommendation function
def recommend_books(user_id, num_recommendations=5):
    # Filter ratings for the given user
    user_ratings = ratings_books[ratings_books['UserID'] == user_id]
    
    # If user not found, fallback to popular books
    if user_ratings.empty:
        return ["User not found in dataset. Recommending popular books."] + list(
            ratings_books.groupby('book_title')['Rating']
            .mean()
            .reset_index()
            .sort_values(by='Rating', ascending=False)
            .head(num_recommendations)['book_title']
        )
    
    # Identify books rated by the user
    rated_books = set(user_ratings['book_title'])
    
    # Build a temporary dataset of similar users
    similar_users = ratings_books[ratings_books['book_title'].isin(rated_books) & (ratings_books['UserID'] != user_id)]
    
    if similar_users.empty:
        return ["No similar users found. Recommending popular books."] + list(
            ratings_books.groupby('book_title')['Rating']
            .mean()
            .reset_index()
            .sort_values(by='Rating', ascending=False)
            .head(num_recommendations)['book_title']
        )
    
    # Group by users and calculate similarity scores (e.g., shared ratings)
    user_similarity = (
        similar_users.groupby('UserID')['book_title']
        .nunique()
        .reset_index()
        .rename(columns={'book_title': 'similarity'})
        .sort_values(by='similarity', ascending=False)
    )
    
    # Get books rated by similar users
    top_similar_users = user_similarity['UserID'].head(10)
    similar_user_ratings = ratings_books[ratings_books['UserID'].isin(top_similar_users)]
    
    # Recommend books not rated by the input user
    recommendations = (
        similar_user_ratings[~similar_user_ratings['book_title'].isin(rated_books)]
        .groupby('book_title')['Rating']
        .mean()
        .reset_index()
        .sort_values(by='Rating', ascending=False)
    )
    
    recommended_books = recommendations['book_title'].head(num_recommendations).tolist()
    if not recommended_books:
        return ["No unique recommendations. Recommending popular books."] + list(
            ratings_books.groupby('book_title')['Rating']
            .mean()
            .reset_index()
            .sort_values(by='Rating', ascending=False)
            .head(num_recommendations)['book_title']
        )
    
    return recommended_books

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = int(request.form['user_id'])
    recommendations = recommend_books(user_id=user_id, num_recommendations=5)
    return render_template('results.html', user_id=user_id, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
