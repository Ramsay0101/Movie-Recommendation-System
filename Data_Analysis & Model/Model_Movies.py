import pandas as pd
import numpy as np
import pickle 
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# Load the datasets containing movie metadata, credits, and links
movies = pd.read_csv('movies_metadata.csv', on_bad_lines='skip')
credits = pd.read_csv('credits.csv')
links = pd.read_csv('links.csv')

# Display the first 5 rows 
print(movies.head(5))
print(movies.shape)
print(movies.columns)

# Keep only relevant columns in the movies dataset and rename the imdbId column in links
movies = movies[['id', 'title', 'release_date', 'overview', 'tagline', 'poster_path', 'genres', 'imdb_id']]
links.rename(columns={"imdbId": "imdb_id"}, inplace=True)

# Ensure consistent data types and format for merging later
movies['imdb_id'] = movies['imdb_id'].astype(str)
links['imdb_id'] = 'tt0' + links['imdb_id'].astype(str)
credits['id'] = credits['id'].astype(str)
movies['id'] = movies['id'].astype(str)
links['movieId'] = links['movieId'].astype(str)

# Verify the alignment of unique values between datasets
print("Unique imdb_id in movies:")
print(movies['imdb_id'].unique()[:10])
print("Unique imdb_id in links:")
print(links['imdb_id'].unique()[:10])
print("Unique movieId in links:")
print(links['movieId'].unique()[:10])

# Merge the movies and links datasets on the 'imdb_id' column
movies = pd.merge(movies, links[['movieId', 'imdb_id']], on='imdb_id', how='left')
print(movies.head(5))
print(movies.shape)
print(movies.columns)

# Merge the movies and credits datasets on the 'id' column
movies = pd.merge(movies, credits, on='id', how='left')
print(movies.head(5))
print(movies.shape)
print(movies.columns)

# Load the ratings dataset
users = pd.read_csv('ratings.csv', on_bad_lines='skip')
print(users.head(5))

# Ensure the movieId column in users is of string type for consistency
users['movieId'] = users['movieId'].astype(str)

# Verify the alignment of unique values between the users and movies datasets
print("Unique movieId in users:")
print(users['movieId'].unique()[:10])

# Print the number of unique users in the ratings dataset
print(users['userId'].unique().shape)

# Filter out users who have rated more than 200 movies
user_rating_counts = users['userId'].value_counts()
frequent_users = user_rating_counts[user_rating_counts > 200].index
users = users[users['userId'].isin(frequent_users)]

# Display the first 5 rows of the filtered users dataset
print(users.head(5))
print(users.shape)

# Merge the filtered users dataset with the movies dataset on 'movieId'
user_movies = users.merge(movies, on="movieId", how='left')
print(user_movies.head(5))
print(user_movies.shape)

# Calculate the number of ratings each movie has received
num_rating = user_movies.groupby('title')['rating'].count().reset_index()
print(num_rating.head(5))
num_rating.rename(columns={"rating": "num_of_rating"}, inplace=True)
new_rating = user_movies.merge(num_rating, on='title')
print(new_rating.head(5))

# Filter out movies with fewer than 50 ratings 
new_rating = new_rating[new_rating['num_of_rating'] >= 50]
print(new_rating.head(5))
new_rating.drop_duplicates(['userId', 'title'], inplace=True)

# Create a pivot table of users and their movie ratings
user_pivot = new_rating.pivot(columns='userId', index='title', values='rating')
print(user_pivot)
user_pivot.fillna(0, inplace=True)

# Convert the pivot table to a sparse matrix format for efficient computation
user_sparse = csr_matrix(user_pivot)

# Calculate the cosine similarity between movies based on user ratings
cosine_sim = cosine_similarity(user_sparse)

# Save the cosine similarity matrix and pivot table for later use
with open('cosine_similarity_matrix.pkl', 'wb') as f:
    pickle.dump(cosine_sim, f)

with open('user_pivot.pkl', 'wb') as f:
    pickle.dump(user_pivot, f)

# Define a function to get movie recommendations based on the cosine similarity
def get_movie_recommendations(title, num_recommendations=5):
    if title not in user_pivot.index:
        return f"Movie title '{title}' not found in the dataset."
    
    movie_index = user_pivot.index.get_loc(title)
    similarity_scores = list(enumerate(cosine_sim[movie_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    recommended_titles = [user_pivot.index[i] for i, _ in similarity_scores if user_pivot.index[i] != title]
    return recommended_titles[:num_recommendations]

# Example usage: Get movie recommendations based on a specific title
movie_title = "The Lion King"  # replace with the actual movie title
recommendations = get_movie_recommendations(movie_title)
print(recommendations)

# Save the final movies dataset to a CSV file
movies.to_csv('movies.csv')
