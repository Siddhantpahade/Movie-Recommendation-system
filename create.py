import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

try:
    data = pd.read_csv('data.csv')  # Replace with the correct path to your raw data
    print("Data loaded successfully")
    
    # Ensure required columns exist
    required_columns = ['actor_1_name', 'actor_2_name', 'actor_3_name', 'director_name', 'genres', 'movie_title']
    for column in required_columns:
        if column not in data.columns:
            raise ValueError(f"'{column}' column not found in data")
    
    # Create 'comb' column by combining relevant text features
    data['comb'] = data['actor_1_name'] + ' ' + data['actor_2_name'] + ' ' + data['actor_3_name'] + ' ' + data['director_name'] + ' ' + data['genres'] + ' ' + data['movie_title']
    data['comb'] = data['comb'].fillna('')  # Fill NaN values with empty string
    
    # Save the processed data to a new CSV file
    data.to_csv('processed_data.csv', index=False)
    print("Processed data saved successfully with 'comb' column")

except Exception as e:
    print(f"Error in create.py: {e}")
