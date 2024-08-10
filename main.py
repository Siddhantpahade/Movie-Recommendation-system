import openai
from dotenv import load_dotenv
import os
from flask import Flask, request, render_template, jsonify
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()  # This will load environment variables from the .env file

app = Flask(__name__)

# Global variables for data and similarity matrix
data = None
sim = None

def create_sim():
    global data, sim
    try:
        data = pd.read_csv('processed_data.csv')  # Load processed data
        print("Data loaded successfully")
        print(data.head())
        if 'comb' not in data.columns:
            raise ValueError("'comb' column not found in data")
        
        cv = CountVectorizer()
        count_matrix = cv.fit_transform(data['comb'])
        sim = cosine_similarity(count_matrix)
        print("Similarity matrix created successfully")
    except Exception as e:
        print(f"Error in create_sim: {e}")
        data, sim = None, None

# Create similarity matrix when the application starts
create_sim()

def format_title(title):
    return title.capitalize()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/recommend', methods=['GET'])
def recommend():
    movie = request.args.get('movie').strip().lower()
    
    if data is None or sim is None:
        return jsonify({'error': 'Data or similarity matrix is not loaded'}), 500

    data['movie_title_normalized'] = data['movie_title'].str.strip().str.lower()

    if movie not in data['movie_title_normalized'].unique():
        return render_template('recommend.html', movie=movie, r='This movie is not in our database. Please check if you spelled it correctly.', t='s')
    else:
        i = data.loc[data['movie_title_normalized'] == movie].index[0]
        lst = list(enumerate(sim[i]))
        lst = sorted(lst, key=lambda x: x[1], reverse=True)
        lst = lst[1:11]
        recommendations = list({format_title(data['movie_title'][a]) for a, _ in lst})
        return render_template('recommend.html', movie=movie, r=recommendations, t='l')

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
