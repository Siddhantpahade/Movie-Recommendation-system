import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def create_sim():
    try:
        data = pd.read_csv('data.csv')
        print("Data loaded successfully")
        if 'comb' not in data.columns:
            print("Error: 'comb' column not found in data")
            return None, None
        
        data['comb'] = data['comb'].fillna('')
        cv = CountVectorizer()
        count_matrix = cv.fit_transform(data['comb'])
        sim = cosine_similarity(count_matrix)
        print("Similarity matrix created successfully")
        return data, sim
    except Exception as e:
        print(f"Error in create_sim: {e}")
        return None, None

def rcmd(m):
    m = m.lower()
    try:
        data.head()
        sim.shape
    except NameError as ne:
        print(f"NameError: {ne}")
        data, sim = create_sim()
        if data is None or sim is None:
            return 'Data or similarity matrix could not be created.'
    
    print(f"Searching for movie: {m}")
    if m not in data['movie_title'].str.lower().unique():
        print(f"Movie '{m}' not found in database.")
        return 'This movie is not in our database. Please check if you spelled it correct.'
    else:
        i = data.loc[data['movie_title'].str.lower() == m].index[0]
        lst = list(enumerate(sim[i]))
        lst = sorted(lst, key=lambda x: x[1], reverse=True)
        lst = lst[1:11]
        l = [data['movie_title'][a] for a, _ in lst]
        print(f"Recommendations for '{m}': {l}")
        return l

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/recommend")
def recommend():
    movie = request.args.get('movie')
    r = rcmd(movie)
    movie = movie.upper()
    if type(r) == str:
        return render_template('recommend.html', movie=movie, r=r, t='s')
    else:
        return render_template('recommend.html', movie=movie, r=r, t='l')

if __name__ == '__main__':
    app.run()
