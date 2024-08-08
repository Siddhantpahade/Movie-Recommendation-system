from dotenv import load_dotenv
import os

load_dotenv()  # This will load environment variables from the .env file

from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

# Load data
try:
    data = pd.read_csv('data.csv')  # Replace with the correct path to your processed data
    print("Data loaded successfully")
    print(data.head())
except Exception as e:
    print(f"Error loading data: {e}")
    data = None

# Check if 'comb' column exists
if data is not None and 'comb' not in data.columns:
    raise ValueError("'comb' column not found in data")

@app.route('/recommend', methods=['GET'])
def recommend():
    movie = request.args.get('movie')
    
    if data is None:
        return jsonify({'error': 'Data is not loaded'}), 500

    # Recommendation logic (example)
    filtered_data = data[data['comb'] == movie]  # Adjust based on your logic

    return jsonify({'message': f'Recommendations for {movie}', 'data': filtered_data.to_dict(orient='records')})

if __name__ == '__main__':
    app.run(debug=True)
