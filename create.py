import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Sample data loading (replace this with your actual data loading)
# data = pd.read_csv('your_data_file.csv')

# Example dataframe for demonstration
data = pd.DataFrame({'comb': ['movie one', 'movie two', None, 'movie four']})

# Check for NaN values and fill them with an empty string
data['comb'] = data['comb'].fillna('')

# Initialize CountVectorizer
cv = CountVectorizer()

# Fit and transform the data
count_matrix = cv.fit_transform(data['comb'])

## print(count_matrix)