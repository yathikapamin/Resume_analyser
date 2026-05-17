import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

print("Loading data...")
df = pd.read_csv('resume_data.csv')
df.rename(columns={'\ufeffjob_position_name': 'job_position_name'}, inplace=True)

# Create a combined 'Resume' text column from relevant columns
print("Combining text features...")
text_cols = ['career_objective', 'skills', 'degree_names', 'major_field_of_studies', 'positions', 'responsibilities']
df['Resume'] = df[text_cols].fillna('').apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

# Use job_position_name as the category
df['Category'] = df['job_position_name'].fillna('Unknown')

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)   # remove special chars
    text = re.sub(r'\s+', ' ', text)          # remove extra spaces
    return text

print("Cleaning text...")
df['clean_resume'] = df['Resume'].apply(clean_text)

print("Vectorizing...")
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['clean_resume']).toarray()

y = df['Category']

print("Training model...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy}")

print("Saving models...")
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
    
with open("tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)
    
print("Done!")
