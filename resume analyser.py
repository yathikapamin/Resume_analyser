#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv("C:/Users/HP/Downloads/archive (1)/UpdatedResumeDataSet.csv")
print(df.head())
print(df.columns)


# In[3]:


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)   # remove special chars
    text = re.sub(r'\s+', ' ', text)          # remove extra spaces
    return text

df['clean_resume'] = df['Resume'].apply(clean_text)

print(df['clean_resume'][0])


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['clean_resume']).toarray()

y = df['Category']


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# In[ ]:


accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)


# In[ ]:


def predict_category(resume_text):
    cleaned = clean_text(resume_text)
    vector = tfidf.transform([cleaned]).toarray()
    prediction = model.predict(vector)
    return prediction[0]


# In[ ]:


sample_resume = """
Experienced in Python, machine learning, deep learning, and data analysis.
"""

print("Predicted Category:", predict_category(sample_resume))


# In[ ]:


important_skills = [
     # Programming Languages
    'python','java','c++','javascript','typescript','matlab','scala','go','ruby','php',

    # Data Science & ML
    'machine learning','deep learning','natural language processing','computer vision',
    'data analysis','data science','artificial intelligence',
    'regression','classification','clustering','nlp'

    # Libraries & Frameworks
    'pandas','numpy','scipy','scikit-learn','matplotlib','seaborn',
    'tensorflow','keras','pytorch','xgboost','lightgbm',
    'opencv','nltk','spacy','transformers',

    # Web Development
    'html','css','react','angular','vue','node js','django','flask','fastapi','bootstrap','jquery',

    # Databases
    'sql','mysql','postgresql','mongodb','oracle','sqlite','redis','cassandra',

    # Tools & Platforms
    'git','github','gitlab','docker','kubernetes','jenkins','linux','unix','bash',

    # Cloud
    'aws','azure','gcp','amazon s3','ec2','aws lambda',

    # Data Visualization
    'tableau','power bi','excel','data visualization','plotly','dash',

    # Big Data
    'hadoop','apache spark','hive','kafka','airflow',

    # Software Engineering
    'rest api','microservices','unit testing','debugging','deployment','optimization',

    # Others
    'internet of things','blockchain','cybersecurity','network security','automation'
]

def extract_skills(text):
    text = text.lower()
    found = []
    for skill in important_skills:
        if skill in text:
            found.append(skill)
    return found


# In[ ]:


print("Skills:", extract_skills(sample_resume))


# In[ ]:


resume = """
Python, SQL, Machine Learning, Data Analysis
"""

print("Category:", predict_category(resume))
print("Skills:", extract_skills(resume))


# In[ ]:


resume = """
Python, SQL, Machine Learning, Data Analysis
"""

# Step 1: Extract skills
skills = extract_skills(resume)

# Step 2: Define required skills
required_skills = ['python','sql','machine learning','tensorflow','nlp']

# Step 3: Find missing skills
missing = [s for s in required_skills if s not in skills]

# Step 4: Print result
print("\n===== RESULT =====")
print(f"Skills Found: {', '.join(skills)}")
print(f"Missing Skills: {', '.join(missing)}")


# In[ ]:


import pickle

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(tfidf, open("tfidf.pkl", "wb"))


# In[ ]:


pip install streamlit


# In[ ]:





# In[ ]:




