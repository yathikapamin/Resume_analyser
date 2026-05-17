import streamlit as st
import pickle
import re
from sklearn.metrics.pairwise import cosine_similarity
skill_alias = {
    'nlp': 'natural language processing',
    'ml': 'machine learning',
    'dl': 'deep learning',
    'ai': 'artificial intelligence',
    'tf': 'tensorflow',
'cv': 'computer vision'
}
# -------------------------------
# Skill List (Cleaned)
# -------------------------------
skills_list = [
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

# -------------------------------
# Text Cleaning & Skill Extraction Function
# -------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)   # remove special chars
    text = re.sub(r'\s+', ' ', text)          # remove extra spaces
    return text

def extract_skills(text):
    text = text.lower()
    found_skills = set()

    # Convert short forms → full forms
    for short, full in skill_alias.items():
        if short in text:
            text += " " + full

    # Match skills
    for skill in skills_list:
        if skill in text:
            found_skills.add(skill)

    return list(found_skills)

# -------------------------------
# Load Model
# -------------------------------
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# -------------------------------
# UI Design & Styling
# -------------------------------
st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="wide")

st.markdown("""
<style>
    h1 {
        color: #00D2FF;
        text-align: center;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        background: linear-gradient(90deg, #00D2FF 0%, #3A7BD5 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-size: 18px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(0, 210, 255, 0.4);
    }
    .metric-card {
        background: #1E2129;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        text-align: center;
        border: 1px solid #333;
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        color: #00D2FF;
    }
    .metric-label {
        font-size: 14px;
        color: #A0AEC0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

st.title(" AI Resume Analyzer")
st.markdown("<p style='text-align: center; color: #A0AEC0; font-size: 18px; margin-bottom: 30px;'>Match your skills with the perfect job description using AI</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("###  Candidate Resume")
    resume = st.text_area("Paste the resume content here", height=300, placeholder="Experienced Data Scientist with 4 years...", label_visibility="collapsed")

with col2:
    st.markdown("###  Job Description")
    jd = st.text_area("Paste the job description here", height=300, placeholder="We are looking for a Data Scientist...", label_visibility="collapsed")

# -------------------------------
# Analyze Button
# -------------------------------
st.markdown("<br>", unsafe_allow_html=True)
analyze_btn = st.button(" Analyze Compatibility")

if analyze_btn:
    if resume.strip() == "" or jd.strip() == "":
        st.error(" Please enter both the Resume and the Job Description to proceed.")
    else:
        st.markdown("---")
        st.markdown("##  Analysis Results")
        
        # Predictions
        cleaned_resume = clean_text(resume)
        resume_vec = tfidf.transform([cleaned_resume])
        prediction = model.predict(resume_vec)[0]
        
        cleaned_jd = clean_text(jd)
        jd_vec = tfidf.transform([cleaned_jd])
        text_score = cosine_similarity(resume_vec, jd_vec)[0][0] * 100
        
        # Skill Extraction
        resume_skills_set = set(extract_skills(resume))
        jd_skills_set = set(extract_skills(jd))
        
        missing_skills = jd_skills_set - resume_skills_set
        matched_skills = jd_skills_set & resume_skills_set
        
        skill_score = (len(matched_skills) / len(jd_skills_set)) * 100 if len(jd_skills_set) > 0 else 0
        final_score = (text_score * 0.4) + (skill_score * 0.6)
        
        # Top Metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Predicted Role</div>
                <div class="metric-value" style="font-size: 24px; color: #E2E8F0;">{prediction}</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Final Match Score</div>
                <div class="metric-value" style="color: {'#38A169' if final_score > 60 else ('#DD6B20' if final_score > 40 else '#E53E3E')}">{round(final_score, 1)}%</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Skill Match</div>
                <div class="metric-value">{round(skill_score, 1)}%</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Text Similarity</div>
                <div class="metric-value">{round(text_score, 1)}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Detailed Analysis
        detail_col1, detail_col2 = st.columns(2)
        
        with detail_col1:
            st.success("### ✅ Matched Skills")
            if matched_skills:
                st.write(", ".join([f"**{s.title()}**" for s in matched_skills]))
            else:
                st.write("No specific skills matched.")
                
        with detail_col2:
            st.error("### ❌ Missing Skills")
            if missing_skills:
                st.write(", ".join([f"**{s.title()}**" for s in missing_skills]))
            else:
                st.write("Excellent! No major skills missing.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Final Decision
        if final_score > 60:
            st.success(" **Decision: Resume Accepted!** This candidate is a strong match for the role.")
        elif final_score > 40:
            st.warning(" **Decision: Needs Improvement.** The candidate has some potential but lacks several key skills.")
        else:
            st.error(" **Decision: Resume Rejected.** The candidate's profile does not align with the job requirements.")