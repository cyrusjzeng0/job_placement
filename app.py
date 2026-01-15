import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle

# Define your model class (MUST match training!)
class BinaryClassifier(nn.Module):
    def __init__(self, input_size=20, hidden_sizes=[64, 32], dropout=0.3):
        super().__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Load model and stats
@st.cache_resource
def load_model():
    model = BinaryClassifier(input_size=20, hidden_sizes=[64, 32], dropout=0.3)
    model.load_state_dict(torch.load('model_weights.pt', map_location=torch.device('cpu')))
    model.eval()
    return model

@st.cache_data
def load_feature_stats():
    with open('feature_stats.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()
feature_stats = load_feature_stats()

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Placement Prediction System")
st.markdown("Enter your academic and skill details to predict your placement probability")
st.info("💡 Select 'N/A' for any field you don't want to include in the prediction")

# ============================================================================
# INPUT COLLECTION WITH N/A OPTIONS
# ============================================================================

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Academic Information")
    
    # Age with N/A option
    age_na = st.checkbox("Age: N/A", key="age_na")
    if not age_na:
        age = st.slider("Age", min_value=18, max_value=30, value=22)
    else:
        age = None
    
    # Gender with N/A option
    gender_options = ["Male", "Female", "N/A"]
    gender = st.selectbox("Gender", options=gender_options)
    if gender == "N/A":
        gender_encoded = None
    else:
        gender_encoded = 1 if gender == "Male" else 0
    
    # CGPA with N/A option
    cgpa_na = st.checkbox("CGPA: N/A", key="cgpa_na")
    if not cgpa_na:
        cgpa = st.slider("CGPA", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
    else:
        cgpa = None
    
    # Degree with N/A option
    degree_options = ["B.Tech", "B.Sc", "BCA", "MCA", "N/A"]
    degree = st.selectbox("Degree", options=degree_options)
    
    # Branch with N/A option
    branch_options = ["CSE", "IT", "ECE", "Civil", "ME", "N/A"]
    branch = st.selectbox("Branch", options=branch_options)
    
    # Backlogs with N/A option
    backlogs_na = st.checkbox("Backlogs: N/A", key="backlogs_na")
    if not backlogs_na:
        backlogs = st.number_input("Number of Backlogs", min_value=0, max_value=10, value=0)
    else:
        backlogs = None

with col2:
    st.subheader("💼 Skills & Experience")
    
    # Internships with N/A option
    internships_na = st.checkbox("Internships: N/A", key="internships_na")
    if not internships_na:
        internships = st.number_input("Number of Internships", min_value=0, max_value=10, value=1)
    else:
        internships = None
    
    # Projects with N/A option
    projects_na = st.checkbox("Projects: N/A", key="projects_na")
    if not projects_na:
        projects = st.number_input("Number of Projects", min_value=0, max_value=20, value=2)
    else:
        projects = None
    
    # Coding Skills with N/A option
    coding_na = st.checkbox("Coding Skills: N/A", key="coding_na")
    if not coding_na:
        coding_skills = st.slider("Coding Skills (1-10)", min_value=1, max_value=10, value=5)
    else:
        coding_skills = None
    
    # Communication Skills with N/A option
    comm_na = st.checkbox("Communication Skills: N/A", key="comm_na")
    if not comm_na:
        communication_skills = st.slider("Communication Skills (1-10)", min_value=1, max_value=10, value=5)
    else:
        communication_skills = None
    
    # Soft Skills with N/A option
    soft_na = st.checkbox("Soft Skills: N/A", key="soft_na")
    if not soft_na:
        soft_skills_rating = st.slider("Soft Skills Rating (1-10)", min_value=1, max_value=10, value=5)
    else:
        soft_skills_rating = None
    
    # Aptitude with N/A option
    aptitude_na = st.checkbox("Aptitude Score: N/A", key="aptitude_na")
    if not aptitude_na:
        aptitude_test_score = st.slider("Aptitude Test Score", min_value=0, max_value=100, value=50)
    else:
        aptitude_test_score = None
    
    # Certifications with N/A option
    cert_na = st.checkbox("Certifications: N/A", key="cert_na")
    if not cert_na:
        certifications = st.number_input("Number of Certifications", min_value=0, max_value=20, value=1)
    else:
        certifications = None

# ============================================================================
# FEATURE ENGINEERING WITH IMPUTATION
# ============================================================================

def impute_features(raw_features, feature_stats):
    """
    Replace None values with mean/median from training data
    
    THEORY: Mean Imputation
    - When feature is missing, use "typical" value from training data
    - Doesn't bias predictions (neutral assumption)
    - Better than using 0 (which might be meaningful)
    
    Args:
        raw_features: Dict with possible None values
        feature_stats: Dict with mean/median values from training data
    
    Returns:
        features: Dict with all None replaced
        missing_features: List of features that were imputed (for display)
    """
    features = {}
    missing_features = []
    
    for key, value in raw_features.items():
        if value is None:
            features[key] = feature_stats[key]
            missing_features.append(key)
        else:
            features[key] = value
    
    return features, missing_features

# One-hot encode degree
if degree == "N/A":
    degree_encoding = {
        'Degree_B.Sc': feature_stats['Degree_B.Sc'],
        'Degree_B.Tech': feature_stats['Degree_B.Tech'],
        'Degree_BCA': feature_stats['Degree_BCA'],
        'Degree_MCA': feature_stats['Degree_MCA']
    }
else:
    degree_encoding = {
        'Degree_B.Sc': 1 if degree == "B.Sc" else 0,
        'Degree_B.Tech': 1 if degree == "B.Tech" else 0,
        'Degree_BCA': 1 if degree == "BCA" else 0,
        'Degree_MCA': 1 if degree == "MCA" else 0
    }

# One-hot encode branch
if branch == "N/A":
    branch_encoding = {
        'Branch_CSE': feature_stats['Branch_CSE'],
        'Branch_Civil': feature_stats['Branch_Civil'],
        'Branch_ECE': feature_stats['Branch_ECE'],
        'Branch_IT': feature_stats['Branch_IT'],
        'Branch_ME': feature_stats['Branch_ME']
    }
else:
    branch_encoding = {
        'Branch_CSE': 1 if branch == "CSE" else 0,
        'Branch_Civil': 1 if branch == "Civil" else 0,
        'Branch_ECE': 1 if branch == "ECE" else 0,
        'Branch_IT': 1 if branch == "IT" else 0,
        'Branch_ME': 1 if branch == "ME" else 0
    }

# Create raw features dictionary
raw_features = {
    'Age': age,
    'Gender': gender_encoded,
    'CGPA': cgpa,
    'Internships': internships,
    'Projects': projects,
    'Coding_Skills': coding_skills,
    'Communication_Skills': communication_skills,
    'Aptitude_Test_Score': aptitude_test_score,
    'Soft_Skills_Rating': soft_skills_rating,
    'Certifications': certifications,
    'Backlogs': backlogs,
    **degree_encoding,
    **branch_encoding
}

# Impute missing values
features, missing_features = impute_features(raw_features, feature_stats)

# Convert to DataFrame
input_df = pd.DataFrame([features])

# ============================================================================
# DISPLAY INPUT SUMMARY
# ============================================================================

st.subheader("📋 Your Input Summary")

# Show which features were imputed
if missing_features:
    st.warning(f"⚠️ The following features were set to N/A and will use typical values: {', '.join(missing_features)}")

st.dataframe(input_df, width="stretch")

# ============================================================================
# PREDICTION
# ============================================================================

if st.button("🔮 Predict Placement Probability", type="primary"):
    st.markdown("---")
    
    # Convert to tensor and predict
    X = torch.tensor(input_df.values, dtype=torch.float32)
    
    with torch.no_grad():
        logits = model(X)
        prediction_proba = torch.sigmoid(logits).item()
    
    # Display result
    st.subheader("🎯 Prediction Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Placement Probability", f"{prediction_proba*100:.1f}%")
    
    with col2:
        status = "Likely Placed ✅" if prediction_proba >= 0.5 else "Needs Improvement ⚠️"
        st.metric("Status", status)
    
    with col3:
        confidence = "High" if abs(prediction_proba - 0.5) > 0.3 else "Medium"
        st.metric("Confidence", confidence)
    
    # Progress bar
    st.progress(prediction_proba)
    
    # Note about missing features
    if missing_features:
        st.info(f"ℹ️ Note: This prediction used typical values for {len(missing_features)} missing features. Providing actual values may improve accuracy.")
    
    # Recommendations
    st.subheader("💡 Recommendations")
    
    if prediction_proba < 0.5:
        recommendations = []
        if features['CGPA'] < 7.0:
            recommendations.append("📚 Focus on improving your CGPA")
        if features['Internships'] < 2:
            recommendations.append("💼 Gain more internship experience")
        if features['Projects'] < 3:
            recommendations.append("🔨 Work on more projects to build your portfolio")
        if features['Coding_Skills'] < 6:
            recommendations.append("💻 Enhance your coding skills through practice")
        if features['Communication_Skills'] < 6:
            recommendations.append("🗣️ Improve communication skills through workshops")
        if features['Backlogs'] > 0:
            recommendations.append("⚠️ Clear your backlogs as soon as possible")
        
        for rec in recommendations:
            st.write(f"- {rec}")
    else:
        st.success("Great profile! Keep up the excellent work and continue building your skills.")

# Footer
st.markdown("---")
st.markdown("**Note:** This is a prediction model. Actual placement depends on various factors including company requirements and interview performance.")