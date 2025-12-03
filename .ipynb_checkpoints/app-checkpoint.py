import streamlit as st
import joblib
import json
from PIL import Image
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="📰 Fake News Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Global Styles */
* {
    font-family: 'Inter', sans-serif;
}

/* Hide Streamlit Branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display:none;}

/* Main Background */
.stApp {
    background: linear-gradient(to bottom, #f8f9fa 0%, #e9ecef 100%);
}

/* Header Styles */
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 15px;
    margin-bottom: 2rem;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.main-title {
    color: white;
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}

.subtitle {
    color: rgba(255, 255, 255, 0.9);
    font-size: 1.1rem;
    font-weight: 400;
}

/* Sidebar Styles */
.css-1d391kg, .css-1lcbmhc {
    background-color: #f8f9fa;
}

/* Card Styles */
.info-card {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    border-left: 4px solid #667eea;
    margin-bottom: 1rem;
}

.metric-card {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    transition: transform 0.2s;
}

.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.12);
}

.metric-value {
    font-size: 2.5rem;
    font-weight: 700;
    color: #667eea;
    margin: 0.5rem 0;
}

.metric-label {
    font-size: 0.9rem;
    color: #6c757d;
    font-weight: 500;
}

/* Button Styles */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.75rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    width: 100%;
    transition: all 0.3s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

/* Sample Button Styles */
.sample-button {
    background: white;
    border: 2px solid #e9ecef;
    border-radius: 8px;
    padding: 0.6rem 1.2rem;
    font-weight: 500;
    transition: all 0.2s;
}

/* Text Input Styles */
.stTextInput input, .stTextArea textarea {
    border-radius: 8px;
    border: 2px solid #e9ecef;
    padding: 0.75rem;
    font-size: 1rem;
}

.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

/* Selectbox Styles */
.stSelectbox > div > div {
    border-radius: 8px;
    border: 2px solid #e9ecef;
}

/* Tab Styles */
.stTabs [data-baseweb="tab-list"] {
    gap: 2rem;
    background-color: transparent;
    border-bottom: 2px solid #e9ecef;
}

.stTabs [data-baseweb="tab"] {
    padding: 1rem 1.5rem;
    font-weight: 600;
    color: #6c757d;
    border-radius: 8px 8px 0 0;
}

.stTabs [aria-selected="true"] {
    background-color: white;
    color: #667eea;
    border-bottom: 3px solid #667eea;
}

/* Result Card Styles */
.prediction-card {
    border-radius: 15px;
    padding: 2rem;
    margin: 2rem 0;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.fake-prediction {
    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
    border: 3px solid #ff5252;
}

.real-prediction {
    background: linear-gradient(135deg, #51cf66 0%, #37b24d 100%);
    border: 3px solid #40c057;
}

.prediction-header {
    color: white;
    font-size: 2rem;
    font-weight: 700;
    text-align: center;
    margin-bottom: 1rem;
}

.confidence-score {
    background: rgba(255, 255, 255, 0.2);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    color: white;
    font-size: 1.8rem;
    font-weight: 700;
    margin: 1rem 0;
}

.prediction-details {
    background: rgba(255, 255, 255, 0.15);
    border-radius: 10px;
    padding: 1rem;
    color: white;
    margin-top: 1rem;
}

/* Section Headers */
.section-header {
    font-size: 1.5rem;
    font-weight: 700;
    color: #212529;
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 3px solid #667eea;
}

/* Loading Animation */
.stSpinner > div {
    border-top-color: #667eea !important;
}

/* Info Box */
.info-box {
    background: #e7f5ff;
    border-left: 4px solid #339af0;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
    color: #1c7ed6;
}

/* Warning Box */
.warning-box {
    background: #fff3bf;
    border-left: 4px solid #fab005;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
    color: #e67700;
}
</style>
""", unsafe_allow_html=True)

# Load models, results, and dataset
@st.cache_resource
def load_models():
    try:
        dt_model = joblib.load("decision_tree_model.pkl")
        nb_model = joblib.load("naive_bayes_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        
        with open("model_results.json", "r") as f:
            results = json.load(f)
        
        return dt_model, nb_model, vectorizer, results
    except Exception as e:
        st.error(f"⚠️ Error loading models: {str(e)}")
        st.info("💡 Please run the training notebook first to generate the model files.")
        return None, None, None, None

@st.cache_data
def load_dataset():
    """Load the news dataset for sampling"""
    try:
        # Try to load the combined dataset
        df = pd.read_csv("news.csv")
        return df
    except:
        try:
            # Try loading separate files
            fake_df = pd.read_csv("Fake.csv")
            real_df = pd.read_csv("True.csv")
            
            fake_df['label'] = 0
            real_df['label'] = 1
            
            df = pd.concat([fake_df, real_df], ignore_index=True)
            return df
        except:
            # Return None if dataset not found
            return None

dt_model, nb_model, vectorizer, results = load_models()
dataset = load_dataset()

def get_random_news(label):
    """Get a random news article from dataset"""
    if dataset is None:
        # Fallback to hardcoded samples if dataset not available
        if label == 1:  # Real news
            return {
                "title": "New COVID-19 Vaccine Shows Promise in Clinical Trials",
                "content": "A new COVID-19 vaccine developed by researchers at a major pharmaceutical company has shown promising results in Phase 3 clinical trials. The vaccine demonstrated 95% efficacy in preventing symptomatic COVID-19 infections. The study included over 30,000 participants across multiple countries. Health authorities are currently reviewing the data for potential emergency use authorization."
            }
        else:  # Fake news
            return {
                "title": "Scientists Discover Miracle Cure That Eliminates All Diseases Instantly",
                "content": "In a shocking breakthrough that doctors don't want you to know about, scientists have discovered a miracle cure made from common household items that can eliminate all diseases instantly. This revolutionary treatment costs nothing and Big Pharma is trying to hide it from the public. Thousands of people have already been cured using this one weird trick."
            }
    
    # Get random article from dataset
    filtered_df = dataset[dataset['label'] == label]
    
    if len(filtered_df) == 0:
        return {"title": "No sample available", "content": ""}
    
    random_article = filtered_df.sample(n=1).iloc[0]
    
    # Extract title and content
    if 'title' in random_article and 'text' in random_article:
        return {
            "title": str(random_article['title']),
            "content": str(random_article['text'])
        }
    elif 'text' in random_article:
        # If no separate title, use first sentence as title
        text = str(random_article['text'])
        sentences = text.split('.')
        return {
            "title": sentences[0] + '.' if len(sentences) > 0 else "Sample News",
            "content": text
        }
    else:
        return {"title": "Sample News", "content": str(random_article.get('content', ''))}

# Header
st.markdown("""
<div class="main-header">
    <div class="main-title">
        📰 Fake News Detector
    </div>
    <div class="subtitle">
        Powered by Decision Tree & Naive Bayes ML Models
    </div>
</div>
""", unsafe_allow_html=True)

# Check if models are loaded
if dt_model is None:
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    st.markdown("**Select Model**")
    model_choice = st.selectbox(
        "Choose classifier:",
        ["Naive Bayes", "Decision Tree", "Both (Ensemble)"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    st.markdown("### 📊 Model Performance")
    
    # Decision Tree Accuracy
    st.markdown(f"""
    <div class="info-card">
        <div class="metric-label">Decision Tree Accuracy</div>
        <div class="metric-value">{results['decision_tree']['accuracy']*100:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Naive Bayes Accuracy
    st.markdown(f"""
    <div class="info-card">
        <div class="metric-label">Naive Bayes Accuracy</div>
        <div class="metric-value">{results['naive_bayes']['accuracy']*100:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📈 Dataset Info")
    st.markdown("""
    <div class="info-card">
        <p><strong>Total Articles:</strong> 44898</p>
        <p><strong>Real News:</strong> 21417</p>
        <p><strong>Fake News:</strong> 23481</p>
    </div>
    """, unsafe_allow_html=True)

# Main content with tabs
tab1, tab2, tab3 = st.tabs(["🔍 Detect News", "📊 Data Analysis", "🎯 Model Insights"])

with tab1:
    st.markdown('<div class="section-header">Enter News Article</div>', unsafe_allow_html=True)
    
    # Sample buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("📗 Load Real News Sample", use_container_width=True):
            sample = get_random_news(label=1)
            st.session_state['title'] = sample['title']
            st.session_state['content'] = sample['content']
            st.rerun()
    
    with col2:
        if st.button("📕 Load Fake News Sample", use_container_width=True):
            sample = get_random_news(label=0)
            st.session_state['title'] = sample['title']
            st.session_state['content'] = sample['content']
            st.rerun()
    
    with col3:
        if st.button("🔄 Clear", use_container_width=True):
            st.session_state['title'] = ""
            st.session_state['content'] = ""
            st.rerun()
    
    # Input fields
    st.markdown("##### 📰 News Title")
    title = st.text_input(
        "Title",
        value=st.session_state.get('title', ''),
        placeholder="Enter the headline...",
        label_visibility="collapsed"
    )
    
    st.markdown("##### 📝 News Content")
    content = st.text_area(
        "Content",
        value=st.session_state.get('content', ''),
        placeholder="Paste the full article text here...",
        height=200,
        label_visibility="collapsed"
    )
    
    # Prediction section
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("### 🎯 Prediction")
        predict_button = st.button("🔍 Analyze & Predict", use_container_width=True, type="primary")
    
    if predict_button:
        if title.strip() or content.strip():
            with st.spinner("🔄 Analyzing news content..."):
                # Combine title and content
                full_text = f"{title} {content}"
                
                # Preprocess
                processed_text = full_text.lower()
                processed_text = ''.join(char for char in processed_text if char.isalnum() or char.isspace())
                
                # Vectorize
                text_vectorized = vectorizer.transform([processed_text])
                
                # Get predictions based on model choice
                if model_choice == "Decision Tree":
                    prediction = dt_model.predict(text_vectorized)[0]
                    probability = dt_model.predict_proba(text_vectorized)[0]
                    model_used = "Decision Tree"
                    
                elif model_choice == "Naive Bayes":
                    prediction = nb_model.predict(text_vectorized)[0]
                    probability = nb_model.predict_proba(text_vectorized)[0]
                    model_used = "Naive Bayes"
                    
                else:  # Ensemble
                    dt_prob = dt_model.predict_proba(text_vectorized)[0]
                    nb_prob = nb_model.predict_proba(text_vectorized)[0]
                    probability = (dt_prob + nb_prob) / 2
                    prediction = 1 if probability[1] > 0.5 else 0
                    model_used = "Ensemble (DT + NB)"
                
                confidence = max(probability) * 100
                
                # Display results
                st.markdown("---")
                
                if prediction == 0:  # Fake News
                    st.markdown(f"""
                    <div class="prediction-card fake-prediction">
                        <div class="prediction-header">⚠️ FAKE NEWS DETECTED</div>
                        <div class="confidence-score">
                            Confidence: {confidence:.2f}%
                        </div>
                        <div class="prediction-details">
                            <p><strong>🤖 Model Used:</strong> {model_used}</p>
                            <p><strong>📊 Fake Probability:</strong> {probability[0]*100:.2f}%</p>
                            <p><strong>📊 Real Probability:</strong> {probability[1]*100:.2f}%</p>
                            <hr style="border-color: rgba(255,255,255,0.3); margin: 1rem 0;">
                            <p><strong>⚡ Analysis:</strong> This content shows characteristics commonly found in fake or misleading news articles.</p>
                            <p><strong>💡 Recommendation:</strong> Verify this information from trusted sources before sharing.</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:  # Real News
                    st.markdown(f"""
                    <div class="prediction-card real-prediction">
                        <div class="prediction-header">✅ REAL NEWS</div>
                        <div class="confidence-score">
                            Confidence: {confidence:.2f}%
                        </div>
                        <div class="prediction-details">
                            <p><strong>🤖 Model Used:</strong> {model_used}</p>
                            <p><strong>📊 Real Probability:</strong> {probability[1]*100:.2f}%</p>
                            <p><strong>📊 Fake Probability:</strong> {probability[0]*100:.2f}%</p>
                            <hr style="border-color: rgba(255,255,255,0.3); margin: 1rem 0;">
                            <p><strong>⚡ Analysis:</strong> This content appears to be authentic and credible based on our analysis.</p>
                            <p><strong>💡 Note:</strong> While our model indicates this is likely real news, always cross-reference important information.</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Probability breakdown
                st.markdown("### 📊 Detailed Probability Breakdown")
                
                prob_col1, prob_col2 = st.columns(2)
                
                with prob_col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">🔴 Fake News Probability</div>
                        <div class="metric-value">{probability[0]*100:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with prob_col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">🟢 Real News Probability</div>
                        <div class="metric-value">{probability[1]*100:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
        else:
            st.warning("⚠️ Please enter news title or content to analyze.")

with tab2:
    st.markdown('<div class="section-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    try:
        st.image("eda_analysis.png", use_container_width=True)
        
        st.markdown("""
        <div class="info-box">
            <strong>📊 Key Insights from EDA:</strong>
            <ul>
                <li>Dataset contains balanced classes of real and fake news</li>
                <li>Fake news tends to have different word count patterns</li>
                <li>Character distribution varies between real and fake articles</li>
                <li>Average word length shows distinct patterns</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    except:
        st.info("📊 Run the training notebook to generate EDA visualizations")

with tab3:
    st.markdown('<div class="section-header">Model Performance Comparison</div>', unsafe_allow_html=True)
    
    # Performance metrics comparison
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    with metrics_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🎯 Best Accuracy</div>
            <div class="metric-value">{max(results['decision_tree']['accuracy'], results['naive_bayes']['accuracy'])*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🎪 Best Precision</div>
            <div class="metric-value">{max(results['decision_tree']['precision'], results['naive_bayes']['precision'])*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🔄 Best Recall</div>
            <div class="metric-value">{max(results['decision_tree']['recall'], results['naive_bayes']['recall'])*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">⭐ Best F1-Score</div>
            <div class="metric-value">{max(results['decision_tree']['f1_score'], results['naive_bayes']['f1_score']):.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model comparison visualization
    try:
        st.image("model_comparison.png", use_container_width=True)
        
        st.markdown(f"""
        <div class="info-box">
            <strong>🏆 Winner: {results['best_model']}</strong><br>
            Based on comprehensive evaluation across multiple metrics including accuracy, precision, recall, F1-score, and ROC-AUC.
        </div>
        """, unsafe_allow_html=True)
    except:
        st.info("📊 Run the training notebook to generate comparison visualizations")
    
    # Detailed comparison table
    st.markdown("### 📋 Detailed Metrics Comparison")
    
    comparison_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'CV Mean'],
        'Decision Tree': [
            f"{results['decision_tree']['accuracy']*100:.2f}%",
            f"{results['decision_tree']['precision']*100:.2f}%",
            f"{results['decision_tree']['recall']*100:.2f}%",
            f"{results['decision_tree']['f1_score']:.4f}",
            f"{results['decision_tree']['roc_auc']:.4f}",
            f"{results['decision_tree']['cv_mean']*100:.2f}%"
        ],
        'Naive Bayes': [
            f"{results['naive_bayes']['accuracy']*100:.2f}%",
            f"{results['naive_bayes']['precision']*100:.2f}%",
            f"{results['naive_bayes']['recall']*100:.2f}%",
            f"{results['naive_bayes']['f1_score']:.4f}",
            f"{results['naive_bayes']['roc_auc']:.4f}",
            f"{results['naive_bayes']['cv_mean']*100:.2f}%"
        ]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown("""
<div class="warning-box">
    <strong>⚠️ Disclaimer:</strong> This is an AI-based educational tool and may not be 100% accurate. 
    Always verify important information from multiple trusted sources before making decisions or sharing content.
</div>
""", unsafe_allow_html=True)