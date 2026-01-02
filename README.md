# 📰 Fake News Detection using Machine Learning

Fake news has become a serious problem in today’s digital world, especially with the rapid spread of information through social media and online news platforms. This project aims to **automatically detect fake news articles** using **Machine Learning and Natural Language Processing (NLP)** techniques and provide real-time predictions through a **Streamlit web application**.

---

## 🎯 Project Objective

The objective of this project is to build an intelligent system that can:
- Analyze news content
- Learn patterns from historical real and fake news data
- Accurately classify news articles as **Real** or **Fake**
- Provide an easy-to-use web interface for users

---

## 🚀 Features

- ✅ Automated fake news classification
- ✅ Text preprocessing using NLP techniques
- ✅ Feature extraction using **TF-IDF**
- ✅ Machine learning models:
  - Decision Tree
  - Naive Bayes
- ✅ Detailed model evaluation and comparison
- ✅ ROC Curve and Cross-Validation analysis
- ✅ Model saving and reuse
- ✅ Interactive **Streamlit web application**
- ✅ Confidence score for predictions

---

## 🧠 Machine Learning Workflow

1. Dataset Loading (Real & Fake News)
2. Data Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Extraction using TF-IDF
5. Model Training
6. Model Evaluation & Comparison
7. Model Saving
8. Web Application Deployment

---

## 🛠️ Technologies Used

- **Programming Language:** Python  
- **Web Framework:** Streamlit  
- **Machine Learning:** scikit-learn  
- **NLP:** TF-IDF Vectorizer  
- **Data Analysis:** Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn  
- **Model Persistence:** joblib  

---

## 📁 Project Structure

```text
fake-news-detector/
│
├── app.py                       # Streamlit web app
├── fake_news_detection.ipynb    # Training & analysis notebook
├── decision_tree_model.pkl      # Saved Decision Tree model
├── naive_bayes_model.pkl        # Saved Naive Bayes model
├── tfidf_vectorizer.pkl         # Saved TF-IDF vectorizer
├── model_results.json           # Saved evaluation metrics
├── model_comparison.png         # Visualization output
├── eda_analysis.png             # EDA plots
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```
---

## 📷 Screenshots
<img width="1919" height="891" alt="Screenshot 2026-01-01 155631" src="https://github.com/user-attachments/assets/c7b55457-bcf9-4515-84a7-3fbcae56678e" />

<img width="1913" height="911" alt="Screenshot 2026-01-01 155745" src="https://github.com/user-attachments/assets/80711750-daba-4a6a-a7bf-96acaefad180" />


---

## ⚙️ Local Setup

### 1️⃣ Clone the Repository
```bash
https://github.com/Navya032006/Fake-News-Detector.git
cd fake-news-detector
```
### 2️⃣ Create a Virtual Environment (Optional but Recommended) 
```bash
python -m venv venv
source venv/bin/activate      # For Windows: venv\Scripts\activate
```
### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 4️⃣ Run the Streamlit Application
```bash
streamlit run app.py
```
---

### 🌐 Deployment:
https://navya032006-fake-news-detector-app-uahsmz.streamlit.app/
