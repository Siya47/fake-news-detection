import streamlit as st
import joblib
import pandas as pd
from datetime import datetime

# ==========================
# Page Configuration
# ==========================
st.set_page_config(
    page_title="AI-Powered News Verification System",
    page_icon="📰",
    layout="wide"
)

# ==========================
# Load Model & Vectorizer
# ==========================
model = joblib.load("model/random_forest_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

# ==========================
# Session State for History
# ==========================
if "history" not in st.session_state:
    st.session_state.history = []

# ==========================
# Sidebar
# ==========================
st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "News Verification",
        "Dataset Insights",
        "About Model"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **AI-Powered News Verification System**

    **Dataset:** Fake.csv + True.csv

    **Total Articles:** 44,898

    **NLP Technique:** TF-IDF

    **Machine Learning Model:** Random Forest
    """
)

# ==================================================
# PAGE 1: NEWS VERIFICATION
# ==================================================
if page == "News Verification":

    # Main Title
    
    st.title("📰 AI-Powered News Verification System")
    
    st.markdown(
        """
        Detect fake and real news articles using 
        **Natural Language Processing (NLP)**
        and **Machine Learning**.
        """
    )

    # KPI Metrics
    
    col1, col2, col3 = st.columns(3)
    
    col1.metric("Dataset Size", "44,898")
    col2.metric("Fake Articles", "23,481")
    col3.metric("Real Articles", "21,417")
    
    st.markdown("---")
    
    # User Input
    
    news_text = st.text_area(
        "Paste News Article",
        height=250
    )
    
    st.caption(
        "Example: Paste a complete news article or headline for analysis."
    )
    
    # Prediction Button
    
    if st.button("Analyze News"):
        
        if not news_text.strip():
            st.warning("Please enter a news article.")
            st.stop()
            
        # Transform Text
        transformed_text = vectorizer.transform([news_text])
        
        # Prediction
        prediction = model.predict(transformed_text)
        
        # Probability
        probability = model.predict_proba(transformed_text)
        
        fake_prob = probability[0][0] * 100
        real_prob = probability[0][1] * 100
        
        confidence = probability.max() * 100
        
        # Prediction Breakdown
        st.subheader("Prediction Breakdown")
        
        col1, col2 = st.columns(2)
        
        col1.metric(
            "Fake Probability", 
            f"{fake_prob:.2f}%"
        )
        
        col2.metric(
            "Real Probability", 
            f"{real_prob:.2f}%"
        )
        
        st.markdown("### Result")
        
        if prediction[0] == 0:
            st.error("Prediction: Fake News")
            result = "Fake News"
        else:
            st.success("Prediction: Real News")
            result = "Real News"
            
        # Confidence Score
        st.subheader("Prediction Confidence")
        
        st.progress(int(confidence))
        
        st.info(
            f"Model Confidence: {confidence:.2f}%"
        )
        
        # Store Prediction History
        st.session_state.history.append(
            {
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Prediction": result,
                "Confidence": f"{confidence:.2f}%"
            }
        )
        
    # Prediction History
    if st.session_state.history:
        
        st.markdown("---")
        st.subheader("Prediction History")
        
        history_df = pd.DataFrame(
            st.session_state.history
        )
        
        st.dataframe(
            history_df,
            use_container_width=True
        )
        
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

# ==================================================
# PAGE 2: DATASET INSIGHTS
# ==================================================
elif page == "Dataset Insights":

    st.title("📊 Dataset Insights")

    st.markdown(
        """
        Overview of the dataset used for
        Fake News Detection.
        """
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Articles",
        "44,898"
    )

    col2.metric(
        "Fake News",
        "23,481"
    )

    col3.metric(
        "Real News",
        "21,417"
    )

    st.markdown("---")

    st.subheader("News Distribution")

    chart_data = pd.DataFrame(
        {
            "Category": ["Fake News", "Real News"],
            "Count": [23481, 21417]
        }
    )

    st.bar_chart(
        chart_data.set_index("Category")
    )

    st.subheader("Dataset Summary")

    st.dataframe(
        chart_data,
        use_container_width=True
    )

# ==================================================
# PAGE 3: ABOUT MODEL
# ==================================================
elif page == "About Model":

    st.title("🤖 About Model")

    st.write(
        """
        ### Project Objective
        
        Detect whether a news article is 
        Fake or Real using Natural Language 
        Processing and Machine Learning.
        """
    )

    st.markdown("---")

    st.subheader("Techniques Used")

    st.markdown(
        """
        - Data Cleaning
        - Text Preprocessing
        - TF-IDF Vectorization
        - Random Forest Classification
        - Streamlit Dashboard
        """
    )

    st.markdown("---")

    st.subheader("Dataset Information")

    st.markdown(
        """
        - Total Articles: 44,898
        - Fake News: 23,481
        - Real News: 21,417
        """
    )

    st.markdown("---")

    st.subheader("Project Highlights")

    st.markdown(
        """
        ✅ 44,898 News Articles

        ✅ NLP-Based Text Classification

        ✅ TF-IDF Feature Engineering

        ✅ Random Forest Model

        ✅ Interactive Streamlit Dashboard

        ✅ Prediction History Tracking
        """
    )


# ==========================
# Footer
# ==========================
st.markdown("---")

st.caption(
    "Built with Python • Scikit-Learn • NLP • Random Forest • Streamlit"
)