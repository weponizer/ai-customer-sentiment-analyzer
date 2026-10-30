import streamlit as st
import pandas as pd
from transformers import pipeline

st.set_page_config(
    page_title="Sentiment AI | Customer Insights",
    page_icon="🤖",
    layout="wide"
)


st.markdown("""
<style>
    .stApp {
        background-color: #0b1120;
        color: #f1f5f9;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
    }

    h1, h2, h3 {
        color: #e2e8f0;
    }

    div[data-testid="stMetric"] {
        background-color: #172033;
        border: 1px solid #334155;
        padding: 20px;
        border-radius: 14px;
    }

    div.stButton > button {
        background-color: #6366f1;
        color: white;
        border: none;
        border-radius: 9px;
        padding: 10px 20px;
        font-weight: 600;
    }

    div.stButton > button:hover {
        background-color: #4f46e5;
        color: white;
        border: none;
    }

    div[data-testid="stFileUploader"] {
        background-color: #111827;
        border-radius: 12px;
        padding: 12px;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
</style>
""", unsafe_allow_html=True)


st.title("🤖 AI Customer Sentiment Analyzer")
st.caption("Turn customer feedback into actionable insights.")

@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    )

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar
st.sidebar.title("⚙️ Settings")
st.sidebar.write("AI-powered customer feedback analysis")
st.sidebar.info("Model: DistilBERT Sentiment Analysis")

# Input section
st.subheader("📝 Analyze a Customer Review")

review = st.text_area(
    "Enter customer feedback",
    placeholder="Example: Excellent product and fast delivery!",
    height=120
)

if st.button("🔍 Analyze Review", type="primary"):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        try:
            with st.spinner("Analyzing customer feedback..."):
                result = load_model()(review[:512])[0]

            label = result["label"]
            score = result["score"]

            st.session_state.history.append({
                "Review": review,
                "Sentiment": label.title(),
                "Confidence": round(score * 100, 2)
            })

            if label == "POSITIVE":
                st.success(f"😊 Positive Sentiment — Confidence: {score:.1%}")
            else:
                st.error(f"😞 Negative Sentiment — Confidence: {score:.1%}")

        except Exception as error:
            st.error("Analysis failed. Please check your connection.")
            st.exception(error)

# CSV upload section
# CSV upload section
st.divider()
st.subheader("📁 Analyze Multiple Reviews")

uploaded_file = st.file_uploader(
    "Upload a CSV file with a column named 'review'",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        if "review" not in data.columns:
            st.error("Your CSV must contain a column named 'review'.")

        elif st.button("🚀 Analyze CSV"):
            reviews = data["review"].dropna().astype(str).tolist()

            if not reviews:
                st.warning("No reviews found in the CSV.")
            else:
                with st.spinner(f"Analyzing {len(reviews)} reviews..."):
                    predictions = load_model()(
                        [text[:512] for text in reviews],
                        batch_size=8,
                        truncation=True
                    )

                results = pd.DataFrame({
                    "Review": reviews,
                    "Sentiment": [
                        item["label"].title() for item in predictions
                    ],
                    "Confidence (%)": [
                        round(item["score"] * 100, 2)
                        for item in predictions
                    ]
                })

                st.session_state.csv_results = results

        if "csv_results" in st.session_state:
            results = st.session_state.csv_results

            st.subheader("📊 Sentiment Analytics")

            total_reviews = len(results)
            positive_count = (results["Sentiment"] == "Positive").sum()
            negative_count = (results["Sentiment"] == "Negative").sum()

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Reviews", total_reviews)
            col2.metric("😊 Positive", positive_count)
            col3.metric("😞 Negative", negative_count)

            st.subheader("📈 Sentiment Distribution")
            st.bar_chart(results["Sentiment"].value_counts())

            st.metric(
                "⭐ Average Confidence",
                f"{results['Confidence (%)'].mean():.2f}%"
            )

            st.dataframe(results, use_container_width=True)

            st.download_button(
                "⬇️ Download Results",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="sentiment_analysis_results.csv",
                mime="text/csv"
            )

    except Exception as error:
        st.error(f"Could not process the CSV: {error}")

# Dashboard section
st.divider()
st.subheader("📊 Your Analysis Dashboard")

if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)

    col1, col2, col3 = st.columns(3)

    total = len(history_df)
    positive = (history_df["Sentiment"] == "Positive").sum()
    negative = (history_df["Sentiment"] == "Negative").sum()

    col1.metric("Total Reviews", total)
    col2.metric("😊 Positive Reviews", int(positive))
    col3.metric("😞 Negative Reviews", int(negative))

    st.bar_chart(
        history_df["Sentiment"].value_counts()
    )

    st.subheader("📋 Recent Reviews")
    st.dataframe(
        history_df.iloc[::-1],
        use_container_width=True
    )
else:
    st.info("Analyze your first review to populate the dashboard.")
