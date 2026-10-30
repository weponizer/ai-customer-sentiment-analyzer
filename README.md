# 🤖 AI Customer Sentiment Analyzer

An AI-powered web application that analyzes customer reviews
and classifies them as positive or negative.

## ✨ Features

- Analyze individual customer reviews
- Analyze multiple reviews using CSV uploads
- Predict positive and negative sentiment
- Display confidence scores
- Visualize sentiment distribution
- Download analysis results as a CSV file

## 🛠️ Technologies Used

- Python
- Streamlit
- Hugging Face Transformers
- DistilBERT
- Pandas
- PyTorch

## ⚙️ Installation

1. Clone this repository:

   git clone https://github.com/weponizer/ai-customer-sentiment-analyzer.git

2. Open the project folder:

   cd ai-customer-sentiment-analyzer

3. Create and activate a virtual environment:

   python -m venv venv

   Windows PowerShell:

   .\venv\Scripts\Activate.ps1

4. Install the required packages:

   pip install streamlit transformers torch pandas

5. Run the application:

   streamlit run app.py

## 🧠 AI Model

This project uses DistilBERT fine-tuned for sentiment
classification to predict positive or negative sentiment.

## 👨‍💻 Author

GitHub: https://github.com/weponizer