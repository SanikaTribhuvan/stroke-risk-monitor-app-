Smart Stroke Risk Monitoring System

An AI-powered web application to predict stroke risk using a Decision Tree model (94.6% accuracy) and interactive visualizations. This project was developed as a field project for New Arts, Commerce and Science College, Ahmednagar, and presented at the ICADMA-2025 conference.


[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sanikatribhuvan-stroke-risk-monitor-app--app-33cixd.streamlit.app/)

## 🚀 Live Demo

**Try the live application here:** [https://sanikatribhuvan-stroke-risk-monitor-app--app-33cixd.streamlit.app/](https://sanikatribhuvan-stroke-risk-monitor-app--app-33cixd.streamlit.app/)

📸 Application Preview

Here is a look at the main application dashboard, showing the real-time risk calculator and interactive data visualizations.

[(This will link to the dashboard-preview.jpg screenshot you upload to GitHub)
](https://github.com/SanikaTribhuvan/stroke-risk-monitor-app-/blob/main/dashboard-preview.jpg)
🌟 Key Features

🧠 Real-Time AI Prediction: Uses a 94.6% accurate Decision Tree model to provide an instant stroke risk assessment (Low, Medium, High, Very High) based on user-input health metrics.

📊 Interactive Dashboard: A built-in dashboard, created with Plotly, that visualizes the key factors from our research, including:

The Wealth-Health Gap: The healthcare disparity in Maharashtra, comparing district income to hospital access.

Risk Factor Impact: How co-morbidities like hypertension and heart disease dramatically increase stroke rates.

🤖 AI Health Assistant: A text-to-speech assistant that provides clear, audible health advice based on the user's predicted risk level, improving accessibility.

🎓 Academic & Research-Backed: Built on a dataset of 5,110+ patient records enriched with demographic and economic data from public sources (Census of India, DMER, PubMed).

🛠️ Technology Stack

Web Framework: Streamlit

Data Analysis & ML: Python, Pandas, Scikit-learn

Data Visualization: Plotly

Deployment: Streamlit Community Cloud

Model: Decision Tree Classifier (.pkl)

Project Context

This application was developed as a 60-hour Field Project for the BCA Science program at New Arts, Commerce and Science College, Ahmednagar. The primary goal was to create a digital solution to improve healthcare access and early detection for stroke in Maharashtra, per NEP 2020 guidelines.

💻 How to Run Locally

Clone the repository:

git clone [YOUR-GITHUB-REPO-URL]
cd stroke-risk-monitor-app


Create and activate a virtual environment (Recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`


Install the required libraries:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run app.py
