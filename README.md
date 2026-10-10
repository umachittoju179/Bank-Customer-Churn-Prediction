# Bank Customer Churn Prediction

## Project Overview

Bank Customer Churn Prediction is a machine learning project that analyzes customer information and predicts whether a customer is likely to leave a bank. It includes an interactive Streamlit dashboard for exploring churn patterns and generating predictions for individual customers.

## Objectives

- Analyze customer churn patterns.
- Explore customer attributes associated with churn.
- Train a machine learning classification model.
- Estimate churn probability using customer details.
- Present results through an interactive dashboard.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Joblib
- Streamlit
- Matplotlib and Seaborn

## Project Features

- Customer statistics and overall churn rate.
- Churn distribution visualization.
- Exploratory data analysis charts.
- Customer information input form.
- Churn prediction and probability estimate.
- Interactive dashboard.

## Project Structure

- `app.py` — Streamlit dashboard and prediction interface.
- `train_model.py` — Model training workflow.
- `generate_eda.py` — Exploratory data analysis charts.
- `churn_utils.py` — Utility module.
- `data/` — Dataset files.
- `outputs/` — Trained model and generated charts.
- `reports/` — Project reports.
- `requirements.txt` — Python dependencies.

## How to Run

1. Clone or download this repository.

2. Open a terminal in the project directory.

3. Create and activate a Python virtual environment.

4. Install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

5. If the trained model is missing, train it using the project's training script:

   ```bash
   python train_model.py
   ```

6. Start the dashboard:

   ```bash
   python -m streamlit run app.py
   ```

7. Open the local URL displayed in the terminal.

## How to Use

1. Open the dashboard.
2. Explore customer statistics and EDA charts.
3. Navigate to the customer prediction section.
4. Enter the customer's details.
5. Click **Predict Churn** to view the estimated churn probability and predicted outcome.

## Disclaimer

Predictions are statistical estimates and may be incorrect. They should not be used as the sole basis for customer-related decisions.

## Author

chittoju uma maheshwari
