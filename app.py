import pandas as pd
import numpy as np
import joblib
import sys
# IMPORTANT RUN THIS PY FILE USING Command python score.py <path_to_payment_default.csv> <path_to_payment_history.csv>
# eg-> python score.py payment_default.csv payment_history.csv
# and the result will be stored as prediction.csv file
def preprocess_and_score(payment_default_path, payment_history_path):
    payment_default_df = pd.read_csv(payment_default_path)
    payment_history_df = pd.read_csv(payment_history_path)
    payment_history_df = payment_history_df[payment_history_df['client_id'].isin(payment_default_df['client_id'])]
    pivoted_df = payment_history_df.pivot(index='client_id', columns='month', values=['payment_status', 'bill_amt', 'paid_amt'])
    pivoted_df.columns = [f"{feature}_month_{month}" for feature, month in pivoted_df.columns]
    pivoted_df = pivoted_df.reset_index()
    final_data = pd.merge(payment_default_df, pivoted_df, on='client_id', how='left')
    final_data.replace({'gender': {1: 'Male', 2: 'Female'}}, inplace=True)
    final_data.replace({'education': {
        1: 'Graduate_school', 2: 'University', 3: 'High School',
        0: 'others', 4: 'others', 5: 'others', 6: 'others'
    }}, inplace=True)
    final_data.replace({'marital_status': {
        1: 'Married', 2: 'Single', 0: 'others', 3: 'others'
    }}, inplace=True)
    df = pd.get_dummies(final_data, drop_first=False)
    df = df.astype(int)
    if 'client_id' in df.columns:
        client_ids = df['client_id']
        df = df.drop(columns=['client_id'])
    else:
        client_ids = None
    if 'default' in df.columns:
        X = df.drop(columns=['default'])
    else:
        X = df
    # Load the scaler and model
    scaler = joblib.load('scaler.pkl')
    X_scaled = scaler.transform(X)
    model = joblib.load('random_forest_model.pkl')
    predictions = model.predict(X_scaled)
    prediction_proba = model.predict_proba(X_scaled)
    output_df = pd.DataFrame({
        'client_id': client_ids if client_ids is not None else range(len(predictions)),
        'prediction': predictions,
        'probability_0': prediction_proba[:, 0],
        'probability_1': prediction_proba[:, 1]
    })

    return output_df

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python score.py payment_default.csv payment_history.csv")
        sys.exit(1)
    default_file = sys.argv[1]
    history_file = sys.argv[2]
    result = preprocess_and_score(default_file, history_file)
    result.to_csv("predictions.csv", index=False)
    print("Scoring complete. Output saved to predictions.csv")
    
# IMPORTANT RUN THIS PY FILE USING Command python score.py <path_to_payment_default.csv> <path_to_payment_history.csv>
# eg-> python score.py payment_default.csv payment_history.csv
# and the result will be stored as prediction.csv file
