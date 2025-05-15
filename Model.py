import joblib  # For loading the trained RF model
import pandas as pd
import numpy as np

# Load the trained Random Forest model
rf_final_model = joblib.load('/content/drive/MyDrive/IT1244 Project/Models/rf_final_model.pkl')

def preprocess_new_data(new_client_path, new_invoice_path):
    """
    Preprocess new client and invoice data to match the input format used during training.

    Args:
    - new_client_path: Path to the new client data CSV.
    - new_invoice_path: Path to the new invoice data CSV.

    Returns:
    - Tuple containing 'id' and preprocessed feature matrix (X) ready for prediction.
    """
    print("Loading and preprocessing new data...")

    # Load the new data files
    new_client_df = pd.read_csv(new_client_path)
    new_invoice_df = pd.read_csv(new_invoice_path)

    # Convert 'date' columns to datetime format
    new_client_df['date'] = pd.to_datetime(new_client_df['date'], format='%d/%m/%Y')
    new_invoice_df['date'] = pd.to_datetime(new_invoice_df['date'], format='%d/%m/%Y')

    # Handle missing values by forward fill for client data and zero fill for invoice data
    new_client_df = new_client_df.fillna(method='ffill')
    new_invoice_df = new_invoice_df.fillna(0)

    # Aggregate invoice data by 'id'
    print("Aggregating new invoice data...")
    invoice_agg = new_invoice_df.groupby('id').agg(
        consommation_level_1_sum=('consommation_level_1', 'sum'),
        consommation_level_1_mean=('consommation_level_1', 'mean'),
        consommation_level_1_std=('consommation_level_1', 'std'),
        consommation_level_1_max=('consommation_level_1', 'max'),
        consommation_level_1_min=('consommation_level_1', 'min'),

        consommation_level_2_sum=('consommation_level_2', 'sum'),
        consommation_level_2_mean=('consommation_level_2', 'mean'),
        consommation_level_2_std=('consommation_level_2', 'std'),
        consommation_level_2_max=('consommation_level_2', 'max'),
        consommation_level_2_min=('consommation_level_2', 'min'),

        consommation_level_3_sum=('consommation_level_3', 'sum'),
        consommation_level_3_mean=('consommation_level_3', 'mean'),
        consommation_level_3_std=('consommation_level_3', 'std'),
        consommation_level_3_max=('consommation_level_3', 'max'),
        consommation_level_3_min=('consommation_level_3', 'min'),

        consommation_level_4_sum=('consommation_level_4', 'sum'),
        consommation_level_4_mean=('consommation_level_4', 'mean'),
        consommation_level_4_std=('consommation_level_4', 'std'),
        consommation_level_4_max=('consommation_level_4', 'max'),
        consommation_level_4_min=('consommation_level_4', 'min'),

        reading_remarque_mean=('reading_remarque', 'mean'),
        counter_statue_avg=('counter_statue', 'mean'),
        counter_statue_mode=('counter_statue', lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan),

        total_transactions=('date', 'count'),
        avg_transactions_per_month=('date', lambda x: x.nunique() /
                                    ((x.max() - x.min()).days / 30 if (x.max() - x.min()).days != 0 else 1)),
        transaction_period_length_days=('date', lambda x: (x.max() - x.min()).days)
    ).reset_index()

    # Calculate modes and unique counts for categorical variables
    print("Calculating modes and unique counts for categorical variables...")
    mode_df = new_invoice_df.groupby('id').agg({
        'tarif_type': lambda x: x.mode()[0] if not x.mode().empty else np.nan,
        'counter_type': lambda x: x.mode()[0] if not x.mode().empty else np.nan,
        'months_number': lambda x: x.mode()[0] if not x.mode().empty else np.nan
    }).reset_index()

    unique_count_df = new_invoice_df.groupby('id').agg({
        'tarif_type': 'nunique',
        'counter_type': 'nunique',
        'months_number': 'nunique'
    }).reset_index()

    # Rename columns for clarity
    mode_df.columns = ['id', 'tarif_type_mode', 'counter_type_mode', 'months_number_mode']
    unique_count_df.columns = ['id', 'tarif_type_unique_count', 'counter_type_unique_count', 'months_number_unique_count']

    # Merge all aggregated data
    print("Merging aggregated data...")
    final_agg_df = invoice_agg.merge(mode_df, on='id').merge(unique_count_df, on='id')

    # Merge with new client data
    merged_new_data = pd.merge(new_client_df, final_agg_df, on='id', how='inner')

    # One-hot encode categorical columns
    categorical_cols = ['dis', 'catg', 'region', 'months_number_mode', 'counter_type_mode']
    merged_new_data = pd.get_dummies(merged_new_data, columns=categorical_cols, drop_first=True)

    # Extract 'id' column for tracking predictions
    ids = merged_new_data['id']

    # Drop unnecessary columns to prepare the feature matrix
    X_new = merged_new_data.drop(['id', 'date'], axis=1, errors='ignore')

    print("Preprocessing complete. Data is ready for prediction.")
    return ids, X_new

# Preprocess the new data for prediction
new_client_path = '/content/drive/MyDrive/IT1244 Project/Dataset/new_client.csv'
new_invoice_path = '/content/drive/MyDrive/IT1244 Project/Dataset/new_invoice.csv'
ids, X_new = preprocess_new_data(new_client_path, new_invoice_path)

# Make predictions using the trained Random Forest model
y_prob_new = rf_final_model.predict_proba(X_new)[:, 1]  # Get fraud probabilities
optimal_threshold = 0.6  # Replace with your optimized threshold

# Convert probabilities to binary predictions using the optimal threshold
y_pred_new = (y_prob_new >= optimal_threshold).astype(int)

# Create a DataFrame with 'id' and 'fraud_prediction'
predictions_df = pd.DataFrame({
    'id': ids,
    'fraud_prediction': y_pred_new  # 1 = Fraud, 0 = Not Fraud
})

# Save predictions to CSV
predictions_path = '/content/drive/MyDrive/IT1244 Project/Predictions/fraud_predictions.csv'
predictions_df.to_csv(predictions_path, index=False)

print(f"Fraud predictions saved to: {predictions_path}")

def summarize_predictions(y_pred):
    """Summarize the number of fraud and non-fraud predictions."""
    fraud_count = (y_pred == 1).sum()
    non_fraud_count = (y_pred == 0).sum()

    print("\nSummary of Predictions:")
    print(f"Fraudulent Transactions: {fraud_count}")
    print(f"Non-Fraudulent Transactions: {non_fraud_count}")

# Summarize the predictions
summarize_predictions(y_pred_new)