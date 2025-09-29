import pandas as pd
import numpy as np
import joblib
import argparse

def create_features(df):
    print("Creating features...")
    # Time-based features
    df['Day'] = (df['Time'] // (3600*24)).astype(int)
    df['Hour'] = (df['Time'] // 3600) % 24

    # Amount-based features
    df['Amount_log'] = np.log1p(df['Amount'])
    df['Amount_scaled'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()
    df['Amount_log_scaled'] = (df['Amount_log'] - df['Amount_log'].mean()) / df['Amount_log'].std()

    # --- PCA features ---
    # If you have the PCA object, this will work. Otherwise, PCA columns will be skipped.
    try:
        pca = joblib.load('models/pca.pkl')
        v_features = [f'V{i}' for i in range(1, 29)]
        pca_values = pca.transform(df[v_features])
        for i in range(pca.n_components_):
            df[f'PCA{i+1}'] = pca_values[:, i]
    except Exception as e:
        print("Error loading or applying PCA:", e)
        # Do not return here, continue with non-PCA features

    return df

def main(input_path, output_path, model_path):
    print(f"Loading data from {input_path}...")
    new_data = pd.read_csv(input_path)
    original_data = new_data.copy()
    processed_data = create_features(new_data)
    print(f"Loading model from {model_path}...")
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return

    # List all features your model expects
    features = [
          'PCA1','PCA2',  # Add more if your model expects more PCA components
        'V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
        'V21','V22','V23','V24','V25','V26','V27','V28',
        'Amount','Amount_log','Amount_scaled','Amount_log_scaled','Day','Hour'
    ]
    # Check if all required features exist
    missing = [f for f in features if f not in processed_data.columns]
    if missing:
        print(f"Error: The following required features are missing: {missing}")
        print("Available columns:", processed_data.columns.tolist())
        return

    data_for_prediction = processed_data[features]

    print("Making predictions...")
    predictions = model.predict(data_for_prediction)
    probabilities = model.predict_proba(data_for_prediction)[:, 1]
    original_data['predicted_class'] = predictions
    original_data['fraud_probability'] = probabilities
    print(f"Saving results to {output_path}...")
    original_data.to_csv(output_path, index=False)
    print("Prediction process completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict fraud on new transaction data.")
    parser.add_argument('--input', required=True, help="Path to the input CSV file with new data.")
    parser.add_argument('--output', required=True, help="Path to save the output CSV file with predictions.")
    parser.add_argument('--model', default="models/best_model.pkl", help="Path to the trained model .pkl file.")
    args = parser.parse_args()
    main(args.input, args.output, args.model)