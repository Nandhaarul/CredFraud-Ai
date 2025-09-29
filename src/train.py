import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import joblib

# Load your dataset
df = pd.read_csv('data/raw/creditcard.csv')

# Feature engineering
df['Day'] = (df['Time'] // (3600*24)).astype(int)
df['Hour'] = (df['Time'] // 3600) % 24
df['Amount_log'] = np.log1p(df['Amount'])
df['Amount_scaled'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()
df['Amount_log_scaled'] = (df['Amount_log'] - df['Amount_log'].mean()) / df['Amount_log'].std()

# PCA on V1-V28
v_features = [f'V{i}' for i in range(1, 29)]
pca = PCA(n_components=2, random_state=42)  # Use the same n_components as in your predict.py
pca_values = pca.fit_transform(df[v_features])
for i in range(pca.n_components_):
    df[f'PCA{i+1}'] = pca_values[:, i]

# Save the PCA object for prediction
joblib.dump(pca, 'models/pca.pkl')

# Features for training (must match your predict.py)
features = [
    'PCA1','PCA2',
    'V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
    'V21','V22','V23','V24','V25','V26','V27','V28',
    'Amount','Amount_log','Amount_scaled','Amount_log_scaled','Day','Hour'
]

X = df[features]
y = df['Class']  # Assuming 'Class' is your target column

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save the model
joblib.dump(model, 'models/best_model.pkl')
print("Model and PCA object saved.")