import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

base_path = r'C:\Users\karan\PycharmProjects\QSAR_toxicity_prediction'
input_file = os.path.join(base_path, 'filtered_descriptors.csv')

df = pd.read_csv(input_file)
X = df.drop(["SMILES", "LD50"], axis=1).apply(pd.to_numeric, errors='coerce')
y = df["LD50"]

print(f"Dataset Loaded: {X.shape[0]} compounds.")

# train and split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)

print("Running Cross-Validation on training set...")
cv_scores = cross_val_score(cv_model, X_train, y_train, cv=kf, scoring="r2", n_jobs=-1)
print(f"Mean CV R2: {cv_scores.mean():.4f}")


# train the model
model = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

# evaluation
y_pred = model.predict(X_test)
r2_score= r2_score(y_test, y_pred)
rmse_score = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Test R2 Score: {r2_score:.4f}")
print(f"Test RMSE: {rmse_score:.4f}")
# save the model
model_path = os.path.join(base_path, "QSAR_Toxicity_model.pkl")
feature_path = os.path.join(base_path, "QSAR_feature_list.pkl")

joblib.dump(model, model_path)
joblib.dump(X.columns.tolist(), feature_path)

print(f"Files saved successfully in {base_path}")