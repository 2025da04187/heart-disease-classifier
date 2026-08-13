import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

# 1. Create Model Directory
if not os.path.exists('project-folder/model'):
    os.makedirs('project-folder/model')

# 2. Load Dataset (Download csv from Kaggle and place it in same directory)
# Using a sampled dataset for faster training and reasonable GitHub file sizes
df = pd.read_csv('heart_disease_health_indicators_BRFSS2015.csv').sample(n=20000, random_state=42)

X = df.drop('HeartDiseaseorAttack', axis=1)
y = df['HeartDiseaseorAttack']

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Save Test Data for Streamlit Upload
test_data = pd.concat([X_test, y_test], axis=1)
test_data.to_csv('project-folder/test_data.csv', index=False)

# 5. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
joblib.dump(scaler, 'project-folder/model/scaler.pkl')

# 6. Initialize Models
models = {
    'Logistic_Regression': LogisticRegression(max_iter=1000),
    'Decision_Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive_Bayes': GaussianNB(),
    'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

# 7. Train, Evaluate, and Save Models
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)

    # Save Model
    joblib.dump(model, f'project-folder/model/{name}.pkl')

print("All models trained and saved successfully in the 'project-folder/model/' folder.")
print("test_data.csv has been generated.")
