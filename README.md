a. **Problem Statement :**

The objective of this project is to build a machine learning classification pipeline to predict whether a patient has a risk of a heart disease or attack based on their medical history and lifestyle indicators. By deploying an interactive web application, medical professionals or users can evaluate patient data against multiple trained models to predict cardiovascular risks effectively.

----

b. **Dataset Description:** 

Dataset Name: Heart Disease Health Indicators Dataset

Source: Kaggle - https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset

Features: 21 independent features (including HighBP, HighChol, BMI, Smoker, Stroke, Diabetes, etc.)

Instances: 253,680 (A representative subset is used for test evaluation)

Target Variable: HeartDiseaseorAttack (Binary: 0 = No Heart Disease, 1 = Heart Disease)

Criteria Check: Features (21 >= 12), Instances (253,680 >= 500).

----

c. **Github Repository Link :**

[Insert your GitHub Repo Link Here]

----

d. **Models Used:**

Logistic Regression

Decision Tree Classifier

K-Nearest Neighbor Classifier (KNN)

Naive Bayes Classifier (GaussianNB)

Ensemble Model - Random Forest

Support Vector Machine (SVM)

----

e. **Comparison Table :**

|ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC|
|--------------|----------|-----|-----------|--------|----------|----|
|Logistic Regression|0.908|0.841|0.534|0.165|0.252|0.258|
|Decision Tree|0.854|0.602|0.264|0.281|0.272|0.185|
|K-Nearest Neighbor|0.896|0.725|0.395|0.134|0.200|0.178|
|Naive Bayes (Gaussian)|0.835|0.812|0.315|0.502|0.387|0.301|
|Random Forest|0.906|0.836|0.505|0.125|0.200|0.215|
|Support Vector Machine|0.910|0.815|0.562|0.145|0.230|0.250|

----

f. **Observations Table :**

|ML Model Name | Observation about model performance|
|--------------|------------------------------------|
|Logistic Regression| Provided strong overall accuracy and the highest AUC, showing reliable probability estimations for cardiovascular risks.|
|Decision Tree| Showed lower overall accuracy and AUC, indicating a tendency to overfit the training data despite capturing complex rules. |
|K-Nearest Neighbor| Performed decently on accuracy but suffered from low recall, meaning it struggled to correctly identify actual positive heart disease cases. |
|Naive Bayes (Gaussian)| Achieved the highest Recall and MCC. Though overall accuracy dropped, it proved best at identifying true positive risk cases.|
|Random Forest| Exhibited excellent accuracy and AUC by reducing the variance of decision trees, but precision/recall balance was slightly skewed towards the majority class. |
|Support Vector Machine| Yielded the highest absolute accuracy. However, similar to RF, it was conservative, resulting in high precision but low recall. |