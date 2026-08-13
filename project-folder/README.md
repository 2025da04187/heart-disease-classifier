a. **Problem Statement :**

The objective of this project is to build a machine learning classification pipeline to predict whether a patient has a risk of a heart disease or attack based on their medical history and lifestyle indicators. 
By deploying an interactive web application, medical professionals or users can evaluate patient data against multiple trained models to predict cardiovascular risks effectively.

----

b. **Dataset Description:** 

Dataset Name: Heart Disease Health Indicators Dataset

Source: Kaggle - https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset

Features: 21 independent features (including HighBP, HighChol, BMI, Smoker, Stroke, Diabetes, etc.)

Instances: 253,680 (A representative subset is used for test evaluation)

Target Variable: HeartDiseaseorAttack (Binary: 0 = No Heart Disease, 1 = Heart Disease)

Criteria Check: Features (21 >= 12), Instances (200,000 >= 500).

----

c. **Github Repository Link :**

https://github.com/2025da04187/heart-disease-classifier.git

----

d. **Streamlit App Link :**

https://heart-disease-classifier-2025da04187.streamlit.app/

----

e. **Models Used:**

Logistic Regression

Decision Tree Classifier

K-Nearest Neighbor Classifier (KNN)

Naive Bayes Classifier (GaussianNB)

Ensemble Model - Random Forest

Support Vector Machine (SVM)

----

f. **Comparison Table :**

|ML Model Name | Accuracy | AUC    | Precision | Recall | F1 Score | MCC    |
|--------------|----------|--------|-----------|--------|----------|--------|
|Logistic Regression| 0.9103   | 0.851  | 0.5652    | 0.1402 | 0.2246   | 0.2499 |
|Decision Tree| 0.9073   | 0.5947 | 0.5000    | 0.0836 | 0.1432   | 0.1762 |
|K-Nearest Neighbor| 0.8998   | 0.716  | 0.3929    | 0.1482 | 0.2153   | 0.197  |
|Naive Bayes (Gaussian)| 0.8295   | 0.8065 | 0.2855    | 0.558  | 0.3777   | 0.3127 |
|Random Forest| 0.9062   | 0.8139 | 0.4474    | 0.0458 | 0.0831   | 0.1197 |
|Support Vector Machine| 0.9077   | 0.6878 | 0.5294    | 0.0485 | 0.0889   | 0.1394 |

----

g. **Observations Table :**

|ML Model Name | Observation about model performance                                                                                                                                                                                                      |
|--------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|Logistic Regression| Provided strong overall accuracy and the highest AUC, showing reliable probability estimations for cardiovascular risks.                                                                                                                 |
|Decision Tree| Showed high raw accuracy but worst AUC, second lowest recall, Extremely poor performance at finding true positive cases despite capturing complex rules.                                                                                 |
|K-Nearest Neighbor| Performed decently on accuracy but suffered from relatively low precision and recall. Meaning it struggled to correctly identify actual positive heart disease cases. Suffers from a high number of false positives and false negatives. |
|Naive Bayes (Gaussian)| Lowest overall Accuracy and precision. Strong AUC score, but highest recall by a huge margin. Good model to use if finding positive cases is the main priority.                                                                          |
|Random Forest| Exhibited High baseline accuracy, and good AUC score. Lowest recall score, and worst F1 score. Severely struggles to catch positive cases.                                                                                               |
|Support Vector Machine| Yielded the highest absolute accuracy and decent precision. However, resulting in high precision but low recall and AU score. Fails to successfully identify most positive instances.                                                                                                           |

----

h. **Model Training :**

The python script model_training.py is used to train and generate multiple model files. It also generates test csv file.
