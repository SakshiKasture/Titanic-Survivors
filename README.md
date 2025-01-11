Titanic Survival Prediction using Machine Learning
Project Description
This project predicts passenger survival on the Titanic using machine learning models. A synthetic dataset mimics the real Titanic data, incorporating features like passenger class, age, and family size. The project emphasizes feature engineering, data preprocessing, and addressing class imbalance to improve model performance.

Technologies Used
Python
Pandas, NumPy
Scikit-learn
Imbalanced-learn

Dataset
Features:
Pclass: Passenger class (1, 2, 3)
Sex: Gender (male, female)
Age: Age of the passenger
Fare: Ticket fare
Embarked: Embarkation port (C, Q, S)
FamilySize: Number of family members traveling
IsAlone: Whether the passenger was traveling alone
Target: Survived (1 for survived, 0 for not)

Model Workflow
Feature Engineering:
New features created to enhance model insights.
Preprocessing:
Missing values imputed, categorical variables one-hot encoded, and numerical features scaled.
Training:
Three models trained: Logistic Regression, Random Forest, and Gradient Boosting.
Evaluation:
Performance measured using classification metrics and ROC-AUC scores.

Future Work
Experiment with additional machine learning algorithms.
Further optimize hyperparameters.
Enhance class imbalance handling with alternative techniques.
