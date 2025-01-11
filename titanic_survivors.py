import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

#Step1: Load the dataset
data = pd.read_csv("synthetic_titanic.csv")
print(data.head())
print(data.shape)
print(type(data))

#Step2: Feature Engineering (Create new features)
#*feature engineering helps capture more relevant information and improve model performance by 
#providing additional, meaningful patterns. However, it’s important to balance this with feature selection 
#to avoid unnecessary complexity and overfitting.

data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
data['IsAlone'] = (data['FamilySize'] == 1).astype(int)

#Step3: Selecting features
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'IsAlone']
target = 'Survived'

#Step 4: Splitting data for preprocessing
x = data[features]
y = data[target]

#Step5: Preprocessing(telling the model, which features are categorical and which are numerical)
cat_features = ['Sex', 'Embarked']
num_features = ['Pclass', 'Age', 'Fare', 'FamilySize', 'IsAlone']

#Step6: Preprocessing Pipeline(for Imputed missing values and scaled numerical features.)
#*Used One-Hot Encoding for categorical variable Models can't directly process text values like Sex or Embarked. 
# *One-Hot Encoding converts them into binary (0/1) format so they can be used as inputs.
#*The ColumnTransformer organizes preprocessing into a pipeline, applying the correct transformations to 
# numerical and categorical columns separately, ensuring a structured and reusable workflow.
#*This is the general framework for most ML projects, with adjustments based on data type and model choice and its requirements.
#*The SimpleImputer is a tool in scikit-learn that fills in missing values in a dataset using a specified strategy,
#  such as mean, median, or the most frequent value
#*We selected the median for imputation because it is less sensitive to outliers compared to the mean, 
# making it a more robust choice for handling missing values in numerical data
#*StandardScaler handles the scaling

numeric_transformer = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
    ])

#*drop='first' would keep only the columns for the places where passengers embarked (like "C" and "Q"),
#  removing the "S" column, as it can be inferred from the other two.
# *SimpleImputer fills missing categorical values with the most common category
#OneHotEncoder outputs a sparse matrix (or dense matrix in some cases), and SMOTE requires a dense numeric array.
#OneHotEncoder by default produces a sparse matrix, which can cause issues with certain algorithms or steps like SMOTE.
#thats why added #sparse = false

categorical_transformer = Pipeline(steps = [
    ('imputer',SimpleImputer(strategy= 'most_frequent')),
    ('onehot', OneHotEncoder(drop='first', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers= [
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features)
    ]
)

#Created separate pipelines for Logistic Regression, Random Forest, and Gradient Boosting.
#Preprocessing is applied consistently within the pipeline.

#Step7: Train-Test Split
#First mistake, changed the order of the x_train, y_train wrong. Following is the correct order
#used stratify so that class distribution in both training and validation sets is similar.

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42, stratify=y) 
print(x_train.shape) #good practice to see the shape
print(y_train.shape)
print(type(x_train)) 
print(type(y_train))

#New Step: SMOTE for balancing the class, as the data is imbalanced 
#Use Synthetic Minority Over-sampling Technique (SMOTE) to balance the classes in the training data.
#The fit_resample method in SMOTE is used to fit the SMOTE model and then resample the dataset to address class imbalance 
# by generating synthetic data points for the minority class.
#Here's a brief breakdown:
#fit(): This step is used to analyze the data and learn how to generate synthetic samples based on the minority class.
#resample(): After fitting, this step generates new synthetic samples to balance the class distribution.
#preprocessor.fit_transform(X_train): This applies preprocessing (e.g., imputation and one-hot encoding) to X_train 
# and transforms it into a suitable format.
#smote.fit_resample(): This method then generates synthetic samples from the minority class (y_train) and returns 
# a new version of the dataset (X_train_resampled, y_train_resampled) with balanced class distributions.
#So, fit_resample combines the process of fitting the model and resampling the data in one step.

#smote = SMOTE(random_state=42) #later changed x_train and y_train to  x_train_resampled and y_train_resampled
#x_train_resampled, y_train_resampled = smote.fit_resample(preprocessor.fit_transform(X_train), y_train)


#Step8: *Pipeline for Logistic Regression(Implementing this first as it is the most obvious choice)
#ensures that all preprocessing steps (like handling missing values and encoding) are applied automatically
#  before the model is trained, which simplifies the process, prevents errors, and ensures consistent application
#  of transformations during both training and testing.

logreg_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor), #added the class weight and random state so that the data is balanced
    ('classifier', LogisticRegression(class_weight='balanced', random_state=42)) 
])

#Step9: Train and Evaluate Logistic Regression. Use fit when you just want to learn from the data 
# (e.g., calculate mean or train a model) but don't need to apply any transformation or make predictions right away.
#*Use fit_transform when you want to both learn from the data and immediately apply the transformation 
# (e.g., scaling or encoding) to the data in one step.
#*Use fit_predict when you want to both learn from the data and predict the labels 
# (e.g., clustering or classification) after fitting the model.
#*we didn't use fit_predict because we were not performing clustering or unsupervised learning; 
# instead, we were training a classifier to predict the target variable Survived. 
# We used fit to train the model and then used predict to make predictions after fitting the model, 
# as fit_predict is typically used for clustering algorithms like k-means, not supervised classification.

logreg_pipeline.fit(x_train,y_train)
y_pred_logreg = logreg_pipeline.predict(x_test)
print("Logistic Regression Performance:")

#*predict: Returns the predicted class labels (e.g., 0 or 1 for binary classification).
#*predict_proba: Returns the predicted probabilities for each class. For binary classification, 
# it gives a two-column array: the first column for class 0 and the second for class 1.
#*predict_log_proba: Similar to predict_proba, but returns the logarithm of the predicted probabilities.

print(classification_report(y_test, y_pred_logreg))

#*we used predict_proba to get the probability for the positive class (class 1, i.e., Survived) 
# and not just the predicted label. This is useful for calculating metrics like ROC-AUC, which requires probabilities, 
# not just the final predicted class. The [:, 1] selects the second column, which represents the probability of class 1.

print(f"ROC-AUC Score: {roc_auc_score(y_test, logreg_pipeline.predict_proba(x_test)[:,1])}")

#Step10: Pipeline for Random Forest 
rf_pipeline = Pipeline(steps = [
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42)) #Second mistake, forgot the parantheses after the RandomForestClassifier
])

rf_pipeline.fit(x_train, y_train)
y_pred_rf = rf_pipeline.predict(x_test)
print("\nRandom Forest Performance:")
print(classification_report(y_test, y_pred_rf))
print(f"ROC-AUC Score: {roc_auc_score(y_test, rf_pipeline.predict_proba(x_test)[:, 1])}")

#Step11: Pipeline for Gradient Boosting

gb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classfier', GradientBoostingClassifier()) #Third mistake, forgot parantheses here as well
])

gb_pipeline.fit(x_train, y_train)
y_pred_gb = gb_pipeline.predict(x_test)
print("\nGradient Boosting Performance:")
print(classification_report(y_test, y_pred_gb))
print(f"ROC-AUC Score: {roc_auc_score(y_test, gb_pipeline.predict_proba(x_test)[:, 1])}")

#Step12: Hyperparameter Tuning for Random Forest
#*hyperparameter tuning involves experimenting with different values for a model’s parameters to 
# find the combination that gives the best performance. For Random Forest, parameters like n_estimators 
# (number of trees), max_depth (tree depth), and min_samples_split (minimum samples to split a node) are 
# commonly tuned.
#* Different models have diff parameters and by mentioning them in param_grid the model knows that which model's 
# hyperparamter tuning needs to be done
#*n_estimators (100, 200): A common range for the number of trees in Random Forest; larger values tend to 
# improve performance but also increase computation time.
#*max_depth (None, 10, 20): Controls how deep the decision trees can grow; None means unlimited depth, while values 
# like 10 or 20 are reasonable for preventing overfitting.
#*min_samples_split (2, 5): Specifies the minimum samples required to split an internal node; 2 is typical for small datasets, 
# but larger values like 5 help reduce overfitting in larger datasets.
#*These values are often fine-tuned through hyperparameter optimization (e.g., GridSearchCV) to find the optimal range 
#Demonstrated tuning for Random Forest with GridSearchCV

param_grid = {
    'classifier__n_estimators': [100,200], #fourth mistake, only using single _ after classifier
    'classifier__max_depth' : [None,10,20],
    'classifier__min_samples_split' : [2,5]
}

#Have used Cross Validation as well here with GridSearchCV to improve performance
grid_search = GridSearchCV(rf_pipeline, param_grid, cv=3, scoring='roc_auc')
grid_search.fit(x_train, y_train)

#print("\nBest model:", grid_search.best_estimator_)
print("\nBest Parameters for Random Forest:", grid_search.best_params_)
print(f"Best ROC-AUC Score from Grid Search: {grid_search.best_score_}")

'''Next steps - Can and should make changes like -
Improvements:
Handling Class Imbalance:

You might want to try SMOTE (Synthetic Minority Over-sampling Technique) or class weight adjustments in your models to address the class imbalance.
SMOTE can help by generating synthetic samples for the minority class to balance the dataset.
Feature Engineering:

You can add more informative features or interactions between existing ones, such as age ranges or interaction terms.
Hyperparameter Tuning:

You can further tune hyperparameters for Random Forest and Gradient Boosting to improve performance, particularly on the minority class.
Ensemble Methods:

Consider ensemble models or stacking models to combine the best of each model.
'''