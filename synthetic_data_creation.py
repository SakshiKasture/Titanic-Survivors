import numpy as np
import pandas as pd

# Set a random seed for reproducibility
np.random.seed(42)

# Number of samples to generate
n_samples = 1000

# Generate synthetic Titanic-like data
data = {
    'Pclass': np.random.choice([1, 2, 3], size=n_samples, p=[0.2, 0.3, 0.5]),  # Class of passenger
    'Sex': np.random.choice(['male', 'female'], size=n_samples),  # Gender
    'Age': np.random.uniform(1, 80, size=n_samples),  # Age between 1 and 80
    'Fare': np.random.uniform(10, 500, size=n_samples),  # Fare between 10 and 500
    'Embarked': np.random.choice(['C', 'Q', 'S'], size=n_samples),  # Embarked (C = Cherbourg, Q = Queenstown, S = Southampton)
    'SibSp': np.random.randint(0, 5, size=n_samples),  # Number of siblings/spouses aboard
    'Parch': np.random.randint(0, 5, size=n_samples),  # Number of parents/children aboard
    'FamilySize': lambda x: x['SibSp'] + x['Parch'] + 1,  # Family size (including the passenger)
    'IsAlone': lambda x: (x['FamilySize'] == 1).astype(int),  # Whether the passenger is alone or not
    'Survived': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])  # Random survival (0 = No, 1 = Yes)
}

# Create the DataFrame
df = pd.DataFrame(data)

# Create FamilySize and IsAlone based on SibSp and Parch
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# Show the first few rows of the dataset
print(df.head())

# Optional: Save the DataFrame to a CSV
df.to_csv("synthetic_titanic.csv", index=False)