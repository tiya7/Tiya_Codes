===1===

import pandas as pd
import numpy as np
df = pd.read_csv(r"C:\Users\supri\Downloads\Titanic-Dataset.csv")
print(df.head())
----------------
print(df.isnull().sum())
-----------------
print(df.describe())
-------------------
print(df.shape)
---------------
print(df.info())
---------------
print(df.dtypes)
----------------
df['Age'] = df['Age'].astype(float)
df['Survived'] = df['Survived'].astype(int)
df['Age'] = (df['Age'] - df['Age'].min()) / (df['Age'].max() - df['Age'].min())
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'])
print(df.head())
_________________________________________________________________________________________________________________
_________________________________________________________________________________________________________________
===2===

import pandas as pd
import numpy as np
data = {
    'Name': ['A','B','C','D','E'],
    'Marks': [85, 90, np.nan, 200, 75],   # missing + outlier
    'StudyHours': [2, 3, 4, 100, 2],      # outlier
    'Attendance': [80, 85, 90, np.nan, 88]
}
df = pd.DataFrame(data)
print("Original Data:\n", df)
df.fillna(df.mean(numeric_only=True), inplace=True)
print("\nAfter Handling Missing Values:\n", df)
numeric_df = df.select_dtypes(include=np.number)

Q1 = numeric_df.quantile(0.25)
Q3 = numeric_df.quantile(0.75)
IQR = Q3 - Q1

df = df[~((numeric_df < (Q1 - 1.5 * IQR)) | 
          (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)]

print("\nAfter Removing Outliers:\n", df)
df['Log_Marks'] = np.log(df['Marks'])
print("\nAfter Transformation:\n", df)
_________________________________________________________________________________________________________________
_________________________________________________________________________________________________________________
===3===

import pandas as pd
data = {
    'AgeGroup': ['Young','Young','Adult','Adult','Senior','Senior'],
    'Income': [20000,25000,40000,42000,30000,28000]
}
df = pd.DataFrame(data)
grouped = df.groupby('AgeGroup')['Income']
print(grouped.agg(['mean','median','min','max','std']))
------------------------
income_list = df.groupby('AgeGroup')['Income'].apply(list)
print(income_list)
-------------------------
import pandas as pd
df = pd.read_csv(r"C:\Users\supri\Downloads\Iris.csv")
print(df.columns) 
grouped = df.groupby('Species')
print(grouped.mean())
print(grouped.std())
print(grouped.quantile([0.25, 0.5, 0.75]))
df.columns = df.columns.str.lower()
grouped = df.groupby('species')
_________________________________________________________________________________________________________________
_________________________________________________________________________________________________________________
===4===

!pip install pandas numpy scikit-learn matplotlib seaborn
-----------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
-----------------------
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)
print(df.head())
-------------------------
print(df.shape)
print(df.info())
print(df.describe())
----------------------------
print(df.isnull().sum())
---------------------------
X = df.drop('medv', axis=1)   # independent variables
y = df['medv']                # dependent variable (price)
------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)
-------------------------
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
print("Model trained successfully!")
------------------
y_pred = model.predict(X_test)
print("Predictions:\n", y_pred[:5])
-----------------------------
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)
-----------------------------
import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()
_________________________________________________________________________________________________________________
_________________________________________________________________________________________________________________
===5===

!pip install pandas numpy scikit-learn matplotlib
-----------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
-----------------------------
df = pd.read_csv(r"C:\Users\supri\Downloads\Social_Network_Ads.csv")
print(df.head())
-------------------------
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']
--------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=0
)
------------------
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
-----------------------------
model = LogisticRegression()
model.fit(X_train, y_train)
-----------------------------
y_pred = model.predict(X_test)
print(y_pred)
------------------------------
cm = confusion_matrix(y_test, y_pred)
matrix = pd.DataFrame(
    cm,
    columns=['Predicted Negative', 'Predicted Positive'],
    index=['Actual Negative', 'Actual Positive']
)

print(matrix)
---------------------
TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TP = cm[1][1]

print("True Positive =", TP)
print("False Positive =", FP)
print("True Negative =", TN)
print("False Negative =", FN)
-------------------
accuracy = (TP + TN) / (TP + TN + FP + FN)
error_rate = (FP + FN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
print("Accuracy =", accuracy)
print("Error Rate =", error_rate)
print("Precision =", precision)
print("Recall =", recall)
--------------------------------
_________________________________________________________________________________________________________________
_________________________________________________________________________________________________________________
===6===

!pip install pandas numpy scikit-learn
-------------
iris = load_iris()
df = pd.read_csv(r"C:\Users\supri\Downloads\Iris.csv")
df['species'] = iris.target
df['species'] = df['species'].map({
    0: 'Iris-setosa',
    1: 'Iris-versicolor',
    2: 'Iris-virginica'
})
print(df.head())
------------------
X = df.drop('species', axis=1)   # features
y = df['species']               # target
-----------------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
----------------------
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)
print("Model trained successfully")
------------------------
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=labels,
    yticklabels=labels
)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")
plt.show()
--------------------------
import numpy as np
y_test_bin = [1 if i == 'Iris-setosa' else 0 for i in y_test]
y_pred_bin = [1 if i == 'Iris-setosa' else 0 for i in y_pred]
cm_bin = confusion_matrix(y_test_bin, y_pred_bin)
TN, FP, FN, TP = cm_bin.ravel()
print("TP:", TP)
print("TN:", TN)
print("FP:", FP)
print("FN:", FN)
accuracy = (TP + TN) / (TP + TN + FP + FN)
error_rate = 1 - accuracy
precision = TP / (TP + FP)
recall = TP / (TP + FN)
print("Accuracy:", accuracy)
print("Error Rate:", error_rate)
print("Precision:", precision)
print("Recall:", recall)
------------------------------------
OR
----------------------------------------
# === 6 ===

# Install required libraries
!pip install pandas numpy scikit-learn seaborn matplotlib

# ---------------------------------------------

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

# ---------------------------------------------

# Load iris dataset
iris = load_iris()

# Read CSV file
df = pd.read_csv(r"C:\Users\supri\Downloads\Iris.csv")

# Add species column
df['species'] = iris.target

# Map numeric values to species names
df['species'] = df['species'].map({
    0: 'Iris-setosa',
    1: 'Iris-versicolor',
    2: 'Iris-virginica'
})

# Display dataset
print(df.head())

# ---------------------------------------------

# Features and target
X = df.drop('species', axis=1)

y = df['species']

# ---------------------------------------------

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# ---------------------------------------------

# Create Gaussian Naive Bayes model
model = GaussianNB()

# Train model
model.fit(X_train, y_train)

print("Model trained successfully")

# ---------------------------------------------

# Prediction
y_pred = model.predict(X_test)

print("Predicted Values:")
print(y_pred)

# ---------------------------------------------

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

labels = [
    'Iris-setosa',
    'Iris-versicolor',
    'Iris-virginica'
]

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=labels,
    yticklabels=labels
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")

plt.show()

# ---------------------------------------------

# Convert into binary classification
# Iris-setosa = 1
# Others = 0

y_test_bin = [1 if i == 'Iris-setosa' else 0 for i in y_test]

y_pred_bin = [1 if i == 'Iris-setosa' else 0 for i in y_pred]

# ---------------------------------------------

# Binary confusion matrix
cm_bin = confusion_matrix(y_test_bin, y_pred_bin)

TN, FP, FN, TP = cm_bin.ravel()

print("TP:", TP)
print("TN:", TN)
print("FP:", FP)
print("FN:", FN)

# ---------------------------------------------

# Performance metrics
accuracy = (TP + TN) / (TP + TN + FP + FN)

error_rate = 1 - accuracy

precision = TP / (TP + FP)

recall = TP / (TP + FN)

print("Accuracy:", accuracy)

print("Error Rate:", error_rate)

print("Precision:", precision)

print("Recall:", recall)
_________________________________________________________________________________________________________________
_________________________________________________________________________________________________________________
===7===

!pip install nltk scikit-learn
-----------------
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
text = """
Text Analytics is the process of converting unstructured text data
into meaningful information. Natural Language Processing helps
computers understand human language.
"""
---------------------
tokens = word_tokenize(text)
print("Tokens:\n", tokens)
-------------
pos_tags = pos_tag(tokens)
print(pos_tags)
---------
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in tokens if word.lower() not in stop_words]
print("After Stopword Removal:\n", filtered_words)
--------------
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_words]
print("Stemmed Words:\n", stemmed_words)
--------------------
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
print("Lemmatized Words:\n", lemmatized_words)
-----------------
documents = [
    "Text analytics converts text into information",
    "Natural language processing understands text"
]
cv = CountVectorizer()
tf_matrix = cv.fit_transform(documents)
tf = pd.DataFrame(
    tf_matrix.toarray(),
    columns=cv.get_feature_names_out()
)
print("\nTerm Frequency:")
print(tf)
---------------------
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(documents)
print("TF-IDF Values:\n", X_tfidf.toarray())
-----------------------------
_________________________________________________________________________________________________________________
_________________________________________________________________________________________________________________
===8===

import seaborn as sns
import matplotlib.pyplot as plt
df = sns.load_dataset('titanic')
print(df.head())
print(df.info())
--------------------
sns.countplot(x='survived', data=df)
plt.show()
sns.countplot(x='survived', hue='sex', data=df)
plt.show()
sns.histplot(df['age'], kde=True)
plt.show()
----------------
import seaborn as sns
import matplotlib.pyplot as plt
df = sns.load_dataset('titanic')
sns.histplot(df['fare'], kde=True)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Count")
plt.show()
-----------------------
OR
-----------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read Titanic dataset using pd.read_csv()

df = pd.read_csv(r"C:\Users\supri\Downloads\titanic.csv")

# Display first 5 rows
print(df.head())

# Display dataset information
print(df.info())

# ---------------------------------------------

# Count plot of survived passengers

sns.countplot(x='survived', data=df)

plt.show()

# ---------------------------------------------

# Count plot based on gender

sns.countplot(x='survived', hue='sex', data=df)

plt.show()

# ---------------------------------------------

# Age distribution graph

sns.histplot(df['age'], kde=True)

plt.show()

# ---------------------------------------------

# Fare distribution graph

sns.histplot(df['fare'], kde=True)

plt.title("Fare Distribution")

plt.xlabel("Fare")

plt.ylabel("Count")

plt.show()
_________________________________________________________________________________________________________________
_________________________________________________________________________________________________________________
===9===
import seaborn as sns
import matplotlib.pyplot as plt
df = sns.load_dataset('titanic')
sns.boxplot(x='sex', y='age', hue='survived', data=df)
plt.title("Age Distribution by Gender and Survival")
plt.xlabel("Gender")
plt.ylabel("Age")
plt.show()
---------------------
OR
---------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read Titanic dataset using pd.read_csv()

df = pd.read_csv(r"C:\Users\supri\Downloads\titanic.csv")

# Boxplot for age distribution by gender and survival

sns.boxplot(
    x='sex',
    y='age',
    hue='survived',
    data=df
)

plt.title("Age Distribution by Gender and Survival")

plt.xlabel("Gender")

plt.ylabel("Age")

plt.show()
_________________________________________________________________________________________________________________
_________________________________________________________________________________________________________________
===10===

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
df = sns.load_dataset('iris')
print(df.head())
print(df.dtypes)
df.hist(figsize=(8,6))
plt.show()
--------------------------
import seaborn as sns
for col in df.columns[:-1]:
    sns.boxplot(x=df[col])
    plt.title(col)
    plt.show()
------------------
OR
--------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read Iris dataset using pd.read_csv()

df = pd.read_csv(r"C:\Users\supri\Downloads\Iris.csv")

# Display first 5 rows
print(df.head())

# Display data types
print(df.dtypes)

# ---------------------------------------------

# Histogram plots

df.hist(figsize=(8,6))

plt.show()

# ---------------------------------------------

# Boxplots for each column

for col in df.columns[:-1]:

    sns.boxplot(x=df[col])

    plt.title(col)

    plt.show()














                           #Assignment - 1
                      # __________________________
                       #__________________________


# ============================================================
# DATA WRANGLING USING PYTHON
# ============================================================

# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import seaborn as sns

# ============================================================
# 2. LOAD OPEN SOURCE DATASET
#
# Dataset Name:
# Titanic Dataset
#
# Dataset Source:
# https://www.kaggle.com/datasets/yasserh/titanic-dataset
#
# Here we are directly loading the Titanic dataset
# using seaborn library.
# ============================================================

df = sns.load_dataset('titanic')

# ============================================================
# 3. DISPLAY FIRST 5 ROWS OF DATASET
# ============================================================

print("\n================ FIRST 5 ROWS ================\n")
print(df.head())

# ============================================================
# 4. CHECK DIMENSIONS OF DATAFRAME
# ============================================================

print("\n================ DATAFRAME SHAPE ================\n")
print("Rows and Columns :", df.shape)

# ============================================================
# 5. CHECK MISSING VALUES
# ============================================================

print("\n================ MISSING VALUES ================\n")
print(df.isnull().sum())

# ============================================================
# 6. DATASET INFORMATION
# ============================================================

print("\n================ DATASET INFORMATION ================\n")
print(df.info())

# ============================================================
# 7. STATISTICAL SUMMARY
# ============================================================

print("\n================ STATISTICAL SUMMARY ================\n")
print(df.describe())

# ============================================================
# 8. VARIABLE DESCRIPTIONS
# ============================================================

print("\n================ VARIABLE DESCRIPTIONS ================\n")

variable_description = {
    "survived": "Survival status (0 = No, 1 = Yes)",
    "pclass": "Passenger class",
    "sex": "Gender of passenger",
    "age": "Age of passenger",
    "sibsp": "Number of siblings/spouses aboard",
    "parch": "Number of parents/children aboard",
    "fare": "Ticket fare",
    "embarked": "Port of embarkation",
    "class": "Travel class",
    "who": "Man/Woman/Child",
    "adult_male": "Whether passenger is adult male",
    "deck": "Deck information",
    "embark_town": "Town of embarkation",
    "alive": "Survival in words",
    "alone": "Whether passenger was alone"
}

for key, value in variable_description.items():
    print(f"{key} --> {value}")

# ============================================================
# 9. CHECK DATA TYPES
# ============================================================

print("\n================ DATA TYPES ================\n")
print(df.dtypes)

# ============================================================
# 10. TYPE CONVERSION
# Convert survived column into category datatype
# ============================================================

df['survived'] = df['survived'].astype('category')

print("\n================ UPDATED DATATYPE ================\n")
print(df['survived'].dtype)

# ============================================================
# 11. HANDLE MISSING VALUES
# ============================================================

# Fill missing age values using mean
df['age'] = df['age'].fillna(df['age'].mean())

# Fill missing embarked values using mode
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# Fill missing embark_town values using mode
df['embark_town'] = df['embark_town'].fillna(df['embark_town'].mode()[0])

# Add new category before filling missing values
df['deck'] = df['deck'].cat.add_categories('Unknown')

# Fill missing deck values
df['deck'] = df['deck'].fillna('Unknown')

print("\n================ MISSING VALUES AFTER CLEANING ================\n")
print(df.isnull().sum())

# ============================================================
# 12. DATA NORMALIZATION
# Normalize fare column between 0 and 1
# ============================================================

df['fare'] = (
    (df['fare'] - df['fare'].min()) /
    (df['fare'].max() - df['fare'].min())
)

print("\n================ NORMALIZED FARE COLUMN ================\n")
print(df['fare'].head())

# ============================================================
# 13. CONVERT CATEGORICAL VARIABLES TO NUMERICAL VARIABLES
# ============================================================

# Convert sex column into numerical values
# male = 0
# female = 1

df['sex'] = df['sex'].map({'male': 0, 'female': 1})

print("\n================ ENCODED SEX COLUMN ================\n")
print(df['sex'].head())

# ============================================================
# 14. ONE HOT ENCODING
# Convert embarked categorical column into numerical columns
# ============================================================

df_encoded = pd.get_dummies(df, columns=['embarked'])

print("\n================ DATA AFTER ONE HOT ENCODING ================\n")
print(df_encoded.head())

# ============================================================
# 15. DISPLAY FINAL DATAFRAME
# ============================================================

print("\n================ FINAL DATAFRAME ================\n")
print(df.head())

# ============================================================
# 16. SAVE CLEANED DATASET
# ============================================================

df_encoded.to_csv("Cleaned_Titanic_Dataset.csv", index=False)

print("\nCleaned dataset saved successfully as:")
print("Cleaned_Titanic_Dataset.csv")

# ============================================================
# END OF PROGRAM
# ============================================================
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#1
# -------------------------------
# Data Wrangling using Python
# Dataset: Iris Dataset
# -------------------------------

# 1. Import Required Libraries
import pandas as pd
import numpy as np

# 2. Load Dataset
# Keep Iris.csv in the same folder as notebook

df = pd.read_csv("Iris.csv")

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# 3. Dataset Information
print("\nDataset Information:")
print(df.info())

# 4. Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# 5. Statistical Summary
print("\nStatistical Summary:")
print(df.describe())

# 6. Dataset Shape
print("\nShape of Dataset:")
print(df.shape)

# 7. Data Types
print("\nData Types:")
print(df.dtypes)

# 8. Convert Species to Category

df['Species'] = df['Species'].astype('category')

print("\nUpdated Data Types:")
print(df.dtypes)

# 9. Normalization
numeric_cols = ['SepalLengthCm', 'SepalWidthCm',
                'PetalLengthCm', 'PetalWidthCm']

# Min-Max Normalization

df[numeric_cols] = (
    df[numeric_cols] - df[numeric_cols].min()
) / (
    df[numeric_cols].max() - df[numeric_cols].min()
)

print("\nNormalized Data:")
print(df.head())

# 10. Convert Categorical to Numeric

df['Species'] = df['Species'].cat.codes

print("\nCategorical to Numerical Conversion:")
print(df.head())
----------------------------------------------------------------------------------------------------------------------------------------------------------------------
 #Assignment - 2
                      # __________________________
                       #__________________________




# ============================================================
# DATA WRANGLING II
# ============================================================

# ============================================================
# IMPORT REQUIRED LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CREATE ACADEMIC PERFORMANCE DATASET
# ============================================================

data = {
    'Student_ID': [1,2,3,4,5,6,7,8,9,10],

    'Name': ['Aman','Riya','Karan','Sneha','Rahul',
             'Priya','Arjun','Neha','Vikas','Pooja'],

    'DS_Marks': [85,90,np.nan,95,40,110,76,88,92,35],

    'CS_Marks': [78,85,82,np.nan,45,98,80,84,500,39],

    'ANN_Marks': [90,95,85,80,np.nan,98,75,88,92,60]
}

# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(data)

print("\n================ ORIGINAL DATASET ================\n")
print(df)

# ============================================================
# CHECK MISSING VALUES
# ============================================================

print("\n================ MISSING VALUES ================\n")
print(df.isnull().sum())

# ============================================================
# HANDLE MISSING VALUES
# ============================================================

df['DS_Marks'] = df['DS_Marks'].fillna(
    df['DS_Marks'].mean()
)

df['CS_Marks'] = df['CS_Marks'].fillna(
    df['CS_Marks'].mean()
)

df['ANN_Marks'] = df['ANN_Marks'].fillna(
    df['ANN_Marks'].mean()
)

print("\n================ DATA AFTER HANDLING MISSING VALUES ================\n")
print(df)

# ============================================================
# STATISTICAL SUMMARY
# ============================================================

print("\n================ STATISTICAL SUMMARY ================\n")
print(df.describe())

# ============================================================
# DETECT OUTLIERS USING IQR METHOD
# ============================================================

Q1 = df['CS_Marks'].quantile(0.25)

Q3 = df['CS_Marks'].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR

upper_limit = Q3 + 1.5 * IQR

print("\nLower Limit :", lower_limit)

print("Upper Limit :", upper_limit)

# ============================================================
# BOXPLOT FOR ALL SUBJECT MARKS
# ============================================================

plt.figure(figsize=(8,5))

sns.boxplot(data=df[['DS_Marks', 'CS_Marks', 'ANN_Marks']])

plt.title("Outlier Detection Using Boxplot")

plt.ylabel("Marks")

plt.show()

# ============================================================
# REMOVE OUTLIERS
# ============================================================

df = df[
    (df['CS_Marks'] >= lower_limit) &
    (df['CS_Marks'] <= upper_limit)
]

print("\n================ DATA AFTER REMOVING OUTLIERS ================\n")
print(df)

# ============================================================
# HISTOGRAM OF ORIGINAL DS MARKS
# ============================================================

plt.figure(figsize=(6,4))

sns.histplot(df['DS_Marks'], bins=5, kde=True)

plt.title("Original DS Marks Distribution")

plt.xlabel("DS Marks")

plt.ylabel("Frequency")

plt.show()

# ============================================================
# LOG TRANSFORMATION
# ============================================================

df['Log_DS_Marks'] = np.log(df['DS_Marks'])

print("\n================ LOG TRANSFORMED DATA ================\n")

print(df[['DS_Marks', 'Log_DS_Marks']])

# ============================================================
# HISTOGRAM OF LOG TRANSFORMED DS MARKS
# ============================================================

plt.figure(figsize=(6,4))

sns.histplot(df['Log_DS_Marks'], bins=5, kde=True)

plt.title("Log Transformed DS Marks Distribution")

plt.xlabel("Log DS Marks")

plt.ylabel("Frequency")

plt.show()

# ============================================================
# NORMALIZATION
# ============================================================

df['Normalized_DS_Marks'] = (
    (df['DS_Marks'] - df['DS_Marks'].min()) /
    (df['DS_Marks'].max() - df['DS_Marks'].min())
)

print("\n================ NORMALIZED DATA ================\n")

print(df[['DS_Marks', 'Normalized_DS_Marks']])

# ============================================================
# SAVE CLEANED DATASET
# ============================================================

df.to_csv("Academic_Performance_Cleaned.csv", index=False)

print("\nDataset Saved Successfully")

# ============================================================
# END OF PROGRAM
# ============================================================
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#2
# ---------------------------------------------
# Academic Performance Dataset - Data Wrangling
# ---------------------------------------------

import pandas as pd
import numpy as np

# Create Dataset

data = {
    "Name": ["Amit", "Riya", "Sam", "Neha", "Raj",
             "Pooja", "Karan", "Sneha", "Arjun", "Meera"],

    "Maths": [78, 85, np.nan, 95, 40,
              120, 76, 88, 90, 35],

    "Science": [80, 79, 85, np.nan, 45,
                92, 75, 89, 91, 38],

    "Attendance": [85, 90, 88, 92, 60,
                   95, np.nan, 89, 91, 58]
}

# Create DataFrame

df = pd.DataFrame(data)

print("Original Dataset:\n")
print(df)

# Missing Values
print("\nMissing Values:\n")
print(df.isnull().sum())

# Fill Missing Values

df["Maths"] = df["Maths"].fillna(df["Maths"].mean())
df["Science"] = df["Science"].fillna(df["Science"].mean())
df["Attendance"] = df["Attendance"].fillna(df["Attendance"].mean())

print("\nDataset After Handling Missing Values:\n")
print(df)

# Outlier Detection using IQR

Q1 = df["Maths"].quantile(0.25)
Q3 = df["Maths"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# Replace Outlier with Median

median = df["Maths"].median()

# Replace outliers

df["Maths"] = np.where(
    (df["Maths"] > upper) | (df["Maths"] < lower),
    median,
print(df.describe())
----------------------------------------------------------------------------------------------------------------------------------------------------------------------
 #Assignment - 3
                      # __________________________
                       #__________________________




# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================

# ============================================================
# IMPORT REQUIRED LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import seaborn as sns

# ============================================================
# LOAD IRIS DATASET
# ============================================================

df = sns.load_dataset('iris')

print("\n================ FIRST 5 ROWS ================\n")

print(df.head())

# ============================================================
# DATASET INFORMATION
# ============================================================

print("\n================ DATASET INFO ================\n")

print(df.info())

# ============================================================
# CHECK DIMENSIONS
# ============================================================

print("\n================ DATAFRAME SHAPE ================\n")

print(df.shape)

# ============================================================
# STATISTICAL SUMMARY
# ============================================================

print("\n================ STATISTICAL SUMMARY ================\n")

print(df.describe())

# ============================================================
# GROUP BY CATEGORICAL VARIABLE
# ============================================================

grouped_data = df.groupby('species')

print("\n================ MEAN GROUPED BY SPECIES ================\n")

print(grouped_data.mean())

# ============================================================
# SUMMARY STATISTICS GROUPED BY SPECIES
# ============================================================

print("\n================ GROUPED SUMMARY STATISTICS ================\n")

print(grouped_data.agg(['mean', 'median', 'min', 'max', 'std']))

# ============================================================
# CREATE LIST OF NUMERIC VALUES
# ============================================================

setosa_list = df[df['species'] == 'setosa']['petal_length'].tolist()

versicolor_list = df[df['species'] == 'versicolor']['petal_length'].tolist()

virginica_list = df[df['species'] == 'virginica']['petal_length'].tolist()

print("\n================ SETOSA PETAL LENGTH LIST ================\n")

print(setosa_list)

print("\n================ VERSICOLOR PETAL LENGTH LIST ================\n")

print(versicolor_list)

print("\n================ VIRGINICA PETAL LENGTH LIST ================\n")

print(virginica_list)

# ============================================================
# MEAN
# ============================================================

print("\n================ MEAN ================\n")

print(df.mean(numeric_only=True))

# ============================================================
# PERCENTILES
# ============================================================

print("\n================ PERCENTILES ================\n")

print(df.quantile([0.25, 0.50, 0.75], numeric_only=True))

# ============================================================
# END OF PROGRAM
# ============================================================
----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#3
import pandas as pd
import numpy as np

# Load Dataset

df = pd.read_csv("Iris.csv")

print("First 5 Rows:\n")
print(df.head())

# Grouping

grouped = df.groupby("Species")["SepalLengthCm"]

print("\nMean:\n")
print(grouped.mean())

print("\nMedian:\n")
print(grouped.median())

print("\nMinimum:\n")
print(grouped.min())

print("\nMaximum:\n")
print(grouped.max())

print("\nStandard Deviation:\n")
print(grouped.std())

# List values

print("\nList of Sepal Length values for each Species:\n")

for species in df["Species"].unique():
    values = df[df["Species"] == species]["SepalLengthCm"].tolist()
    print(species, ":", values)

# Species-wise statistics

setosa = df[df["Species"] == "Iris-setosa"]
versicolor = df[df["Species"] == "Iris-versicolor"]
virginica = df[df["Species"] == "Iris-virginica"]

print("\nStatistics for Iris-setosa:\n")
print(setosa.describe())

print("\nStatistics for Iris-versicolor:\n")
print(versicolor.describe())

print("\nStatistics for Iris-virginica:\n")
print(virginica.describe())
-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#3
# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================

# ============================================================
# IMPORT REQUIRED LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import seaborn as sns

# ============================================================
# LOAD IRIS DATASET
# ============================================================

df = sns.load_dataset('iris')

# ============================================================
# DISPLAY FIRST 5 ROWS
# ============================================================

print("\n================ FIRST 5 ROWS ================\n")

print(df.head())

# ============================================================
# DATASET INFORMATION
# ============================================================

print("\n================ DATASET INFO ================\n")

print(df.info())

# ============================================================
# CHECK SHAPE
# ============================================================

print("\n================ SHAPE ================\n")

print(df.shape)

# ============================================================
# DESCRIBE DATASET
# ============================================================

print("\n================ STATISTICAL SUMMARY ================\n")

print(df.describe())

# ============================================================
# DATA TYPES
# ============================================================

print("\n================ DATA TYPES ================\n")

print(df.dtypes)

# ============================================================
# GROUP BY SPECIES
# ============================================================

grouped_data = df.groupby('species')

print("\n================ MEAN GROUPED BY SPECIES ================\n")

print(grouped_data.mean(numeric_only=True))

# ============================================================
# GROUPED SUMMARY STATISTICS
# ============================================================

print("\n================ GROUPED SUMMARY ================\n")

print(
    grouped_data.agg(
        ['mean', 'median', 'min', 'max', 'std']
    )
)

# ============================================================
# CREATE LISTS
# ============================================================

setosa_list = df[
    df['species'] == 'setosa'
]['petal_length'].tolist()

versicolor_list = df[
    df['species'] == 'versicolor'
]['petal_length'].tolist()

virginica_list = df[
    df['species'] == 'virginica'
]['petal_length'].tolist()

print("\n================ SETOSA LIST ================\n")

print(setosa_list)

print("\n================ VERSICOLOR LIST ================\n")

print(versicolor_list)

print("\n================ VIRGINICA LIST ================\n")

print(virginica_list)

# ============================================================
# MEAN
# ============================================================

print("\n================ MEAN ================\n")

print(df.mean(numeric_only=True))

# ============================================================
# MEDIAN
# ============================================================

print("\n================ MEDIAN ================\n")

print(df.median(numeric_only=True))

# ============================================================
# MINIMUM VALUES
# ============================================================

print("\n================ MINIMUM VALUES ================\n")

print(df.min(numeric_only=True))

# ============================================================
# MAXIMUM VALUES
# ============================================================

print("\n================ MAXIMUM VALUES ================\n")

print(df.max(numeric_only=True))

# ============================================================
# STANDARD DEVIATION
# ============================================================

print("\n================ STANDARD DEVIATION ================\n")

print(df.std(numeric_only=True))

# ============================================================
# PERCENTILES
# ============================================================

print("\n================ PERCENTILES ================\n")

print(
    df.quantile(
        [0.25, 0.50, 0.75],
        numeric_only=True
    )
)

# ============================================================
# END OF PROGRAM
# ============================================================
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Assignment - 4
                      # __________________________
                       #__________________________


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load Dataset
df = pd.read_csv(r"D:\Madhura's Projects\INSEM\DataSets\HousingData.csv")

# Handle missing values
df.fillna(df.select_dtypes(include='number').mean(), inplace=True)

# Features and Target
X = df.drop("MEDV", axis=1)
y = df["MEDV"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Output
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Assignment - 5
                      # __________________________
                       #__________________________
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

df = pd.read_csv(r"D:/Madhura's Projects/INSEM/DataSets/Social_Network_Ads.csv")

X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TP = cm[1][1]

print("Confusion Matrix:\n", cm)
print("TP:", TP)
print("FP:", FP)
print("TN:", TN)
print("FN:", FN)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Error Rate:", 1 - accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Assignment - 6
                      # __________________________
                       #__________________________

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

df = pd.read_csv(r"D:/Madhura's Projects/INSEM/DataSets/Iris.csv")

X = df.iloc[:, 1:5]
y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:\n", cm)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Error Rate:", 1 - accuracy_score(y_test, y_pred))

print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall:", recall_score(y_test, y_pred, average='macro'))
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Assignment - 7
                      # __________________________
                       #__________________________




# ============================================================
# TEXT ANALYTICS
# ============================================================

# ============================================================
# IMPORT REQUIRED LIBRARIES
# ============================================================

import nltk
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer

# ============================================================
# DOWNLOAD REQUIRED NLTK PACKAGES
# ============================================================

nltk.download('punkt')

nltk.download('punkt_tab')

nltk.download('stopwords')

nltk.download('averaged_perceptron_tagger')

nltk.download('averaged_perceptron_tagger_eng')

nltk.download('wordnet')

# ============================================================
# SAMPLE DOCUMENT
# ============================================================

document = """
Artificial Intelligence is transforming the world.
Machine Learning and Deep Learning are important parts of AI.
Text Analytics helps computers understand human language.
"""

print("\n================ ORIGINAL DOCUMENT ================\n")

print(document)

# ============================================================
# TOKENIZATION
# ============================================================

tokens = word_tokenize(document)

print("\n================ TOKENS ================\n")

for i in range(0, len(tokens), 4):
    print(" | ".join(tokens[i:i+4]))

# ============================================================
# POS TAGGING
# ============================================================

pos_tags = pos_tag(tokens)

print("\n================ POS TAGGING ================\n")

pos_df = pd.DataFrame(
    pos_tags,
    columns=["Word", "POS_Tag"]
)

print(pos_df)

# ============================================================
# STOPWORDS REMOVAL
# ============================================================

stop_words = set(stopwords.words('english'))

filtered_words = [
    word for word in tokens
    if word.lower() not in stop_words
]

print("\n================ AFTER STOPWORDS REMOVAL ================\n")

for i in range(0, len(filtered_words), 4):
    print(" | ".join(filtered_words[i:i+4]))

# ============================================================
# STEMMING
# ============================================================

ps = PorterStemmer()

stemmed_words = [
    ps.stem(word)
    for word in filtered_words
]

print("\n================ STEMMED WORDS ================\n")

stem_df = pd.DataFrame({
    "Original_Word": filtered_words,
    "Stemmed_Word": stemmed_words
})

print(stem_df)

# ============================================================
# LEMMATIZATION
# ============================================================

lemmatizer = WordNetLemmatizer()

lemmatized_words = [
    lemmatizer.lemmatize(word)
    for word in filtered_words
]

print("\n================ LEMMATIZED WORDS ================\n")

lemma_df = pd.DataFrame({
    "Original_Word": filtered_words,
    "Lemmatized_Word": lemmatized_words
})

print(lemma_df)

# ============================================================
# TF-IDF REPRESENTATION
# ============================================================

documents = [

    "Artificial Intelligence is transforming the world",

    "Machine Learning and Deep Learning are important parts of AI",

    "Text Analytics helps computers understand human language"
]

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(documents)

feature_names = vectorizer.get_feature_names_out()

tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=feature_names
)

print("\n================ TF-IDF MATRIX ================\n")

print(tfidf_df)

# ============================================================
# END OF PROGRAM
# ============================================================
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#7
import nltk
# ------------------------------------------------

stop = stopwords.words('english')

filtered_words = [w for w in words if w.lower() not in stop]

print("\nAfter Stopword Removal:")
print(filtered_words)

# ------------------------------------------------
# 4. Stemming
# ------------------------------------------------

ps = PorterStemmer()

stemmed_words = [ps.stem(w) for w in filtered_words]

print("\nStemmed Words:")
print(stemmed_words)

# ------------------------------------------------
# 5. Lemmatization
# ------------------------------------------------

lemmatizer = WordNetLemmatizer()

lemmatized_words = [lemmatizer.lemmatize(w) for w in filtered_words]

print("\nLemmatized Words:")
print(lemmatized_words)

# ------------------------------------------------
# 6. TF-IDF Representation
# ------------------------------------------------

docs = [
    "text analytics",
    "text mining",
    "useful data"
]

vectorizer = TfidfVectorizer()

result = vectorizer.fit_transform(docs)

print("\nTF-IDF Matrix:")
print(result.toarray())
------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 #Assignment - 8
                      # __________________________
                       #__________________________


# ============================================================
# DATA VISUALIZATION I
# ============================================================

# ============================================================
# IMPORT REQUIRED LIBRARIES
# ============================================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================================
# LOAD TITANIC DATASET
# ============================================================

titanic = sns.load_dataset('titanic')

print("\n================ FIRST 5 ROWS ================\n")

print(titanic.head())

# ============================================================
# DATASET INFORMATION
# ============================================================

print("\n================ DATASET SHAPE ================\n")

print(titanic.shape)

print("\n================ COLUMN NAMES ================\n")

print(titanic.columns)

print("\n================ DATA TYPES ================\n")

print(titanic.dtypes)

print("\n================ MISSING VALUES ================\n")

print(titanic.isnull().sum())

# ============================================================
# SURVIVAL COUNT
# ============================================================

plt.figure(figsize=(6,5))

sns.countplot(
    x='survived',
    data=titanic
)

plt.title("Survival Count")

plt.xlabel("Survived (0 = No, 1 = Yes)")

plt.ylabel("Number of Passengers")

plt.show()

# ============================================================
# SURVIVAL BASED ON GENDER
# ============================================================

plt.figure(figsize=(7,5))

sns.countplot(
    x='sex',
    hue='survived',
    data=titanic
)

plt.title("Survival Based on Gender")

plt.xlabel("Gender")

plt.ylabel("Count")

plt.show()

# ============================================================
# PASSENGER CLASS DISTRIBUTION
# ============================================================

plt.figure(figsize=(6,5))

sns.countplot(
    x='class',
    data=titanic
)

plt.title("Passenger Class Distribution")

plt.xlabel("Class")

plt.ylabel("Count")

plt.show()

# ============================================================
# HISTOGRAM OF FARE
# ============================================================

plt.figure(figsize=(8,5))

sns.histplot(
    titanic['fare'],
    bins=30,
    kde=True
)

plt.title("Distribution of Ticket Fare")

plt.xlabel("Fare")

plt.ylabel("Frequency")

plt.show()

# ============================================================
# END OF PROGRAM
# ============================================================
-------------------------------------------------------------------------------------------------------------------------------------------------------------------
#8
import seaborn as sns
import matplotlib.pyplot as plt

# Load Titanic Dataset

df = sns.load_dataset('titanic')

# Display first 5 rows
print(df.head())

# ------------------------------------------------
# Countplot
# ------------------------------------------------

sns.countplot(x='sex', hue='survived', data=df)

plt.title("Survival Count by Gender")

plt.show()

# ------------------------------------------------
# Histogram
# ------------------------------------------------

plt.hist(df['fare'].dropna(), bins=20)

plt.xlabel("Fare")
plt.ylabel("Passengers")
plt.title("Fare Distribution")

plt.show()
---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Assignment - 9
                      # __________________________
                       #__________________________



# ============================================================
# DATA VISUALIZATION II
# ============================================================

# ============================================================
# IMPORT REQUIRED LIBRARIES
# ============================================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================================
# LOAD TITANIC DATASET
# ============================================================

titanic = sns.load_dataset('titanic')

print("\n================ FIRST 5 ROWS ================\n")

print(titanic.head())

# ============================================================
# DATASET INFORMATION
# ============================================================

print("\n================ DATASET SHAPE ================\n")

print(titanic.shape)

print("\n================ MISSING VALUES ================\n")

print(titanic[['sex', 'age', 'survived']].isnull().sum())

# ============================================================
# BOX PLOT
# ============================================================

plt.figure(figsize=(10,6))

sns.boxplot(
    x='sex',
    y='age',
    hue='survived',
    data=titanic
)

plt.title("Age Distribution by Gender and Survival")

plt.xlabel("Gender")

plt.ylabel("Age")

plt.show()

# ============================================================
# OBSERVATIONS
# ============================================================

print("\n================ OBSERVATIONS ================\n")

print("1. Female passengers had a higher survival rate than male passengers.")

print("\n2. Most male passengers who did not survive were between age 20 to 40.")

print("\n3. Children had comparatively better survival chances.")

print("\n4. Median age of passengers is approximately between 25 and 35 years.")

print("\n5. Presence of outliers indicates some passengers were very old or very young.")

print("\n6. Female survivors show a more concentrated age distribution.")

# ============================================================
# END OF PROGRAM
# ============================================================
---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#9
import seaborn as sns
import matplotlib.pyplot as plt

# Load Titanic Dataset
df = sns.load_dataset('titanic')

# Display first 5 rows
print(df.head())

# Box Plot
sns.boxplot(x='sex', y='age', hue='survived', data=df)

# Title
plt.title("Age Distribution by Gender and Survival")

# Show Plot
plt.show()
-------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Assignment - 10
                      # __________________________
                       #__________________________


# ============================================================
# DATA VISUALIZATION III
# ============================================================

# ============================================================
# IMPORT REQUIRED LIBRARIES
# ============================================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================================
# LOAD IRIS DATASET
# ============================================================

iris = sns.load_dataset('iris')

print("\n================ FIRST 5 ROWS ================\n")

print(iris.head())

# ============================================================
# DATASET INFORMATION
# ============================================================

print("\n================ DATASET SHAPE ================\n")

print(iris.shape)

print("\n================ COLUMN NAMES ================\n")

print(iris.columns)

print("\n================ DATA TYPES ================\n")

print(iris.dtypes)

# ============================================================
# FEATURES AND TYPES
# ============================================================

print("\n================ FEATURES AND TYPES ================\n")

for column in iris.columns:
    
    if iris[column].dtype == 'object':
        
        print(column, " --> Nominal / Categorical")
    
    else:
        
        print(column, " --> Numeric")

# ============================================================
# HISTOGRAMS
# ============================================================

iris.hist(
    figsize=(10,8),
    bins=20
)

plt.suptitle("Histograms of Iris Dataset Features")

plt.show()

# ============================================================
# BOXPLOTS
# ============================================================

plt.figure(figsize=(12,6))

sns.boxplot(data=iris)

plt.title("Boxplots of Iris Dataset Features")

plt.show()

# ============================================================
# INDIVIDUAL BOXPLOTS
# ============================================================

numeric_columns = iris.select_dtypes(include=['float64']).columns

for column in numeric_columns:
    
    plt.figure(figsize=(6,4))
    
    sns.boxplot(
        y=iris[column]
    )
    
    plt.title(f"Boxplot of {column}")
    
    plt.show()

# ============================================================
# OBSERVATIONS
# ============================================================

print("\n================ OBSERVATIONS ================\n")

print("1. All flower measurement features are numeric.")

print("\n2. Species column is categorical (nominal).")

print("\n3. Histograms show that sepal and petal measurements are distributed differently.")

print("\n4. Petal length and petal width show wider variation compared to sepal features.")

print("\n5. Boxplots help identify outliers in some features.")

print("\n6. Sepal width contains a few outliers.")

print("\n7. Most data values are concentrated within the interquartile range.")

print("\n8. Iris dataset is comparatively clean with very few extreme outliers.")

# ============================================================
# END OF PROGRAM
# ============================================================
-------------------------------------------------------------------------------------------------------------------------------------------------------------------
#10
import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("Iris.csv")

# Display first 5 rows
print(df.head())

# Display Data Types
print("\nFeature Types:\n")
print(df.dtypes)

# ------------------------------------------------
# Histograms
# ------------------------------------------------

df.hist(figsize=(10, 8))

plt.suptitle("Histograms of Iris Dataset Features")

plt.show()

# ------------------------------------------------
# Boxplots
# ------------------------------------------------

df.plot(kind='box',
        subplots=True,
        layout=(3,2),
        figsize=(10,8))

plt.suptitle("Boxplots of Iris Dataset Features")

plt.show()
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------



























































