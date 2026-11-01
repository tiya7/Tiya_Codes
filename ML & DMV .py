# ML 1 Feature Transformation (Any one)
A. To use PCA Algorithm for dimensionality reduction.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load dataset
data = pd.read_csv("Wine.csv")

print(data.head())
print(data.info())

# Separate features and target
X = data.drop("Customer_Segment", axis=1)
y = data["Customer_Segment"]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create new dataframe
pca_data = pd.DataFrame(
    X_pca, columns=["PC1", "PC2"]
)
pca_data["Class"] = y.values

print("\nExplained Variance:")
print(pca.explained_variance_ratio_)

print("\nTotal Variance Captured:")
print(pca.explained_variance_ratio_.sum() * 100, "%")

# Visualize PCA
for group in sorted(pca_data["Class"].unique()):
    subset = pca_data[pca_data["Class"] == group]
    plt.scatter(
        subset["PC1"], subset["PC2"],
        label="Class " + str(group)
    )

plt.title("PCA on Wine Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()

# Save output
pca_data.to_csv("PCA_Output.csv", index=False)

#OR#

# ML 1

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = pd.read_csv("https://media.geeksforgeeks.org/wp-content/uploads/Wine.csv")

X = data.drop("Customer_Segment", axis=1)
y = data["Customer_Segment"]

X = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
result = pca.fit_transform(X)

print("PCA Result:")
print(result[:5])

print("Variance captured:",
      sum(pca.explained_variance_ratio_) * 100, "%")

plt.scatter(result[:, 0], result[:, 1], c=y)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA on Wine Dataset")
plt.show()
---------------------------------------------------------------------------------------------
# ML 2 Regression Analysis:(Any one)
A. Predict the price of the Uber ride from a given pickup point to the agreed drop-off 
location. 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load dataset
data = pd.read_csv("uber.csv")

print(data.head())
print(data.info())

# Remove unnecessary columns and duplicates
data = data.drop(columns=["Unnamed: 0", "key"], errors="ignore")
data = data.drop_duplicates()

# Convert date and numerical columns
data["pickup_datetime"] = pd.to_datetime(
    data["pickup_datetime"], errors="coerce", utc=True
)

cols = [
    "fare_amount", "pickup_longitude", "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude", "passenger_count"
]

for col in cols:
    data[col] = pd.to_numeric(data[col], errors="coerce")

data = data.dropna()

# Remove invalid values
data = data[
    (data["fare_amount"] > 0) &
    (data["fare_amount"] <= 100) &
    (data["passenger_count"] >= 1) &
    (data["passenger_count"] <= 6) &
    (data["pickup_longitude"].between(-75, -72)) &
    (data["dropoff_longitude"].between(-75, -72)) &
    (data["pickup_latitude"].between(40, 42)) &
    (data["dropoff_latitude"].between(40, 42))
].copy()

# Identify outliers using IQR
Q1 = data["fare_amount"].quantile(0.25)
Q3 = data["fare_amount"].quantile(0.75)
IQR = Q3 - Q1

lower = max(0, Q1 - 1.5 * IQR)
upper = Q3 + 1.5 * IQR

outliers = data[
    (data["fare_amount"] < lower) |
    (data["fare_amount"] > upper)
]

print("Number of outliers:", len(outliers))

# Remove outliers
data = data[
    (data["fare_amount"] >= lower) &
    (data["fare_amount"] <= upper)
].copy()

# Calculate ride distance using Haversine formula
lat1 = np.radians(data["pickup_latitude"])
lat2 = np.radians(data["dropoff_latitude"])
lon1 = np.radians(data["pickup_longitude"])
lon2 = np.radians(data["dropoff_longitude"])

dlat = lat2 - lat1
dlon = lon2 - lon1

a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2

data["distance"] = 6371 * 2 * np.arcsin(np.sqrt(a))

# Extract time features
data["hour"] = data["pickup_datetime"].dt.hour
data["day"] = data["pickup_datetime"].dt.day
data["month"] = data["pickup_datetime"].dt.month

# Check correlation
features = [
    "fare_amount", "distance", "passenger_count",
    "hour", "day", "month"
]

print("\nCorrelation:")
print(data[features].corr())

sns.heatmap(data[features].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Split data into input and output
X = data[[
    "distance", "passenger_count", "hour", "day", "month"
]]

y = data["fare_amount"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create regression models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.01, max_iter=10000)
}

# Train and evaluate models
for name, model in models.items():

    model.fit(X_train, y_train)
    prediction = model.predict(X_test)

    r2 = r2_score(y_test, prediction)
    rmse = np.sqrt(mean_squared_error(y_test, prediction))
    mae = mean_absolute_error(y_test, prediction)

    print("\n", name)
    print("R2 Score:", r2)
    print("RMSE:", rmse)
    print("MAE:", mae)

#OR#

# ML 2

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error

# Load data
data = pd.read_csv(r"D:\Insem\uber.csv")
data = data.dropna()
data = data[data["fare_amount"] > 0]

# Identify outliers
Q1 = data["fare_amount"].quantile(0.25)
Q3 = data["fare_amount"].quantile(0.75)
IQR = Q3 - Q1
data = data[(data["fare_amount"] >= Q1 - 1.5 * IQR) &
            (data["fare_amount"] <= Q3 + 1.5 * IQR)]

print("Correlation:")
print(data[["fare_amount", "passenger_count"]].corr())

# Select input and output
X = data[["pickup_longitude", "pickup_latitude",
          "dropoff_longitude", "dropoff_latitude",
          "passenger_count"]]
y = data["fare_amount"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Apply models
for name, model in [
    ("Linear Regression", LinearRegression()),
    ("Ridge Regression", Ridge()),
    ("Lasso Regression", Lasso())
]:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print("\n", name)
    print("R2 Score:", r2_score(y_test, pred))
    print("RMSE:", mean_squared_error(y_test, pred) ** 0.5)
 ---------------------------------------------------------------------------------------------
# ML 3 Classification Analysis (Any one)
A. Implementation of Support Vector Machines (SVM) for classifying images of hand-
written digits into their respective numerical classes (0 to 9).

                                                        
# 1. Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 2. Load handwritten digits dataset
digits = load_digits()

X = digits.data
y = digits.target

# 3. Display all digits from 0 to 9
fig, axes = plt.subplots(2, 5, figsize=(10, 5))

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='gray')
    ax.set_title("Digit: " + str(digits.target[i]))
    ax.axis('off')

plt.show()

# 4. Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Create and train SVM model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# 6. Predict digits
y_pred = model.predict(X_test)

# 7. Calculate accuracy
print("Accuracy:", accuracy_score(y_test, y_pred) * 100)

# 8. Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 9. Display confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 10. Display actual and predicted digits
fig, axes = plt.subplots(2, 5, figsize=(10, 5))

for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(8, 8), cmap='gray')
    ax.set_title("Actual: " + str(y_test[i]) +
                 "\nPredicted: " + str(y_pred[i]))
    ax.axis('off')

plt.tight_layout()
plt.show()

#OR#

import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load handwritten digits dataset
digits = load_digits()

X = digits.data
y = digits.target

print("Dataset Shape:", X.shape)

# Display sample images
fig, axes = plt.subplots(1, 5, figsize=(10, 3))

for ax, image, label in zip(axes, digits.images[:5], y[:5]):
    ax.imshow(image, cmap="gray")
    ax.set_title("Digit: " + str(label))
    ax.axis("off")

plt.show()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create SVM model
model = SVC(kernel="rbf")

# Train model
model.fit(X_train, y_train)

# Predict digits
y_pred = model.predict(X_test)

# Evaluate model
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#OR#

#ML 3

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
data = load_digits()
X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train SVM model
model = SVC()
model.fit(X_train, y_train)

# Predict digits
pred = model.predict(X_test)

# Display result
print("Actual Digits:", y_test[:10])
print("Predicted Digits:", pred[:10])
print("Accuracy:", accuracy_score(y_test, pred) * 100, "%")
---------------------------------------------------------------------------------------------
# ML 4 Clustering Analysis (Any one)
A. Implement K-Means clustering on Iris.csv dataset. Determine the number of clusters 
using the elbow method.

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_csv("Iris.csv")

print(data.head())

# Remove ID and Species columns
X = data.drop(columns=["Id", "Species"], errors="ignore")

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method
wcss = []

for k in range(1, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)
    wcss.append(model.inertia_)

# Plot Elbow graph
plt.plot(range(1, 11), wcss, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Apply K-Means with 3 clusters
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

data["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nCluster Results:")
print(data[["Species", "Cluster"]].head(10))

print("\nCluster Counts:")
print(data["Cluster"].value_counts())

# Visualize clusters
plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=data["Cluster"],
    cmap="viridis"
)

plt.title("K-Means Clustering")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()

# Save results
data.to_csv("Iris_Clustered.csv", index=False)

#OR#

#ML 4

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# Load dataset
data = load_iris()
X = data.data

# Elbow Method
wcss = []

for i in range(1, 11):
    model = KMeans(n_clusters=i, random_state=42, n_init=10)
    model.fit(X)
    wcss.append(model.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# K-Means clustering
model = KMeans(n_clusters=3, random_state=42, n_init=10)
model.fit(X)

print("Cluster Labels:")
print(model.labels_)
---------------------------------------------------------------------------------------------
# ML 5 Ensemble Learning (Any one)
A. Implement Random Forest Classifier model to predict the safety of the car.

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
columns = [
    "buying", "maint", "doors",
    "persons", "lug_boot", "safety", "class"
]

data = pd.read_csv(
    "car_evaluation.csv",
    header=None,
    names=columns
)

print(data.head())

# Separate features and target
X = data.drop("class", axis=1)
y = data["class"]

# Convert categorical data into numerical data
for col in X.columns:
    encoder = LabelEncoder()
    X[col] = encoder.fit_transform(X[col])

# Encode target variable
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predict output
y_pred = model.predict(X_test)

# Evaluate model
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=target_encoder.classes_
))

# Feature importance
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

importance.plot(kind="bar")
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()

#OR#

# ML 5

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv(r"D:\Insem\car_evaluation.csv", header=None)
data.columns = ["buying", "maint", "doors", "persons",
                "lug_boot", "safety", "class"]

# Convert text into numbers
encoder = LabelEncoder()

for col in data.columns:
    data[col] = encoder.fit_transform(data[col])

# Split data
X = data.drop("class", axis=1)
y = data["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict and check accuracy
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred) * 100, "%")
print("Predictions:", pred[:10])
---------------------------------------------------------------------------------------------
# ML 6 A Maze

import numpy as np
import random

# Maze: 0 = Path, 1 = Wall, 2 = Goal
maze = np.array([
    [0, 0, 0, 1, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 2]
])

start = (0, 0)
goal = (4, 4)

rows, cols = maze.shape

# Actions: Up, Down, Left, Right
actions = [
    (-1, 0), (1, 0), (0, -1), (0, 1)
]

Q = np.zeros((rows * cols, 4))

alpha = 0.1
gamma = 0.9
epsilon = 0.2

episodes = 3000

def get_state(position):
    return position[0] * cols + position[1]

def valid_moves(position):
    moves = []

    for i, (dr, dc) in enumerate(actions):

        nr = position[0] + dr
        nc = position[1] + dc

        if (0 <= nr < rows and
            0 <= nc < cols and maze[nr, nc] != 1):

            moves.append(i)

    return moves

# Training
for episode in range(episodes):

    position = start

    for step in range(100):

        state = get_state(position)
        possible = valid_moves(position)

        if random.random() < epsilon:
            action = random.choice(possible)
        else:
            action = possible[
                np.argmax(Q[state, possible])
            ]

        dr, dc = actions[action]

        next_position = (
            position[0] + dr,
            position[1] + dc
        )

        if next_position == goal:
            reward = 100
            done = True
        else:
            reward = -1
            done = False

        next_state = get_state(next_position)

        if done:
            target = reward
        else:
            next_moves = valid_moves(next_position)
            target = reward + gamma * np.max(
                Q[next_state, next_moves]
            )

        Q[state, action] += alpha * (
            target - Q[state, action]
        )

        position = next_position

        if done:
            break

# Testing
position = start
path = [position]

for step in range(50):

    state = get_state(position)
    possible = valid_moves(position)

    action = possible[
        np.argmax(Q[state, possible])
    ]

    dr, dc = actions[action]

    position = (
        position[0] + dr,
        position[1] + dc
    )

    path.append(position)

    if position == goal:
        break

print("Learned Path:", path)
print("Goal Reached:", position == goal)

#OR#

# ML 6 C Tic Tac Toe

import random

# Q-table
Q = {}

alpha = 0.2
gamma = 0.9
epsilon = 1.0
episodes = 30000

# Find available positions
def available_moves(board):
    return [i for i in range(9) if board[i] == 0]

# Check winner
def check_winner(board):

    lines = [
        (0,1,2), (3,4,5), (6,7,8),
        (0,3,6), (1,4,7), (2,5,8),
        (0,4,8), (2,4,6)
    ]

    for a, b, c in lines:

        if board[a] == board[b] == board[c] != 0:
            return board[a]

    if 0 not in board:
        return 0

    return None

# Get Q-values
def get_q(state):

    if state not in Q:
        Q[state] = [0.0] * 9

    return Q[state]

# Choose action
def choose_action(board, explore=True):

    moves = available_moves(board)
    state = tuple(board)

    if explore and random.random() < epsilon:
        return random.choice(moves)

    values = get_q(state)

    return max(moves, key=lambda move: values[move])

# Training
for episode in range(episodes):

    board = [0] * 9
    game_over = False

    while not game_over:

        state = tuple(board)

        action = choose_action(board)
        board[action] = 1

        result = check_winner(board)

        if result == 1:
            reward = 1
            done = True

        elif result == 0:
            reward = 0
            done = True

        else:
            # Opponent plays randomly
            opponent = random.choice(available_moves(board))
            board[opponent] = -1

            result = check_winner(board)

            if result == -1:
                reward = -1
                done = True

            elif result == 0:
                reward = 0
                done = True

            else:
                reward = 0
                done = False

        old_values = get_q(state)

        if done:
            target = reward
        else:
            next_state = tuple(board)
            next_moves = available_moves(board)
            next_values = get_q(next_state)

            target = reward + gamma * max(
                next_values[m] for m in next_moves
            )

        old_values[action] += alpha * (
            target - old_values[action]
        )

        game_over = done

    epsilon = max(0.05, epsilon * 0.999)

# Testing
board = [0] * 9

print("\nTic-Tac-Toe Testing:")

while True:

    action = choose_action(board, explore=False)
    board[action] = 1

    result = check_winner(board)

    if result is not None:
        break

    opponent = random.choice(available_moves(board))
    board[opponent] = -1

    result = check_winner(board)

    if result is not None:
        break

# Display board
for i in range(0, 9, 3):
    print(board[i:i+3])

if result == 1:
    print("Agent Wins!")

elif result == -1:
    print("Opponent Wins!")

else:
    print("Draw!")

#OR#

# ML 6 C

import random

# Maze: 0 = path, 1 = wall
maze = [
    [0, 0, 0],
    [1, 1, 0],
    [0, 0, 0]
]

position = 0
goal = 8

# Agent explores the maze
while position != goal:
    move = random.choice([1, 3])  # Right or Down
    next_position = position + move

    if next_position < 9:
        row = next_position // 3
        col = next_position % 3

        if maze[row][col] == 0:
            position = next_position

    print("Agent position:", position)

print("Goal reached!")
---------------------------------------------------------------------------------------------
Viva questions

1. Basic Machine Learning Viva Questions
Question	Easy answer
1. What is Machine Learning?	Machine Learning is a branch of AI that allows computers to learn from data and make predictions.
2. What is Artificial Intelligence?	AI is the ability of machines to perform tasks that normally require human intelligence.
3. What is the difference between AI and ML?	AI is the broader concept. ML is a subset of AI that learns from data.
4. What are the types of ML?	Supervised, Unsupervised and Reinforcement Learning.
5. What is supervised learning?	Learning using labelled data, where both input and correct output are provided.
6. What is unsupervised learning?	Learning from unlabelled data to find hidden patterns.
7. What is reinforcement learning?	Learning through rewards and penalties based on actions.
8. What is a dataset?	A collection of data used for training and testing a model.
9. What is a feature?	An input variable used to make predictions.
10. What is a label?	The output or target variable that the model predicts.
11. What is model training?	The process of teaching a model using data.
12. What is model testing?	Checking how well the trained model performs on unseen data.
13. What is overfitting?	When a model performs well on training data but poorly on new data.
14. What is underfitting?	When a model is too simple to learn the patterns in data.
15. What is accuracy?	The percentage of correct predictions made by a model.

2. Experiment-wise Viva Questions
Experiment 1: Principal Component Analysis (PCA)
Unsupervised learning
Question	Easy answer
What is PCA?	PCA is a dimensionality reduction technique used to reduce the number of features.
Why do we use PCA?	To reduce complexity, remove redundancy and make data easier to visualize.
Is PCA supervised or unsupervised?	Unsupervised learning.
What is dimensionality reduction?	Reducing the number of input features while preserving important information.
What is a principal component?	A new variable that captures maximum variation in the data.
What is variance?	A measure of how much data values are spread out.
Why is standardization required in PCA?	To bring features to a common scale.
What is explained variance?	The amount of information or variation retained by a principal component.
Which dataset did we use?	Wine.csv
What is the output of PCA?	Reduced-dimensional data, such as two principal components.

Important theory questions
1. Explain PCA and its applications.
2. What are the advantages of dimensionality reduction?
3. Explain the steps involved in PCA.
4. Differentiate between PCA and feature selection.
Answers : 
Q1. Explain PCA and its applications.
Answer:
- PCA stands for Principal Component Analysis.
- It is a dimensionality reduction technique.
- It reduces the number of features in a dataset.
- It keeps the most important information from the original data.
- It makes data easier to understand and process.
Example: If a dataset has 10 columns, PCA can reduce them to 2 or 3 important components.

Applications:
1. Image processing.
2. Data visualization.
3. Face recognition.
4. Medical data analysis.
5. Machine learning.
Q2. What are the advantages of dimensionality reduction?
Answer:

Dimensionality reduction means reducing the number of features in a dataset.

Advantages:
1. Reduces data complexity.
2. Saves memory and storage.
3. Reduces training time.
4. Removes unnecessary information.
5. Helps in data visualization.
6. Can reduce overfitting.
Q3. Explain the steps involved in PCA.
Answer:

The steps involved in PCA are:
1. Collect data: Take the required dataset.
2. Standardization: Convert data into a common scale.
3. Calculate covariance: Find the relationship between features.
4. Calculate eigenvalues and eigenvectors: Find important directions in data.
5. Select components: Choose components with higher eigenvalues.
6. Transform data: Convert original data into new principal components.
\(\hat{y} = b_0 + b_1x\)
Q4. Differentiate between PCA and Feature Selection.
PCA	Feature Selection
Reduces dimensions by creating new features.	Selects important existing features.
Combines original features.	Keeps original features.
New components are created.	No new features are created.
May reduce interpretability.	Easy to understand.
Example: PCA on 10 columns to get 2 components.	Example: Selecting 2 important columns from 10.

Experiment 2: Uber Fare Prediction using Regression
Supervised learning
Question	Easy answer
What is regression?	A technique used to predict continuous numerical values.
What is Linear Regression?	It predicts output using a straight-line relationship between input and output.
What is Ridge Regression?	Regression that uses L2 regularization to reduce overfitting.
What is Lasso Regression?	Regression that uses L1 regularization and can reduce some coefficients to zero.
What is the difference between Ridge and Lasso?	Ridge shrinks coefficients. Lasso can make coefficients zero.
What is the dependent variable?	The variable we want to predict.
What is the independent variable?	The input variable used for prediction.
What is RMSE?	Root Mean Squared Error. It measures prediction error.
What is MAE?	Mean Absolute Error. It measures average absolute prediction error.
What is R² score?	It indicates how much variation in the target is explained by the model.
What is outlier detection?	Finding unusually high or low values in a dataset.
Why do we calculate distance in Uber data?	To estimate the journey distance between pickup and drop-off locations.

Important theory questions
1. Explain Linear Regression with an example.
2. Differentiate between Linear, Ridge and Lasso Regression.
3. Explain MAE, MSE, RMSE and R².
4. What is regularization? Explain L1 and L2.
5. Explain data preprocessing in fare prediction.
Answers:
Q1. Explain Linear Regression with an example.
Answer:
- Linear Regression is a supervised machine learning algorithm.
- It is used to predict continuous numerical values.
- It finds the relationship between input and output variables.
- It uses a straight line to make predictions.
Example: Predicting taxi fare based on distance travelled.

Formula:
\[
y = mx+c
\]

Where:
- y = Predicted output
- m = Slope
- x = Input value
- c = Intercept
\(\hat{y} = b_0 + b_1x\)
Q2. Differentiate between Linear, Ridge and Lasso Regression.
Linear Regression	Ridge Regression	Lasso Regression
Basic regression algorithm.	Uses L2 regularization.	Uses L1 regularization.
No penalty is added.	Adds penalty to squared coefficients.	Adds penalty to absolute coefficients.
May overfit.	Helps reduce overfitting.	Helps reduce overfitting.
Keeps all features.	Keeps features but reduces coefficients.	Can make some coefficients zero.

Q3. Explain MAE, MSE, RMSE and R².
Answer:

These are evaluation metrics used to check the performance of a regression model.

1. MAE (Mean Absolute Error):
- Calculates the average absolute difference between actual and predicted values.
- Lower MAE means better prediction.
2. MSE (Mean Squared Error):
- Calculates the average squared difference between actual and predicted values.
- Penalizes large errors more.
3. RMSE (Root Mean Squared Error):
- It is the square root of MSE.
- It measures prediction error in the original unit.
4. R² Score (R-squared):
- Measures how well the model explains the data.
- A value closer to 1 generally indicates a better fit.
Q4. What is Regularization? Explain L1 and L2.
Answer:
- Regularization is a technique used to prevent overfitting.
- It adds a penalty to the model to control large coefficients.
- It helps the model perform better on new data.
Types:

1. L1 Regularization (Lasso):
- Adds the absolute values of coefficients as a penalty.
- Can make some coefficients zero.
- Helps in feature selection.
2. L2 Regularization (Ridge):
- Adds the squared values of coefficients as a penalty.
- Reduces coefficient values.
- Usually does not make them exactly zero.
Q5. Explain data preprocessing in fare prediction.
Answer:

Data preprocessing means cleaning and preparing data before applying a machine learning algorithm.

Steps:
1. Load the fare dataset.
2. Remove duplicate records.
3. Handle missing values.
4. Remove incorrect or unwanted values.
5. Convert date and time into proper format.
6. Select important features like distance and passenger count.
7. Split data into training and testing sets.
8. Train the regression model.
Example: Removing records with negative fare amounts or incorrect distances.


Experiment 3: SVM Handwritten Digit Classification
Supervised learning
Question	Easy answer
What is SVM?	Support Vector Machine is an algorithm used for classification and regression.
What is classification?	Predicting a category or class.
What is a hyperplane?	A decision boundary that separates different classes.
What are support vectors?	Data points closest to the decision boundary.
What is a kernel?	A function used to handle data that cannot be separated linearly.
Name different SVM kernels.	Linear, Polynomial, RBF and Sigmoid.
Which kernel did we use?	RBF kernel.
What is RBF?	Radial Basis Function.
What is a confusion matrix?	A table that compares actual and predicted classes.
What is precision?	Out of all predicted positive cases, how many were actually positive.
What is recall?	Out of all actual positive cases, how many were correctly identified.
Which dataset did we use?	Scikit-learn's built-in Digits dataset.

Important theory questions
1. Explain the working of SVM.
2. Explain the different types of SVM kernels.
3. What are hyperplanes and support vectors?
4. Explain confusion matrix, precision, recall and F1-score.
5. Differentiate between classification and regression.
Answers:
Q1. Explain the working of SVM.
Answer:
- SVM stands for Support Vector Machine.
- It is a supervised machine learning algorithm.
- It is mainly used for classification.
- It separates data into different classes using a decision boundary.
- It tries to find the best boundary with the maximum margin.
Example: Classifying emails as Spam or Not Spam.







mdpi.com


Q2. Explain the different types of SVM kernels.
Answer:

A kernel is used to separate data when a straight line cannot separate it properly.

Types of kernels:
1. Linear Kernel: Used when data can be separated using a straight line.
2. Polynomial Kernel: Used for curved and more complex relationships.
3. RBF Kernel: Used for non-linear data. It is commonly used.
4. Sigmoid Kernel: Works similarly to the activation function used in neural networks.
Q3. What are Hyperplanes and Support Vectors?
Answer:

Hyperplane:
- A hyperplane is a decision boundary that separates different classes.
- In 2D, it is a line.
- In 3D, it is a plane.
Support Vectors:
- Support vectors are the nearest data points to the hyperplane.
- They help decide the position of the hyperplane.
- They are very important in SVM.







medium.com


Q4. Explain Confusion Matrix, Precision, Recall and F1-score.
Answer:

These are evaluation metrics used to measure classification model performance.

1. Confusion Matrix:

It shows the correct and incorrect predictions made by a model.
Term	Meaning
True Positive (TP)	Correctly predicted positive
True Negative (TN)	Correctly predicted negative
False Positive (FP)	Incorrectly predicted positive
False Negative (FN)	Incorrectly predicted negative

2. Precision:

It tells how many predicted positive values are actually positive.
\[
Precision=\frac{TP}{TP+FP}
\]

3. Recall:

It tells how many actual positive values are correctly identified.
\[
Recall=\frac{TP}{TP+FN}
\]

4. F1-score:

It is the balance between precision and recall.
\[
F1=2\times\frac{Precision\times Recall}{Precision+Recall}
\]

Q5. Differentiate between Classification and Regression.
Classification	Regression
Predicts categories.	Predicts numerical values.
Output is discrete.	Output is continuous.
Used for class prediction.	Used for value prediction.
Example: Spam or Not Spam.	Example: House price prediction.
Algorithms: SVM, KNN.	Algorithms: Linear Regression, Ridge.


Experiment 4: K-Means Clustering on Iris Dataset
Unsupervised learning
Question	Easy answer
What is clustering?	Grouping similar data points together.
What is K-Means?	An unsupervised algorithm that divides data into K clusters.
What does K represent?	Number of clusters.
What is a centroid?	The centre of a cluster.
What is the elbow method?	A method used to find the optimal number of clusters.
What is WCSS?	Within-Cluster Sum of Squares. It measures the spread of points within clusters.
What is the purpose of scaling?	To prevent large-valued features from dominating distance calculations.
Is K-Means supervised?	No, it is unsupervised.
Which distance is commonly used?	Euclidean distance.
Which dataset did we use?	Iris.csv
How many clusters did we select?	Three clusters.

Important theory questions
1. Explain the K-Means clustering algorithm.
2. Explain the elbow method with a diagram.
3. Differentiate between supervised and unsupervised learning.
4. What are the advantages and limitations of K-Means?
5. Explain the role of centroids in clustering.
Answers:
Q1. Explain the K-Means clustering algorithm.
Answer:
- K-Means is an unsupervised machine learning algorithm.
- It is used to group similar data points into clusters.
- K represents the number of clusters.
- Each cluster has a centre called a centroid.
- Data points are grouped based on their distance from the centroid.
Steps:
1. Select the number of clusters (K).
2. Initialize K centroids.
3. Calculate the distance of each point from the centroids.
4. Assign each point to its nearest centroid.
5. Recalculate the centroids.
6. Repeat until the clusters become stable.







medium.com


Q2. Explain the Elbow Method with a diagram.
Answer:
- The Elbow Method is used to find the best value of K.
- It helps to select the optimal number of clusters.
- It calculates WCSS (Within-Cluster Sum of Squares) for different K values.
- We draw a graph between K values and WCSS.
- The point where the graph looks like an elbow is selected as the best K.







ceve543.github.io


Q3. Differentiate between Supervised and Unsupervised Learning.
Supervised Learning	Unsupervised Learning
Uses labelled data.	Uses unlabelled data.
Output is already known during training.	Output is not given.
Used for prediction.	Used for finding patterns.
Example: Classification.	Example: Clustering.
Algorithms: SVM, Linear Regression.	Algorithms: K-Means, Hierarchical Clustering.

Q4. What are the advantages and limitations of K-Means?
Advantages:
1. Easy to understand.
2. Simple to implement.
3. Fast for large datasets.
4. Easy to identify clusters.
5. Works well for clearly separated groups.
Limitations:
1. We must select K manually.
2. Sensitive to outliers.
3. Different initial centroids may give different results.
4. Not suitable for all types of cluster shapes.
5. Requires numerical data.
Q5. Explain the role of Centroids in Clustering.
Answer:
- A centroid is the centre point of a cluster.
- It represents all the data points in that cluster.
- Every data point is assigned to its nearest centroid.
- Centroids are updated by calculating the mean of the points in each cluster.
- They help form and improve clusters.
Example: If we divide students into 3 groups based on marks, each group will have one centroid.


Experiment 5: Random Forest for Car Safety Classification
Supervised learning
Question	Easy answer
What is Random Forest?	An ensemble learning algorithm that combines multiple decision trees.
What is a decision tree?	A tree-like model that makes decisions based on conditions.
What is ensemble learning?	Combining multiple models to improve prediction performance.
What is bagging?	Training multiple models on different random samples of data.
How does Random Forest predict?	It uses majority voting for classification.
What is a root node?	The starting node of a decision tree.
What is a leaf node?	The final node that gives the prediction.
What is feature importance?	It indicates how useful a feature is in making predictions.
What is label encoding?	Converting categorical values into numerical values.
Which dataset did we use?	car_evaluation.csv
What are the output classes?	Unacceptable, acceptable, good and very good.

Important theory questions
1. Explain the working of Random Forest.
2. Differentiate between Decision Tree and Random Forest.
3. Explain ensemble learning.
4. What are the advantages of Random Forest?
5. What is bagging and how does it reduce overfitting?
Answers:
Q1. Explain the working of Random Forest.
Answer:
- Random Forest is a supervised machine learning algorithm.
- It uses multiple Decision Trees to make predictions.
- Each tree is trained on a random sample of data.
- Each tree gives its own prediction.
- The final output is decided by combining all tree predictions.
For classification: Majority voting is used.

For regression: Average prediction is calculated.







lightning.ai


Q2. Differentiate between Decision Tree and Random Forest.
Decision Tree	Random Forest
Uses a single tree.	Uses multiple trees.
Easy to understand.	More complex.
Can easily overfit.	Reduces overfitting.
May give less stable results.	Usually gives more stable results.
Faster to train.	Usually takes more training time.

Q3. Explain Ensemble Learning.
Answer:
- Ensemble Learning is a technique that combines multiple machine learning models.
- It improves prediction performance.
- It gives better and more stable results than using a single model in many cases.
- Random Forest is an example of ensemble learning.
Example: Combining the predictions of 10 Decision Trees to get one final prediction.
Q4. What are the advantages of Random Forest?
Answer:
1. Gives good prediction accuracy.
2. Reduces overfitting compared to a single Decision Tree.
3. Works for classification and regression.
4. Handles large datasets.
5. Can identify important features.
6. Works with many input features.
Q5. What is Bagging and how does it reduce overfitting?
Answer:
- Bagging stands for Bootstrap Aggregating.
- It creates multiple random samples from the original dataset.
- Different models are trained on these samples.
- Their predictions are combined using voting or averaging.
- It reduces variance and helps prevent overfitting.
Example: Random Forest uses Bagging to train multiple Decision Trees.


Experiment 6: Reinforcement Learning
Reinforcement learning

Question	Easy answer
What is Reinforcement Learning?	Learning by interacting with an environment and receiving rewards or penalties.
What is an agent?	The learner or decision-maker.
What is an environment?	The world in which the agent operates.
What is an action?	A move or decision taken by the agent.
What is a reward?	Feedback received after taking an action.
What is a state?	The current situation of the agent.
What is a policy?	A strategy that tells the agent which action to take.
What is Q-Learning?	A reinforcement learning algorithm that learns the value of actions in different states.
What is a Q-table?	A table that stores Q-values for state-action pairs.
What is exploration?	Trying new actions to discover better rewards.
What is exploitation?	Choosing the best-known action.
What is epsilon?	A parameter controlling exploration in epsilon-greedy learning.
What is a terminal state?	A state where an episode ends.

A. Maze problem
Question	Easy answer
What is the goal in the maze?	Reach the destination from the starting point.
What are the actions?	Up, down, left and right.
What is a wall?	An obstacle that the agent cannot cross.
What reward is given for reaching the goal?	A positive reward.
Why does the agent receive a penalty for each step?	To encourage it to find a shorter path.

B. Taxi problem
Question	Easy answer
What is Taxi-v3?	A built-in Gymnasium environment for reinforcement learning.
What is the goal?	Pick up a passenger and drop them at the correct destination.
What are the main actions?	Move, pick up and drop off.
What is Gymnasium?	A Python library for creating and interacting with RL environments.
What happens after a successful drop-off?	The agent receives a positive reward and the episode ends.

C. Tic-Tac-Toe problem
Question	Easy answer
What is Tic-Tac-Toe?	A two-player game played on a 3×3 board.
What are the symbols?	X and O.
How does the agent win?	By placing three symbols in a row, column or diagonal.
What is a draw?	When all cells are filled and nobody wins.
How does Q-Learning help?	It learns which moves lead to winning outcomes.

Important theory questions
1. Explain the Reinforcement Learning architecture.
2. Explain the Q-Learning algorithm.
3. Differentiate between exploration and exploitation.
4. Explain the reward and penalty mechanism.
5. Explain the applications of Reinforcement Learning.
6. Differentiate between supervised and reinforcement learning.
Answers:
Q1. Explain the Reinforcement Learning architecture.
Answer:
- Reinforcement Learning is a type of machine learning.
- An agent learns by interacting with its environment.
- It performs actions and receives rewards or penalties.
- The agent learns from its experience.
- Its main aim is to get maximum total reward.
Main components:
1. Agent: The learner or decision-maker.
2. Environment: The place where the agent works.
3. State: The current situation.
4. Action: The step taken by the agent.
5. Reward: Feedback received after an action.
6. Policy: The strategy used to select actions.







medium.com


Q2. Explain the Q-Learning algorithm.
Answer:
- Q-Learning is a model-free Reinforcement Learning algorithm.
- It helps an agent learn the best action in a particular state.
- It uses a Q-table to store the values of actions.
- The agent selects an action and receives a reward.
- It updates the Q-value based on the reward.
- It repeats this process until it learns the best actions.
Steps:
1. Initialize the Q-table with zeros.
2. Observe the current state.
3. Select an action.
4. Perform the action.
5. Receive a reward.
6. Update the Q-value.
7. Repeat until the agent learns the best policy.
Q3. Differentiate between Exploration and Exploitation.
Exploration	Exploitation
Trying new actions.	Choosing the best-known action.
Helps discover new rewards.	Uses existing knowledge.
May involve some risk.	Usually chooses a familiar action.
Example: Trying a new path.	Example: Choosing the shortest known path.

Q4. Explain the Reward and Penalty Mechanism.
Answer:
- Reward and penalty are feedback given to the agent.
- A reward is given for a good action.
- A penalty is given for a wrong or unwanted action.
- The agent learns from this feedback.
- Its aim is to maximize rewards and minimize penalties.
Example:
- A robot reaching its destination gets +10 reward.
- A robot hitting a wall gets -5 penalty.
Q5. Explain the Applications of Reinforcement Learning.
Answer:
1. Robotics.
2. Self-driving cars.
3. Game playing.
4. Traffic signal control.
5. Healthcare.
6. Recommendation systems.
7. Industrial automation.
Q6. Differentiate between Supervised and Reinforcement Learning.
Supervised Learning	Reinforcement Learning
Uses labelled data.	Learns through interaction.
Correct output is provided during training.	Feedback is given through rewards or penalties.
Learns from examples.	Learns from experience.
No action-based environment is required.	Interacts with an environment.
Example: Email classification.	Example: Robot navigation.

3. Important Theory Syllabus Questions
These are useful short-answer questions to prepare in addition to the experiment-wise viva.
No.	Question	Short answer
1	Define Machine Learning.	A technique that enables computers to learn from data.
2	What is a model?	A mathematical representation learned from data.
3	What is training data?	Data used to train the model.
4	What is testing data?	Data used to evaluate the model.
5	What is data preprocessing?	Cleaning and preparing raw data before training.
6	What is normalization?	Scaling values to a common range, often 0 to 1.
7	What is standardization?	Transforming data to have mean 0 and standard deviation 1.
8	What is feature selection?	Selecting the most useful input features.
9	What is feature extraction?	Creating new features from existing data.
10	What is cross-validation?	Evaluating a model using multiple data splits.
11	What is a loss function?	A function that measures prediction error.
12	What is gradient descent?	An optimization method used to minimize loss.
13	What is bias?	Error caused by overly simple assumptions in a model.
14	What is variance?	Sensitivity of a model to changes in training data.
15	What is hyperparameter tuning?	Finding suitable model settings to improve performance.
16	What is a classification problem?	Predicting categories, such as spam or not spam.
17	What is a regression problem?	Predicting numerical values, such as house prices.
18	What is clustering?	Grouping similar data points.
19	What is an algorithm?	A step-by-step procedure to solve a problem.
20	What is prediction?	Using a trained model to estimate an output for new data.

4. Full Forms to Remember
Term	Full form
AI	Artificial Intelligence
ML	Machine Learning
PCA	Principal Component Analysis
SVM	Support Vector Machine
KNN	K-Nearest Neighbors
K-Means	K-Means Clustering Algorithm
RL	Reinforcement Learning
RF	Random Forest
RBF	Radial Basis Function
MSE	Mean Squared Error
RMSE	Root Mean Squared Error
MAE	Mean Absolute Error
WCSS	Within-Cluster Sum of Squares
CSV	Comma-Separated Values
L1	Lasso Regularization
L2	Ridge Regularization
API	Application Programming Interface
IDE	Integrated Development Environment
Q-Learning	Quality Learning (commonly called Q-Learning, not an acronym)
R²	Coefficient of Determination

5. Python and Libraries Viva
These questions are especially common when the examiner asks you to explain your code.
Question	Answer
What is Python?	A high-level programming language.
What is Pandas?	A Python library used for data manipulation and analysis.
What is NumPy?	A library used for numerical calculations and arrays.
What is Matplotlib?	A library used for data visualization.
What is Seaborn?	A library used to create statistical graphs.
What is Scikit-learn?	A Python library used for machine learning.
What is Gymnasium?	A library used for reinforcement learning environments.
What is read_csv()?	Reads a CSV file into a DataFrame.
What is head()?	Displays the first five rows by default.
What is shape?	Returns the number of rows and columns.
What is drop()?	Removes specified rows or columns.
What is fit()?	Trains or learns from the supplied data.
What is predict()?	Predicts outputs using a trained model.
What is train_test_split()?	Divides data into training and testing sets.
What is StandardScaler()?	Standardizes numerical features.

6. Very Important Differences
These are common viva questions. Learn these tables properly.

1. Supervised vs Unsupervised Learning
Supervised	Unsupervised
Uses labelled data	Uses unlabelled data
Has known output	No known output
Classification and regression	Clustering and PCA
Example: SVM	Example: K-Means

2. Classification vs Regression
Classification	Regression
Predicts categories	Predicts numerical values
Example: Car safety	Example: Uber fare
Output: Good/Bad	Output: ₹250

3. Overfitting vs Underfitting
Overfitting	Underfitting
Learns training data too closely	Fails to learn important patterns
High training accuracy	Low training accuracy
Poor generalization	Poor performance on training and testing data

4. Ridge vs Lasso
Ridge	Lasso
L2 regularization	L1 regularization
Shrinks coefficients	Can make coefficients zero
Keeps all features generally	Can perform feature selection

_____________________________________________________________________________________________________________________
      
# DMV 1  Data Loading, Storage and File Formats
Problem Statement: Analyzing Sales Data from Multiple File Formats
Dataset: Sales data in multiple file formats (e.g., CSV, Excel, JSON)
Description: The goal is to load and analyze sales data from different file formats, including 
CSV, Excel, and JSON, and perform data cleaning, transformation, and analysis on the 
dataset.

import pandas as pd
import matplotlib.pyplot as plt

# Create sales data
df = pd.DataFrame({
    "Product": ["Pen", "Book", "Bag"],
    "Quantity": [10, 5, 2],
    "Price": [10, 100, 500]
})

# Save and load CSV file
df.to_csv("sales.csv", index=False)
data = pd.read_csv("sales.csv")

# Clean data
data = data.dropna()
data = data.drop_duplicates()

# Calculate sales
data["Total"] = data["Quantity"] * data["Price"]

print(data)
print("Total Sales:", data["Total"].sum())
print("Average Sales:", data["Total"].mean())

# Display chart
data.plot(x="Product", y="Total", kind="bar")
plt.show()

#OR#

#DMV 1
import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV, Excel and JSON files
csv_data = pd.read_csv("sales_data.csv")
excel_data = pd.read_excel("sales_data.xlsx")
json_data = pd.read_json("sales_data.json")

print("CSV DATA")
print(csv_data.head())
print("\nEXCEL DATA")
print(excel_data.head())
print("\nJSON DATA")
print(json_data.head())

# Remove duplicate rows and clean missing values
csv_data = csv_data.drop_duplicates()
excel_data = excel_data.drop_duplicates()
json_data = json_data.drop_duplicates()

csv_data = csv_data.dropna()
excel_data = excel_data.dropna()
json_data = json_data.dropna()

# Combine all three files into one DataFrame
data = pd.concat([csv_data, excel_data, json_data], ignore_index=True)
data = data.drop_duplicates()

# Convert columns to correct data types
data["Quantity"] = pd.to_numeric(data["Quantity"], errors="coerce")
data["Price"] = pd.to_numeric(data["Price"], errors="coerce")
data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
data = data.dropna(subset=["Quantity", "Price", "Product"])

# Calculate total sales
data["Total_Sales"] = data["Quantity"] * data["Price"]

print("\nTotal Sales:", data["Total_Sales"].sum())
print("Average Sales:", data["Total_Sales"].mean())
print("\nProduct-wise Sales:")
print(data.groupby("Product")["Total_Sales"].sum())

# Bar chart
data.groupby("Product")["Total_Sales"].sum().plot(kind="bar")
plt.title("Product-wise Sales")
plt.xlabel("Product")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.show()

# Pie chart
data.groupby("Product")["Total_Sales"].sum().plot(
    kind="pie", autopct="%1.1f%%"
)
plt.title("Product Sales Distribution")
plt.ylabel("")
plt.tight_layout()
plt.show()

# Box plot
data["Total_Sales"].plot(kind="box")
plt.title("Sales Distribution")
plt.ylabel("Total Sales")
plt.show()

data.to_csv("Cleaned_Sales_Data.csv", index=False)
print("\nCleaned data saved successfully.")
---------------------------------------------------------------------------------------------
# DMV 2  Interacting with Web APIs
Problem Statement: Analyzing Weather Data from OpenWeatherMap API
Dataset: Weather data retrieved from OpenWeatherMap API
Description: The goal is to interact with the OpenWeatherMap API to retrieve weather data 
for a specific location and perform data modeling and visualization to analyze weather 
patterns over time.
    
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

api_key = "YOUR_API_KEY"
city = "Pune"

url = "https://api.openweathermap.org/data/2.5/forecast"
params = {
    "q": city,
    "appid": api_key,
    "units": "metric"
}

response = requests.get(url, params=params, timeout=20)

if response.status_code != 200:
    print("API Error:", response.status_code, response.text)
else:
    weather = response.json()

    records = []
    for item in weather["list"]:
        records.append({
            "Date": item["dt_txt"],
            "Temperature": item["main"]["temp"],
            "Humidity": item["main"]["humidity"],
            "Wind_Speed": item["wind"]["speed"],
            "Rain": item.get("rain", {}).get("3h", 0)
        })

    data = pd.DataFrame(records)
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.drop_duplicates().dropna()

    print(data.head())
    print("\nAverage Temperature:", data["Temperature"].mean())
    print("Maximum Temperature:", data["Temperature"].max())
    print("Minimum Temperature:", data["Temperature"].min())

    # Temperature trend
    plt.plot(data["Date"], data["Temperature"])
    plt.title("Weather Temperature Trend in " + city)
    plt.xlabel("Date")
    plt.ylabel("Temperature (C)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Rainfall bar chart
    plt.bar(data["Date"], data["Rain"])
    plt.title("Rainfall Forecast")
    plt.xlabel("Date")
    plt.ylabel("Rain (mm)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Temperature and humidity relationship
    plt.scatter(data["Temperature"], data["Humidity"])
    plt.title("Temperature vs Humidity")
    plt.xlabel("Temperature (C)")
    plt.ylabel("Humidity (%)")
    plt.show()

    # Correlation heatmap
    sns.heatmap(
        data[["Temperature", "Humidity", "Wind_Speed", "Rain"]].corr(),
        annot=True,
        cmap="coolwarm"
    )
    plt.title("Weather Parameter Correlation")
    plt.tight_layout()
    plt.show()

    data.to_csv("Weather_Data.csv", index=False)
    print("Weather data saved successfully.")
---------------------------------------------------------------------------------------------
# DMV 3 Data Cleaning and Preparation
Problem Statement: Analyzing Customer Churn in a Telecommunications Company
Dataset: "Telecom_Customer_Churn.csv"
Description: The dataset contains information about customers of a telecommunications 
company and whether they have churned (i.e., discontinued their services). The dataset 
includes various attributes of the customers, such as their demographics, usage patterns, and 
account information. The goal is to perform data cleaning and preparation to gain insights 
into the factors that contribute to customer churn.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("telecom_customer_churn.csv")

# Explore data
print(data.head())
print(data.isnull().sum())

# Clean data
data = data.drop_duplicates()
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
data = data.dropna()

# Convert text to lowercase
for col in data.select_dtypes(include="object").columns:
    data[col] = data[col].str.lower().str.strip()

# Create new feature
data["CustomerYears"] = data["tenure"] / 12

# Separate input and output BEFORE encoding
X = data.drop("Churn", axis=1)
y = data["Churn"].map({"yes": 1, "no": 0})

# Remove customer ID
X = X.drop("customerID", axis=1)

# Convert text into numbers
X = pd.get_dummies(X, drop_first=True, dtype=int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save cleaned data
data.to_csv("cleaned_churn.csv", index=False)

print("Data cleaning completed!")
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

#OR#

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("telecom_customer_churn.csv")

print("First five records:")
print(data.head())
print("\nDataset information:")
data.info()
print("\nMissing values:")
print(data.isnull().sum())

# Remove duplicate records
data = data.drop_duplicates()

# Clean column names and text values
data.columns = data.columns.str.strip()
for col in data.select_dtypes(include="object").columns:
    data[col] = data[col].str.strip()

# Fill missing numerical values with the median
for col in data.select_dtypes(include="number").columns:
    data[col] = data[col].fillna(data[col].median())

# Fill optional categorical values
for col in data.select_dtypes(include="object").columns:
    data[col] = data[col].fillna("Not Applicable")

# Remove invalid negative monthly charges
data = data[data["Monthly Charge"] >= 0]

# Feature engineering
data["Customer_Years"] = data["Tenure in Months"] / 12

# Convert customer status into a churn indicator
data["Churn"] = data["Customer Status"].map({
    "Churned": 1,
    "Stayed": 0,
    "Joined": 0
})

# Remove identifiers, high-cardinality location fields and original target text
data = data.drop(
    columns=[
        "Customer ID", "Customer Status", "Churn Category", "Churn Reason",
        "City", "Zip Code", "Latitude", "Longitude"
    ],
    errors="ignore"
)

# Convert categorical columns into numeric columns
data = pd.get_dummies(data, drop_first=True)

X = data.drop("Churn", axis=1)
y = data["Churn"]

# Split before scaling to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

data.to_csv("Cleaned_Telecom_Data.csv", index=False)
print("Cleaned telecom data saved successfully.")
---------------------------------------------------------------------------------------------

# DMV 4  Data Wrangling
Problem Statement: Data Wrangling on Real Estate Market
Dataset: "RealEstate_Prices.csv"
Description: The dataset contains information about housing prices in a specific real estate 
market. It includes various attributes such as property characteristics, location, sale prices, 
and other relevant features. The goal is to perform data wrangling to gain insights into the 
factors influencing housing prices and prepare the dataset for further analysis or modeling.
    
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("telecom_customer_churn.csv")

# Explore data
print(data.head())
print(data.isnull().sum())

# Clean data
data = data.drop_duplicates()
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
data = data.dropna()

# Convert text to lowercase
for col in data.select_dtypes(include="object").columns:
    data[col] = data[col].str.lower().str.strip()

# Create new feature
data["CustomerYears"] = data["tenure"] / 12

# Separate input and output BEFORE encoding
X = data.drop("Churn", axis=1)
y = data["Churn"].map({"yes": 1, "no": 0})

# Remove customer ID
X = X.drop("customerID", axis=1)

# Convert text into numbers
X = pd.get_dummies(X, drop_first=True, dtype=int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save cleaned data
data.to_csv("cleaned_churn.csv", index=False)

print("Data cleaning completed!")
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

#OR#

# DMV 4

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv("realestate_prices.csv")

# Explore data
print(data.head())
print(data.isnull().sum())

# Clean data
data = data.drop_duplicates()
data = data.dropna()

# Convert TotalCharges to numbers
data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"], errors="coerce"
)
data = data.dropna()

# Create new feature
data["CustomerYears"] = data["tenure"] / 12

# Convert text into numbers
data = pd.get_dummies(data, drop_first=True, dtype=int)

# Split data
X = data.drop("Churn_Yes", axis=1)
y = data["Churn_Yes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save cleaned data
data.to_csv("cleaned_churn.csv", index=False)

print("Data cleaning completed!")
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)
---------------------------------------------------------------------------------------------
# DMV 5  Data Visualization using matplotlib
Problem Statement: Analyzing Air Quality Index (AQI) Trends in a City 
Dataset: "City_Air_Quality.csv"
Description: The dataset contains information about air quality measurements in a specific 
city over a period of time. It includes attributes such as date, time, pollutant levels (e.g., PM2.5, 
PM10, CO), and the Air Quality Index (AQI) values. The goal is to use the matplotlib library 
to create visualizations that effectively represent the AQI trends and patterns for different 
pollutants in the city. 

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("city_day.csv")

print("First five records:")
print(data.head())
print("\nDataset information:")
data.info()

# Convert date and select Delhi records
data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
data = data[data["City"] == "Delhi"].copy()
data = data.dropna(subset=["Date", "AQI"])
data = data.sort_values("Date")

# Fill missing pollutant values with their median
pollutants = ["PM2.5", "PM10", "CO"]
for col in pollutants:
    data[col] = data[col].fillna(data[col].median())

print("\nDelhi AQI records:")
print(data[["Date", "AQI", "AQI_Bucket"]].head())

# AQI trend
plt.plot(data["Date"], data["AQI"], label="AQI")
plt.title("Daily AQI Trend in Delhi")
plt.xlabel("Date")
plt.ylabel("AQI")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Pollutant trends
for col in pollutants:
    plt.plot(data["Date"], data[col], label=col)
    plt.title(col + " Trend in Delhi")
    plt.xlabel("Date")
    plt.ylabel(col)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Bar chart for the latest 10 AQI records
latest = data.tail(10)
plt.bar(latest["Date"].dt.strftime("%Y-%m-%d"), latest["AQI"])
plt.title("Latest 10 AQI Records in Delhi")
plt.xlabel("Date")
plt.ylabel("AQI")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Box plot for pollutant distributions
data[pollutants].plot(kind="box")
plt.title("Pollutant Distribution in Delhi")
plt.ylabel("Pollutant Level")
plt.tight_layout()
plt.show()

# AQI versus PM2.5
plt.scatter(data["PM2.5"], data["AQI"])
plt.title("AQI vs PM2.5 in Delhi")
plt.xlabel("PM2.5")
plt.ylabel("AQI")
plt.show()

data.to_csv("Cleaned_Delhi_AQI_Data.csv", index=False)
print("AQI data saved successfully.")

#OR#

# DMV 5

import pandas as pd

# Create sample data
data = pd.DataFrame({
    "Area": [1000, 1500, 1200, 1800, 2000],
    "Location": ["Pune", "Mumbai", "Pune", "Mumbai", "Pune"],
    "Price": [3000000, 5000000, 3500000, 6000000, 90000000]
})

# 1. Clean column names
data.columns = data.columns.str.strip().str.lower()

# 2. Handle missing values
data = data.dropna()

# 3. Filter data
data = data[data["area"] > 0]

# 4. Encode categorical data
data = pd.get_dummies(data, columns=["location"], dtype=int)

# 5. Calculate average price
print("Average Price:", data["price"].mean())

# 6. Remove outliers using IQR
q1 = data["price"].quantile(0.25)
q3 = data["price"].quantile(0.75)
iqr = q3 - q1

data = data[
    (data["price"] >= q1 - 1.5 * iqr) &
    (data["price"] <= q3 + 1.5 * iqr)
]

# Display cleaned data
print("\nCleaned Data:")
print(data)

# 7. Save data
data.to_csv("cleaned_realestate.csv", index=False)
print("\nCleaned data saved!")
---------------------------------------------------------------------------------------------
# DMV 6 Data Aggregation
Problem Statement: Analyzing Sales Performance by Region in a Retail Company
Dataset: "Retail_Sales_Data.csv"
Description: The dataset contains information about sales transactions in a retail company. It 
includes attributes such as transaction date, product category, quantity sold, and sales 
amount. The goal is to perform data aggregation to analyze the sales performance by region 
and identify the top-performing regions.

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Retail_Sales_Data.csv")

print("First five records:")
print(data.head())
print("\nDataset information:")
data.info()
print("\nSummary statistics:")
print(data.describe())

# In this dataset, State is used as the region
# and Shop_Category is used as the product category.
data = data.drop_duplicates()
data = data.dropna(subset=["State", "Sales", "Shop_Category"])

# Total sales by state
state_sales = data.groupby("State")["Sales"].sum()
print("\nSales by State:")
print(state_sales)

# Bar chart
state_sales.plot(kind="bar")
plt.title("Sales Performance by State")
plt.xlabel("State")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.show()

# Pie chart
state_sales.plot(kind="pie", autopct="%1.1f%%")
plt.title("State-wise Sales Distribution")
plt.ylabel("")
plt.tight_layout()
plt.show()

# Highest sales state
print("\nHighest Sales State:", state_sales.idxmax())
print("Highest Sales Amount:", state_sales.max())

# Sales by state and shop category
category_sales = data.groupby(
    ["State", "Shop_Category"]
)["Sales"].sum().unstack(fill_value=0)

print("\nState and Category-wise Sales:")
print(category_sales)

# Stacked bar chart
category_sales.plot(kind="bar", stacked=True)
plt.title("Category-wise Sales by State")
plt.xlabel("State")
plt.ylabel("Total Sales")
plt.legend(title="Shop Category")
plt.tight_layout()
plt.show()

data.to_csv("Cleaned_Retail_Sales_Data.csv", index=False)
print("Retail sales data saved successfully.")

#OR#

# DMV 6

import pandas as pd
import matplotlib.pyplot as plt

# Read dataset
data = pd.read_csv(r"D:\Insem\Delhi_AQI_Dataset.csv")

# Display data
print(data.head())

# AQI trend
plt.plot(data["Date"], data["AQI"])
plt.title("AQI Trend")
plt.xlabel("Date")
plt.ylabel("AQI")
plt.xticks(rotation=45)
plt.show()

# Pollutant levels
plt.plot(data["Date"], data["PM2.5"], label="PM2.5")
plt.plot(data["Date"], data["PM10"], label="PM10")
plt.plot(data["Date"], data["CO"], label="CO")
plt.title("Pollutant Levels")
plt.xlabel("Date")
plt.ylabel("Pollutant Level")
plt.legend()
plt.xticks(rotation=45)
plt.show()

# Bar plot
plt.bar(data["Date"], data["AQI"])
plt.title("AQI Comparison")
plt.xlabel("Date")
plt.ylabel("AQI")
plt.xticks(rotation=45)
plt.show()

# Box plot
plt.boxplot(data["AQI"].dropna())
plt.title("AQI Distribution")
plt.show()

# Scatter plot
plt.scatter(data["PM2.5"], data["AQI"])
plt.title("PM2.5 vs AQI")
plt.xlabel("PM2.5")
plt.ylabel("AQI")
plt.show()

________________________________________________________________________________________________________________________
VIVA Questions

Practical 1: Analyzing Sales Data from Multiple File Formats
Aim: To load, clean, combine and analyze sales data from CSV, Excel and JSON files.

Libraries used:
- pandas: Data handling and analysis.
- matplotlib: Data visualization.
Code explanation:
Code	Purpose
pd.read_csv()	Reads CSV file
pd.read_excel()	Reads Excel file
pd.read_json()	Reads JSON file
head()	Displays first five records
drop_duplicates()	Removes duplicate records
dropna()	Removes rows containing missing values
pd.concat()	Combines multiple DataFrames
groupby()	Groups data by product
sum()	Calculates total sales
mean()	Calculates average sales
plot(kind="bar")	Creates bar chart
plot(kind="pie")	Creates pie chart
plot(kind="box")	Creates box plot
to_csv()	Saves cleaned data

Important formula:
\[
\text{Total Sales} = \text{Quantity} \times \text{Price}
\]

Viva questions:

1. What is a CSV file?
A file that stores tabular data separated by commas.

2. What is JSON?
JavaScript Object Notation, a lightweight data-interchange format.

3. What is a DataFrame?
A two-dimensional tabular data structure in Pandas.

4. Why do we use concat()?
To combine multiple DataFrames.

5. What is data cleaning?
Removing errors, duplicates and handling missing values.

6. What is a box plot?
A graph used to understand data distribution and identify outliers.

7. What is aggregation?
Summarizing data using operations like sum, mean and count.

Practical 2: Analyzing Weather Data using OpenWeatherMap API
Aim: To retrieve weather information using an API and analyze temperature, humidity, wind speed and rainfall.

Libraries used:
- requests: Fetches data from the API.
- pandas: Processes weather data.
- matplotlib: Creates graphs.
- seaborn: Creates a correlation heatmap.
Code explanation:
Code	Purpose
requests.get()	Sends API request
params	Stores API parameters
response.json()	Converts API response into JSON
weather["list"]	Accesses forecast records
item["main"]["temp"]	Extracts temperature
item["main"]["humidity"]	Extracts humidity
item["wind"]["speed"]	Extracts wind speed
pd.DataFrame()	Creates weather DataFrame
pd.to_datetime()	Converts date into datetime format
plt.plot()	Displays temperature trend
plt.scatter()	Shows relationship between two variables
sns.heatmap()	Displays correlation between variables

Important terms:
- API: Application Programming Interface.
- API Key: Unique key used to authenticate API requests.
- Humidity: Amount of water vapour in the air.
- Correlation: Measures the relationship between two variables.
Viva questions:

1. What is an API?
An interface that allows two applications to communicate.

2. What is OpenWeatherMap?
An online platform that provides weather data through APIs.

3. What is an API key?
A unique authentication key used to access an API.

4. What is JSON response?
Data returned in JSON format.

5. What is correlation?
A statistical measure showing how two variables are related.

6. What is a heatmap?
A graphical representation of data using colours.

7. What is rainfall measured in?
Millimetres (mm).

8. What is the difference between temperature and humidity?
Temperature measures how hot or cold the air is. Humidity measures moisture in the air.

Practical 3: Analyzing Customer Churn in a Telecommunications Company
Aim: To clean and prepare customer churn data for machine learning analysis.

Libraries used:
- pandas: Data cleaning.
- sklearn: Data preprocessing and splitting.
- StandardScaler: Feature scaling.
Code explanation:
Code	Purpose
read_csv()	Loads telecom dataset
info()	Displays dataset information
isnull().sum()	Counts missing values
drop_duplicates()	Removes duplicate records
fillna()	Fills missing values
median()	Finds middle value
pd.to_numeric()	Converts values into numeric type
get_dummies()	Converts categorical data into numerical form
train_test_split()	Divides data into training and testing sets
StandardScaler()	Standardizes numerical features
fit_transform()	Learns scaling values and transforms training data
transform()	Applies the same scaling to test data

Important terms:
- Customer Churn: When a customer stops using a company's services.
- Feature Engineering: Creating new useful variables from existing data.
- Normalization: Scaling data to a specified range, commonly 0 to 1.
- Standardization: Transforming data to have mean 0 and standard deviation 1.
Viva questions:

1. What is customer churn?
When a customer stops using a company's service.

2. Why is data cleaning important?
It improves data quality and analysis accuracy.

3. What is a missing value?
A value that is absent from a dataset.

4. What is an outlier?
A value that is unusually different from other observations.

5. What is feature engineering?
Creating new features from existing data.

6. What is categorical data?
Data represented by categories, such as gender or city.

7. Why do we use get_dummies()?
To convert categorical values into numerical columns.

8. What is StandardScaler?
A method that standardizes data using mean and standard deviation.

9. What is train-test split?
Dividing data into training and testing subsets.

10. Why do we split data?
To train a model and evaluate its performance on unseen data.

Practical 4: Data Wrangling on Real Estate Market
Aim: To clean, transform and organize real estate data for analysis.

Libraries used:

- pandas

Code explanation:
Code	Purpose
read_csv()	Loads real estate data
columns.str.strip()	Removes extra spaces from column names
str.lower()	Converts column names to lowercase
str.replace()	Replaces spaces in column names
drop_duplicates()	Removes duplicate records
dropna()	Removes rows with missing values
groupby()	Groups properties by zipcode
mean()	Calculates average property price
quantile()	Calculates quartiles
get_dummies()	Encodes categorical columns
to_csv()	Exports cleaned dataset

Important formula:
\[
\text{Price per Sqft} =
\frac{\text{Property Price}}{\text{Living Area}}
\]

IQR formula:
\[
IQR = Q3-Q1
\]

Where:
- Q1 = First quartile (25%)
- Q3 = Third quartile (75%)
- IQR = Interquartile Range
Viva questions:

1. What is data wrangling?
Converting raw data into a clean and structured format.

2. What is data transformation?
Changing data into a suitable format for analysis.

3. What is data filtering?
Selecting records based on specific conditions.

4. What is grouping?
Combining records with similar values.

5. What is IQR?
Interquartile Range, the difference between Q3 and Q1.

6. Why is IQR used?
To identify and handle outliers.

7. What is categorical encoding?
Converting categorical data into numerical values.

8. What is quantile()?
A function used to calculate data distribution cut-off points.

Practical 5: Analyzing Air Quality Index (AQI) Trends in a City
Aim: To visualize and analyze AQI trends and pollutant levels using Matplotlib.

Libraries used:
- pandas
- matplotlib
Code explanation:
Code	Purpose
read_csv()	Loads AQI dataset
pd.to_datetime()	Converts date into datetime format
data["City"] == "Delhi"	Filters Delhi records
dropna()	Removes records with missing AQI
sort_values()	Sorts records by date
fillna()	Fills missing pollutant values
plt.plot()	Creates line graph
plt.bar()	Creates bar chart
plt.scatter()	Creates scatter plot
plot(kind="box")	Creates box plot
plt.xlabel()	Adds X-axis label
plt.ylabel()	Adds Y-axis label
plt.legend()	Displays graph legend

Important pollutants:
- PM2.5: Fine particulate matter with diameter 2.5 micrometres or less.
- PM10: Particulate matter with diameter 10 micrometres or less.
- CO: Carbon Monoxide.
- NO₂: Nitrogen Dioxide.
- SO₂: Sulphur Dioxide.
- O₃: Ozone.
Viva questions:

1. What is AQI?
Air Quality Index, which indicates the quality of air.

2. What is PM2.5?
Fine particles with a diameter of 2.5 micrometres or less.

3. What is PM10?
Particles with a diameter of 10 micrometres or less.

4. What is the purpose of AQI analysis?
To monitor air pollution and understand air quality trends.

5. Why is a line graph used?
To show changes over time.

6. Why is a bar chart used?
To compare different categories or values.

7. What is a scatter plot?
A graph showing the relationship between two numerical variables.

8. What is a legend?
It identifies different data series in a graph.

9. What is time-series data?
Data collected over time.

Practical 6: Analyzing Sales Performance by Region in a Retail Company
Aim: To aggregate and visualize sales based on state and shop category.

Libraries used:
- pandas
- matplotlib
Code explanation:
Code	Purpose
read_csv()	Loads retail sales data
head()	Displays first five records
describe()	Shows statistical summary
groupby("State")	Groups sales by state
sum()	Calculates total sales
idxmax()	Returns the state with highest sales
unstack()	Converts grouped data into columns
plot(kind="bar")	Creates bar graph
plot(kind="pie")	Creates pie chart
plot(kind="bar", stacked=True)	Creates stacked bar chart

Viva questions:

1. What is data aggregation?
Summarizing data using functions such as sum, count and mean.

2. What is groupby?
It groups records based on one or more columns.

3. What is the purpose of idxmax()?
It returns the index or label of the maximum value.

4. What is a stacked bar chart?
A bar chart that displays multiple categories within each bar.

5. What is the difference between bar chart and pie chart?
A bar chart compares values, while a pie chart shows proportions.

6. What is sales analysis?
Studying sales data to understand performance and trends.

7. Why do we group sales by state?
To compare the sales performance of different states.

Part 2: Important Python and Data Mining theory questions
These questions are common across practicals.
A. Name the following
Question	Answer
Library used for data analysis	Pandas
Library used for numerical operations	NumPy
Library used for plotting graphs	Matplotlib
Library used for statistical visualization	Seaborn
Library used for HTTP requests	Requests
Library used for machine learning	Scikit-learn
Function to read CSV	read_csv()
Function to read Excel	read_excel()
Function to read JSON	read_json()
Function to display first 5 rows	head()
Function to display last 5 rows	tail()
Function to check missing values	isnull()
Function to remove duplicates	drop_duplicates()
Function to remove missing values	dropna()
Function to fill missing values	fillna()
Function to calculate average	mean()
Function to calculate total	sum()
Function to find maximum	max()
Function to find minimum	min()
Function to group data	groupby()
Function to save CSV	to_csv()
Function to create a line graph	plt.plot()
Function to create a bar graph	plt.bar()
Function to create a scatter plot	plt.scatter()
Function to create a histogram	plt.hist()
Function to create a pie chart	plt.pie()
Function to create a box plot	plt.boxplot()

B. Write the full forms
Short form	Full form
DMV	Data Mining and Visualization
CSV	Comma-Separated Values
JSON	JavaScript Object Notation
API	Application Programming Interface
AQI	Air Quality Index
AI	Artificial Intelligence
ML	Machine Learning
EDA	Exploratory Data Analysis
IQR	Interquartile Range
SD	Standard Deviation
KNN	K-Nearest Neighbors
SVM	Support Vector Machine
PCA	Principal Component Analysis
HTTP	Hypertext Transfer Protocol
URL	Uniform Resource Locator
PM	Particulate Matter
CO	Carbon Monoxide
NO₂	Nitrogen Dioxide
SO₂	Sulphur Dioxide
O₃	Ozone
XML	Extensible Markup Language
SQL	Structured Query Language
GUI	Graphical User Interface

C. Short theory questions
1. What is Data Mining?

Extracting useful patterns and knowledge from large datasets.

2. What is Data Visualization?

Representing data graphically using charts, graphs and plots.

3. What is EDA?

Exploratory Data Analysis. It is used to understand datasets through statistics and visualization.

4. What is Data Preprocessing?

Preparing raw data by cleaning, transforming and organizing it.

5. What are the types of data?
- Structured data
- Semi-structured data
- Unstructured data
6. What is structured data?

Data arranged in rows and columns, such as CSV files.

7. What is unstructured data?

Data without a fixed tabular structure, such as images, videos and text documents.

8. What is a missing value?

A data entry that is not available or recorded.

9. What is an outlier?

An unusually high or low value compared with the rest of the data.

10. What is normalization?

Scaling values to a common range, often 0 to 1.

11. What is standardization?

Scaling data to mean 0 and standard deviation 1.

12. What is correlation?

A measure of the relationship between two variables.

13. What is positive correlation?

When one variable increases, the other also tends to increase.

14. What is negative correlation?

When one variable increases, the other tends to decrease.

15. What is a histogram?

A graph showing the frequency distribution of numerical data.

16. What is a line chart?

A graph used to show trends over time.

17. What is a pie chart?

A circular graph showing proportions of a whole.

18. What is a box plot?

A graph showing median, quartiles, spread and potential outliers.

19. What is a heatmap?

A visualization that represents values using different colours.

20. What is machine learning?

A branch of AI that enables computers to learn patterns from data.

Part 3: Important differences for viva
Topic 1	Topic 2
CSV: Stores tabular data in comma-separated text	JSON: Stores data in key-value format
dropna(): Removes missing values	fillna(): Replaces missing values
mean(): Calculates average	median(): Finds middle value
Normalization: Scales to a fixed range	Standardization: Uses mean and standard deviation
Bar chart: Compares categories	Histogram: Shows numerical frequency distribution
Line chart: Shows trends	Scatter plot: Shows relationships
Training data: Used to train model	Testing data: Used to evaluate model
Data cleaning: Corrects errors	Data wrangling: Transforms data into useful structure
max(): Returns maximum value	idxmax(): Returns its index or label
concat(): Combines DataFrames	groupby(): Groups records

Part 4: Most important practical viva questions
These are worth revising first because they connect directly to your programs.

1. Why do we import Pandas?
To read, clean, manipulate and analyze datasets.

2. Why do we import Matplotlib?
To visualize data using graphs.

3. What is the difference between head() and info()?
head() displays records, while info() displays column names, data types and non-null counts.

4. What does axis=1 mean?
It represents columns. axis=0 represents rows.

5. What is inplace=True?
It modifies the original DataFrame directly.

6. What is random_state=42?
It ensures reproducible random splitting.

7. What does test_size=0.2 mean?
20% of the dataset is used for testing.

8. What is drop_first=True?
It removes the first dummy column to avoid redundant categorical encoding.

9. What is the purpose of errors="coerce"?
Invalid values are converted into missing values (NaN).

10. Why do we use plt.show()?
To display the graph.

11. What does plt.tight_layout() do?
It adjusts graph spacing to prevent overlapping labels.

12. What is a DataFrame?
A two-dimensional table with rows and columns.

13. What is a Series?
A one-dimensional labelled data structure in Pandas.

14. What is the difference between loc and iloc?
loc accesses data by labels, while iloc accesses data by integer positions.

15. Why do we export cleaned datasets?
To save processed data for future analysis.

Part 5: Quick revision before entering the lab
Remember these 6 practicals
1. Sales Data: Read → Clean → Combine → Calculate Sales → Visualize
2. Weather API: Request → Extract → DataFrame → Analyze → Visualize
3. Customer Churn: Load → Clean → Encode → Scale → Split
4. Real Estate: Load → Clean → Transform → Group → Handle Outliers
5. AQI: Load → Filter City → Clean → Plot Trends → Compare Pollutants
6. Retail Sales: Load → Group by State → Aggregate → Find Highest Sales → Visualize

Top 10 things to memorize
- read_csv()
- head()
- info()
- isnull().sum()
- dropna()
- drop_duplicates()
- groupby()
- fillna()
- plt.plot()
- to_csv()
