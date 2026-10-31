
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
# ML 3

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
# ML 4

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
# ML 6

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
# DMV 1

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
# DMV 2
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
# DMV 3

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

# DMV 4
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
# DMV 5

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
# DMV 6

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
