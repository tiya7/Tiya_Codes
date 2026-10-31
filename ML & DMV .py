
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
