# IR 1

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

file = open("D:\\Insem\\document.txt", "r")
text = file.read()
file.close()

words = word_tokenize(text)
stop = stopwords.words('english')
stemmer = PorterStemmer()

for word in words:
    if word.lower() not in stop and word.isalpha():
        print(stemmer.stem(word.lower()))

document.txt file :: 
The students are studying and playing games.
Information retrieval is an important field.

#OR#

# IR 1
from nltk.stem import PorterStemmer

# Input text
text = "The students are studying and playing games"

# Convert text to lowercase
words = text.lower().split()

# Define stop words
stop_words = {"the", "are", "and"}

# Create stemmer
ps = PorterStemmer()

# Remove stop words and perform stemming
result = []

for word in words:
    if word.isalpha() and word not in stop_words:
        result.append(ps.stem(word))

# Display output
print("Original Text:", text)
print("Processed Text:", result)

#OR#

import nltk
import re

nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

text = "Information retrieval systems are retrieving useful documents from large collections."

words = re.findall(r'\b[a-zA-Z]+\b', text.lower())

stop_words = set(stopwords.words('english'))

filtered_words = [word for word in words if word not in stop_words]

stemmer = PorterStemmer()

stemmed_words = [stemmer.stem(word) for word in filtered_words]

print("Original Text:", text)
print("Tokens:", words)
print("After Stop Word Removal:", filtered_words)
print("After Stemming:", stemmed_words)

------------------------------------------------------------------------------------------

# IR 2

documents = {
    1: "python is easy",
    2: "python is useful",
    3: "java is useful"
}

inverted = {}

for doc_id, text in documents.items():
    for word in text.split():
        if word not in inverted:
            inverted[word] = []
        if doc_id not in inverted[word]:
            inverted[word].append(doc_id)

print("Inverted Index:", inverted)

search = input("Enter word to search: ")

if search in inverted:
    print("Documents found:", inverted[search])
else:
    print("No documents found")

#OR#

from collections import defaultdict

documents = {
    "D1": "information retrieval searches documents",
    "D2": "data mining finds useful patterns",
    "D3": "retrieval systems use inverted index",
    "D4": "documents are ranked for search query"
}

inverted_index = defaultdict(list)

for doc_id, text in documents.items():

    words = text.lower().split()

    for word in set(words):
        inverted_index[word].append(doc_id)

print("Inverted Index:")

for word, doc_ids in inverted_index.items():
    print(word, ":", doc_ids)

query = input("Enter search word: ").lower()

print("Retrieved Documents:")

if query in inverted_index:
    print(inverted_index[query])
else:
    print("No documents found")

------------------------------------------------------------------------------------------

# IR 3

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

data = {
    'email': [
        'Win a free prize',
        'Claim your free money',
        'Meeting is at 10 am',
        'Please submit your assignment',
        'Win cash now',
        'Let us meet tomorrow'
    ],
    'label': ['spam', 'spam', 'ham', 'ham', 'spam', 'ham']
}

df = pd.DataFrame(data)

X = df['email']
y = df['label']

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X, y)

email = input("Enter email text: ")
email = vectorizer.transform([email])

result = model.predict(email)

print("Email is:", result[0])

examples spam::
You won a car worth rupees 100000. to claim this you need to pay rupees 50000 which is refundable amount.
"URGENT: Act immediately or your account will be deleted today!!!"
Congratulations! You won a free lottery prize.
Win cash now. Claim your reward.
Get free money by clicking this link.
You have won a free iPhone.
Urgent! Claim your prize today.
examples ham ::
Please attend the college lecture tomorrow.
Submit your assignment before Monday.
Let us meet at 10 am.
Your examination timetable is ready.
Please send me the project report.
The meeting is scheduled for tomorrow.

#OR#

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("spam.csv", encoding="latin-1")
#df = pd.read_csv("spam_ham_dataset.csv")

df = df[["Category", "Message"]]

X = df["Message"]
y = df["Category"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words="english")

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

model = MultinomialNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

message = ["Congratulations! You won a free prize"]

message = vectorizer.transform(message)

print("Prediction:", model.predict(message))
------------------------------------------------------------------------------------------

# IR 4

from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering

# Load dataset
data = load_iris()
X = data.data

# Apply clustering
model = AgglomerativeClustering(n_clusters=3)
result = model.fit_predict(X)

print("Cluster labels:")
print(result)

#OR#

# IR 4

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("spam_ham_dataset.csv")

# Select required columns
df = df[["label", "text"]].dropna()

# Separate input and output
X = df["text"]
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create Naive Bayes model
model = make_pipeline(
    CountVectorizer(),
    MultinomialNB()
)

# Train model
model.fit(X_train, y_train)

# Predict test data
predictions = model.predict(X_test)

# Evaluate model
print("Accuracy:",
      accuracy_score(y_test, predictions) * 100, "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Test a new email
email = input("\nEnter email message: ")

prediction = model.predict([email])

print("Email Classification:", prediction[0])

------------------------------------------------------------------------------------------

# IR 5

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

# Select numerical features
X = df[[
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]]

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create clustering model
model = AgglomerativeClustering(n_clusters=3)

# Fit model and predict clusters
labels = model.fit_predict(X_scaled)

# Display cluster labels
df["Cluster"] = labels

print("Cluster Labels:")
print(df[["CustomerID", "Cluster"]].head(10))

print("\nNumber of Customers in Each Cluster:")
print(df["Cluster"].value_counts())

# Generate linkage matrix
linked = linkage(X_scaled, method="ward")

# Plot dendrogram
plt.figure(figsize=(10, 6))

dendrogram(linked)

plt.title("Agglomerative Hierarchical Clustering")
plt.xlabel("Customers")
plt.ylabel("Distance")

plt.show()

# Save clustered data
df.to_csv("Clustered_Customers.csv", index=False)

print("\nResults saved in Clustered_Customers.csv")

#OR#

# IR 5

import requests
from bs4 import BeautifulSoup

url = "https://books.toscrape.com/"
page = requests.get(url)

soup = BeautifulSoup(page.text, "html.parser")

products = soup.find_all("article", class_="product_pod")

for product in products:
    name = product.h3.a["title"]
    price = product.find("p", class_="price_color").text
    link = product.h3.a["href"]

    print("Name:", name)
    print("Price:", price)
    print("Link:", link)
    print()

------------------------------------------------------------------------------------------

# BIF 1

dna = "ATGCGATCGATGAAATAG"

# Find sequence length
print("DNA Sequence:", dna)
print("Length:", len(dna))

# Calculate GC content
gc = dna.count('G') + dna.count('C')
gc_content = (gc / len(dna)) * 100
print("GC Content:", gc_content, "%")

# Find motif
motif = input("Enter motif to search: ")
print("Motif Count:", dna.count(motif))

# Find coding region (start and stop codons)
start = dna.find("ATG")
stop = dna.find("TAG", start)

if start != -1 and stop != -1:
    print("Coding Region:", dna[start:stop + 3])
else:
    print("Coding region not found")

------------------------------------------------------------------------------------------

# BIF 2

import pandas as pd

# Create dataset
data = pd.DataFrame({
    "Gene": ["GeneA", "GeneB", "GeneC"],
    "Control1": [100, 200, 50],
    "Control2": [110, 190, 55],
    "Treatment1": [250, 80, 52],
    "Treatment2": [260, 90, 49]
})

# Calculate averages
data["Control_Avg"] = data[["Control1", "Control2"]].mean(axis=1)
data["Treatment_Avg"] = data[["Treatment1", "Treatment2"]].mean(axis=1)

# Calculate fold change
data["Fold_Change"] = data["Treatment_Avg"] / data["Control_Avg"]

# Identify genes
for i, row in data.iterrows():
    if row["Fold_Change"] > 1.5:
        print(row["Gene"], "is upregulated")
    elif row["Fold_Change"] < 0.67:
        print(row["Gene"], "is downregulated")
    else:
        print(row["Gene"], "shows little change")

------------------------------------------------------------------------------------------

# BIF 4

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict results
y_pred = model.predict(X_test)

# Calculate accuracy
print("Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")

# Show important features
print("Important Features:")
for name, score in zip(data.feature_names, model.feature_importances_):
    print(name, ":", round(score, 3))
