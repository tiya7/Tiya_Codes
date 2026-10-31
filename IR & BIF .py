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

------------------------------------------------------------------------------------------

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
