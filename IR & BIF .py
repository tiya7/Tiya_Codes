# IR 1 (1st)
Write a program for pre-processing of a text document such as stop word removal, stemming.

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

# IR 2 (2nd)
Implement a program for retrieval of documents using inverted files.

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

# IR 3 (4th)
Implement e-mail spam filtering using text classification algorithm with appropriate dataset. 

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

# IR 4 (5th)
Implement Agglomerative hierarchical clustering algorithm using appropriate dataset.

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

#OR#

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

df = pd.read_csv("Mall_Customers.csv")

X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

Z = linkage(X_scaled, method="ward")

plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Distance")
plt.show()

model = AgglomerativeClustering(n_clusters=5, linkage="ward")

df["Cluster"] = model.fit_predict(X_scaled)

print(df.head())

plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=df["Cluster"]
)

plt.title("Agglomerative Clustering")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.show()
------------------------------------------------------------------------------------------

# IR 5 (7th)
Build the web crawler to pull product information and links from an e-commerce website.

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

#OR#

import requests
import pandas as pd

from bs4 import BeautifulSoup
from urllib.parse import urljoin

url = "https://books.toscrape.com/catalogue/page-1.html"

response = requests.get(url)

soup = BeautifulSoup(response.text, "html.parser")

products = []

books = soup.find_all("article", class_="product_pod")

for book in books:

    title = book.h3.a["title"]

    price = book.find("p", class_="price_color").text

    link = book.h3.a["href"]

    full_link = urljoin(url, link)

    products.append({
        "Title": title,
        "Price": price,
        "Link": full_link
    })

df = pd.DataFrame(products)

print(df)

df.to_csv("products.csv", index=False)
------------------------------------------------------------------------------------------
_____________________________________________________________________________________________________________
VIVA Questions for IR with algos

PART A: PRACTICAL-WISE VIVA QUESTIONS
1. INFORMATION RETRIEVAL (IR) PRACTICALS
Practical 1: Text preprocessing – Stop-word removal and stemming
Aim: To preprocess a text document by removing stop words and performing stemming.
A. Theory to write in the practical exam
Theory:

Text preprocessing is the process of cleaning and preparing raw text before applying information retrieval or NLP techniques.

It involves:
1. Tokenization: Dividing text into individual words.
2. Lowercasing: Converting all text into lowercase.
3. Stop-word removal: Removing common words such as is, the, and, are.
4. Stemming: Reducing words to their root form.
5. Cleaning: Removing unnecessary punctuation and special characters.
Example:

Input: The students are studying information retrieval.

Output after preprocessing: student studi inform retriev
B. One-line algorithm
1. Start.
2. Import the required NLP libraries.
3. Take a text document as input.
4. Convert text into lowercase.
5. Tokenize the text into words.
6. Remove stop words.
7. Apply stemming to remaining words.
8. Display the preprocessed text.
9. Stop.
C. Practical viva questions and answers
Q1. What is text preprocessing?

It is the process of cleaning and preparing raw text for further analysis.

Q2. What is tokenization?

Dividing a sentence into smaller units called tokens.

Q3. What are stop words?

Common words that usually carry little information, such as is, the, are, an, and.

Q4. Why do we remove stop words?

To reduce unnecessary words and improve processing efficiency.

Q5. What is stemming?

Stemming reduces words to their root or stem by removing prefixes or suffixes.

Q6. Give an example of stemming.

Playing → Play, Studies → Studi, Connected → Connect.

Q7. What is lemmatization?

Lemmatization converts words into their meaningful dictionary base form.

Example: Better → Good.

Q8. Difference between stemming and lemmatization?
Stemming	Lemmatization
Removes word endings	Finds dictionary base form
May produce meaningless words	Produces meaningful words
Faster	Comparatively slower
Example: Studies → Studi	Example: Studies → Study

Q9. What is NLTK?

NLTK stands for Natural Language Toolkit. It is a Python library for natural language processing.

Q10. What is Porter Stemmer?

It is a stemming algorithm used to remove common suffixes from English words.

Q11. What is NLP?

Natural Language Processing is a branch of AI that enables computers to understand and process human language.

Q12. What is the difference between a token and a sentence?

A sentence is a complete statement, while a token is an individual word or unit obtained from it.

Q13. Why do we convert text to lowercase?

To treat uppercase and lowercase versions of the same word equally.

Q14. Where is text preprocessing used?

Search engines, chatbots, spam detection, sentiment analysis and document classification.

Q15. What is corpus?

A corpus is a collection of text documents used for language processing.

Practical 2: Document retrieval using inverted files
Aim: To implement document retrieval using an inverted index.
A. Theory to write
An inverted index is a data structure that stores each unique word along with the list of documents in which that word appears.

It helps retrieve documents quickly without searching every document individually.

Example:

Documents:
- D1: Information Retrieval System
- D2: Information Storage System
- D3: Retrieval of Documents
Inverted index:
Word	Document IDs
Information	D1, D2
Retrieval	D1, D3
System	D1, D2
Storage	D2
Documents	D3

When a user searches for Retrieval, the system directly returns D1 and D3.
B. One-line algorithm
1. Start.
2. Take multiple documents as input.
3. Preprocess the documents.
4. Tokenize each document.
5. Create an empty inverted index.
6. Store each word with its document ID.
7. Accept a search query.
8. Search the inverted index for matching terms.
9. Display the relevant documents.
10. Stop.
C. Practical viva questions and answers
Q1. What is an inverted index?

It is a data structure that maps each word to the documents containing that word.

Q2. Why is it called an inverted index?

Because it reverses the relationship between documents and words.

Q3. What is indexing?

Indexing is the process of organizing documents so that information can be retrieved quickly.

Q4. What is a forward index?

A forward index maps each document to the words it contains.

Q5. Difference between forward and inverted index?
- Forward index: Document → Words.
- Inverted index: Word → Documents.
Q6. What is a posting list?

A posting list contains the document IDs in which a particular term appears.

Q7. What is a dictionary in an inverted index?

It stores all unique terms present in the document collection.

Q8. What is a query?

A query is a user's search input.

Q9. What is Boolean retrieval?

It retrieves documents using Boolean operators such as AND, OR and NOT.

Q10. Explain AND, OR and NOT.
- AND: Both conditions must be true.
- OR: At least one condition must be true.
- NOT: Excludes a particular term.
Q11. What is document frequency?

The number of documents containing a particular term.

Q12. What is term frequency?

The number of times a term appears in a particular document.

Q13. What are the advantages of an inverted index?
- Faster searching.
- Efficient storage of searchable terms.
- Supports Boolean queries.
- Suitable for large document collections.
Q14. Where is an inverted index used?

Google, search engines, digital libraries and document search systems.

Q15. What is the difference between indexing and searching?

Indexing organizes the documents, while searching retrieves relevant documents based on a query.

Practical 3: Email spam filtering using text classification
Aim: To classify emails as Spam or Ham using a machine learning classification algorithm.
A. Theory to write
Email spam filtering is a text classification technique used to identify unwanted emails.

A machine learning model learns from labelled emails and predicts whether a new email is Spam or Ham.

Steps:
1. Collect an email dataset.
2. Clean and preprocess the text.
3. Convert text into numerical features using TF-IDF.
4. Split the dataset into training and testing sets.
5. Train a classification model.
6. Predict whether emails are Spam or Ham.
7. Evaluate the model using accuracy, precision and recall.
Algorithms used:
- Naive Bayes
- Logistic Regression
- Support Vector Machine
- Decision Tree
B. One-line algorithm
1. Start.
2. Import required libraries.
3. Load the spam email dataset.
4. Remove missing values.
5. Separate email text and labels.
6. Convert text into numerical features using TF-IDF.
7. Split data into training and testing sets.
8. Train a classification model.
9. Predict Spam or Ham.
10. Calculate performance metrics.
11. Display the results.
12. Stop.
C. Practical viva questions and answers
Q1. What is spam?

Spam is an unwanted or unsolicited email, usually sent for advertising, fraud or malicious purposes.

Q2. What is Ham?

Ham refers to a genuine or legitimate email.

Q3. What is text classification?

It is the process of assigning predefined categories to text documents.

Q4. What is supervised learning?

A machine learning technique in which a model learns from labelled training data.

Q5. Which algorithm is commonly used for spam detection?

Naive Bayes is commonly used because it works well with text classification.

Q6. What is Naive Bayes?

It is a probabilistic classification algorithm based on Bayes' theorem and the assumption of conditional independence between features.

Q7. What is TF-IDF?

TF-IDF stands for Term Frequency–Inverse Document Frequency. It measures the importance of a word in a document relative to a collection of documents.

Q8. Why is TF-IDF used?

It converts text into numerical values so machine learning algorithms can process it.

Q9. What is CountVectorizer?

It converts text into numerical features based on word occurrence counts.

Q10. What is a training dataset?

The data used to teach the machine learning model.

Q11. What is a testing dataset?

The data used to evaluate the trained model on unseen examples.

Q12. What is classification?

Predicting a category or class for given input data.

Q13. What is a confusion matrix?

A table that compares actual and predicted classifications.

Q14. What is accuracy?

The percentage of total predictions that are correct.
\[
Accuracy=\frac{TP+TN}{TP+TN+FP+FN}
\]

Q15. What are precision and recall?
- Precision: Out of all emails predicted as spam, how many were actually spam?
- Recall: Out of all actual spam emails, how many were correctly detected?
Q16. What is F1-score?

The harmonic mean of precision and recall.
\[
F1=2\times\frac{Precision\times Recall}{Precision+Recall}
\]

Q17. What is overfitting?

When a model performs well on training data but poorly on unseen testing data.

Q18. What is underfitting?

When a model fails to learn the important patterns in the training data.

Q19. What is a false positive in spam detection?

A genuine email incorrectly classified as spam.

Q20. What is a false negative?

A spam email incorrectly classified as genuine.

Q21. What is the use of train_test_split()?

It divides the dataset into training and testing subsets.

Q22. Which Python libraries are used?

Pandas, NumPy and Scikit-learn.

Q23. What is Scikit-learn?

A Python library used for machine learning, classification, regression and clustering.

Q24. What is a label?

A label is the expected output or category, such as Spam or Ham.

Practical 4: Agglomerative hierarchical clustering
Aim: To implement Agglomerative Hierarchical Clustering on a suitable dataset.
A. Theory to write
Agglomerative clustering is an unsupervised machine learning algorithm that groups similar data points into clusters.

It follows a bottom-up approach. Initially, every data point is considered an individual cluster. The algorithm repeatedly merges the closest clusters until the required number of clusters is obtained.

Steps:
1. Consider every data point as a separate cluster.
2. Calculate distances between clusters.
3. Find the two nearest clusters.
4. Merge them.
5. Recalculate distances.
6. Repeat until the required clusters are formed.
Dendrogram: A tree-like diagram that shows how clusters are merged.
B. One-line algorithm
1. Start.
2. Import the required libraries.
3. Load the dataset.
4. Select numerical features.
5. Standardize the data if required.
6. Calculate hierarchical distances.
7. Create a dendrogram.
8. Apply Agglomerative Clustering.
9. Assign cluster labels.
10. Visualize the clusters.
11. Stop.
C. Practical viva questions and answers
Q1. What is clustering?

Clustering is the process of grouping similar data points together.

Q2. What is unsupervised learning?

Learning from data without predefined output labels.

Q3. What is hierarchical clustering?

It is a clustering method that creates a hierarchy of clusters.

Q4. What are the two types of hierarchical clustering?
1. Agglomerative clustering.
2. Divisive clustering.
Q5. What is Agglomerative clustering?

It is a bottom-up clustering method in which individual clusters are repeatedly merged.

Q6. What is Divisive clustering?

It is a top-down method in which one large cluster is repeatedly divided into smaller clusters.

Q7. What is a dendrogram?

A tree-like diagram that represents the merging or splitting of clusters.

Q8. What is linkage?

Linkage determines how the distance between two clusters is calculated.

Q9. Name the types of linkage.
- Single linkage.
- Complete linkage.
- Average linkage.
- Ward linkage.
Q10. What is single linkage?

It calculates the minimum distance between points of two clusters.

Q11. What is complete linkage?

It calculates the maximum distance between points of two clusters.

Q12. What is average linkage?

It calculates the average distance between all pairs of points from two clusters.

Q13. What is Ward linkage?

It merges clusters by minimizing the increase in within-cluster variance.

Q14. What is Euclidean distance?

It is the straight-line distance between two data points.
\[
d=\sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}
\]

Q15. What is the difference between K-Means and Agglomerative clustering?
K-Means	Agglomerative
Partition-based	Hierarchical
Requires K initially	Can inspect dendrogram
Uses centroids	Uses linkage
Iteratively updates centroids	Repeatedly merges clusters

Q16. What is the use of StandardScaler?

It standardizes features to have approximately zero mean and unit variance.

Q17. What is the use of SciPy?

SciPy is a Python library used for scientific computing and hierarchical clustering.

Q18. What is the use of Scikit-learn?

It provides machine learning algorithms, including AgglomerativeClustering.

Q19. Is clustering supervised or unsupervised?

Unsupervised learning.

Q20. What are the applications of clustering?

Customer segmentation, image segmentation, document grouping and biological data analysis.

Practical 5: Web crawler for e-commerce product information
Aim: To develop a web crawler to extract product information and links from an e-commerce website.
A. Theory to write
A web crawler is an automated program that visits web pages and collects information.

It is also called a spider or bot.

For an e-commerce website, it can collect:
- Product name.
- Product price.
- Product description.
- Product URL.
- Product image link.
Libraries used:
- Requests: To send HTTP requests.
- BeautifulSoup: To parse HTML.
- Pandas: To store extracted information in tabular form.
B. One-line algorithm
1. Start.
2. Import Requests and BeautifulSoup.
3. Specify the website URL.
4. Send an HTTP request to the website.
5. Retrieve the HTML page.
6. Parse the HTML using BeautifulSoup.
7. Locate product details using HTML tags and CSS selectors.
8. Extract product names, prices and links.
9. Store the extracted information in a list or DataFrame.
10. Display or save the results.
11. Stop.
C. Practical viva questions and answers
Q1. What is a web crawler?

A web crawler is an automated program that visits web pages and collects information.

Q2. What is web scraping?

Web scraping is the process of extracting specific information from web pages.

Q3. Difference between crawling and scraping?
- Crawling: Discovering and visiting web pages.
- Scraping: Extracting specific data from web pages.
Q4. What is HTML?

HTML stands for HyperText Markup Language. It is used to structure web pages.

Q5. What is BeautifulSoup?

BeautifulSoup is a Python library used to parse HTML and XML documents.

Q6. What is Requests?

Requests is a Python library used to send HTTP requests and receive responses.

Q7. What is an HTTP request?

A request sent by a client to a web server to access a resource.

Q8. What is an HTTP response?

The reply received from the server after processing a request.

Q9. What is status code 200?

It means the HTTP request was successful.

Q10. What is status code 404?

It means the requested page or resource was not found.

Q11. What is a URL?

URL stands for Uniform Resource Locator. It is the address of a resource on the internet.

Q12. What is a CSS selector?

It is a pattern used to select HTML elements based on their tags, classes or IDs.

Q13. What is a tag?

An HTML element identifier such as p, h1, a or div.

Q14. What is an attribute?

An attribute provides additional information about an HTML element.

Example: href in an anchor tag.

Q15. What is the href attribute?

It specifies the destination URL of a hyperlink.

Q16. What is robots.txt?

A file that provides crawling instructions to web crawlers.

Q17. What is pagination?

Dividing website content into multiple pages.

Q18. What is User-Agent?

A header that identifies the client making an HTTP request.

Q19. What is the use of Pandas?

It is used for data manipulation and analysis.

Q20. Is web scraping allowed on every website?

No. We should follow website terms, access restrictions, robots instructions and applicable laws.

Theory questions
_________________
PART B: IR THEORY QUESTIONS AND ANSWERS
These questions cover the important IR syllabus concepts, especially Unit I, along with indexing, retrieval, evaluation, text classification and web search concepts connected to your practicals.
Unit I: Introduction to Information Retrieval
Q1. What is Information Retrieval?

Information Retrieval (IR) is the process of finding relevant information or documents from a large collection based on a user's query.

Example: Searching for machine learning notes on Google.

Q2. What are the main components of an IR system?
1. Document collection.
2. Document preprocessing.
3. Indexing.
4. Query processing.
5. Matching and ranking.
6. Retrieval of results.
Q3. Draw the block diagram of an IR system.
Document Collection


Document Preprocessing


Indexing / Inverted Index


Query Processing


Matching and Ranking


Relevant Search Results


Q4. What are the objectives of an IR system?
- Retrieve relevant documents.
- Reduce search time.
- Improve search accuracy.
- Organize large document collections.
- Provide relevant information to users.
Q5. What are the challenges in Information Retrieval?
1. Large document collections.
2. Ambiguous queries.
3. Synonyms and different word forms.
4. Language differences.
5. Irrelevant results.
6. Information overload.
Q6. What is the difference between Information Retrieval and Data Retrieval?
Information Retrieval	Data Retrieval
Retrieves relevant information	Retrieves exact data
Often handles unstructured data	Often handles structured data
Results may be ranked	Results generally match conditions
Relevance-based	Exact-match based
Example: Google Search	Example: SQL query

Q7. What are the applications of IR?

Search engines, digital libraries, e-commerce, medical information systems and document management.

Q8. What are the different IR models?
1. Boolean model.
2. Vector Space Model.
3. Probabilistic model.
4. Latent Semantic Indexing model.
Unit I: Boolean Retrieval Model
Q9. What is the Boolean model?

It is an IR model that retrieves documents based on Boolean operators AND, OR and NOT.

Q10. What are the advantages of the Boolean model?
- Simple to understand.
- Easy to implement.
- Supports complex queries.
- Gives exact matching results.
Q11. What are its disadvantages?
- Does not rank documents by relevance.
- Requires users to understand Boolean operators.
- Exact matching can miss relevant documents.
Q12. Explain Boolean operators with examples.

Suppose:
- D1 = AI and ML
- D2 = AI and Data Science
- D3 = ML and Python
Query	Result
AI AND ML	D1
AI OR ML	D1, D2, D3
AI NOT ML	D2

Q13. What is an exact-match retrieval system?

A system that returns documents satisfying the specified query conditions.
Unit I: Vector Space Model
Q14. What is Vector Space Model?

It represents documents and queries as numerical vectors and calculates their similarity.

Q15. What is a document vector?

A numerical representation of the terms present in a document.

Q16. What is cosine similarity?

It measures the similarity between two vectors using the cosine of the angle between them.
\[
Cosine(A,B)=\frac{A\cdot B}{|A||B|}
\]

Q17. What is the range of cosine similarity?

For nonnegative term-weight vectors, the range is 0 to 1.
- 0 = No similarity.
- 1 = Maximum similarity.
Q18. What is TF-IDF?

It is a term-weighting method used to measure the importance of a word in a document relative to a document collection.

Q19. Explain TF and IDF.
- TF: Term Frequency, how often a word appears in a document.
- IDF: Inverse Document Frequency, how rare or informative a word is across documents.
\[
IDF(t)=\log\frac{N}{DF(t)}
\]

\[
TFIDF=TF\times IDF
\]

Where:
- N = Total number of documents.
- DF = Number of documents containing the term.
Q20. What are the advantages of Vector Space Model?
- Supports document ranking.
- Calculates similarity.
- Allows partial matching.
- Simple mathematical representation.
Q21. What are the disadvantages of Vector Space Model?
- Ignores word order.
- May ignore semantic meaning.
- High-dimensional vectors can be expensive to process.
Unit I: Probabilistic Model
Q22. What is a Probabilistic IR model?

It retrieves documents based on the probability that they are relevant to a user's query.

Q23. What is the Probability Ranking Principle (PRP)?

It states that documents should be ranked in decreasing order of their probability of relevance to the query.

Q24. What is relevance?

Relevance measures how useful a retrieved document is for a user's information need.

Q25. What is Bayes' theorem?

It is a mathematical theorem used to calculate conditional probability.
\[
P(A|B)=\frac{P(B|A)P(A)}{P(B)}
\]

Q26. What are the advantages of probabilistic models?
- Rank documents by estimated relevance.
- Handle uncertainty.
- Support relevance-based searching.
Unit I: Latent Semantic Indexing (LSI)
Q27. What is LSI?

Latent Semantic Indexing is an information retrieval technique that identifies hidden relationships between terms and documents.

Q28. Which mathematical technique is used in LSI?

Singular Value Decomposition (SVD).

Q29. What is SVD?

Singular Value Decomposition is a matrix factorization technique used to represent a matrix using three matrices.
\[
A=U\Sigma V^T
\]

Q30. Why is LSI used?

To identify semantic relationships and reduce dimensionality in document representation.

Q31. What is dimensionality reduction?

Reducing the number of features while preserving important information.

Q32. What is the main advantage of LSI?

It can identify relationships between documents and terms even when they do not share exactly the same words.

Q33. What is the limitation of LSI?

It can be computationally expensive and may lose some information during dimensionality reduction.
Unit II: Indexing and Query Processing
Q34. What is an index?

An index is an organized structure used to locate information quickly.

Q35. What is an inverted file?

A structure that stores terms and their corresponding document references.

Q36. What is a posting?

A document reference associated with a term in an inverted index.

Q37. What is a positional index?

An inverted index that also stores the positions of terms within documents.

Q38. Why is a positional index useful?

It supports phrase queries and proximity searches.

Q39. What is a phrase query?

A query that searches for an exact sequence of words.

Example: "Information Retrieval".

Q40. What is query processing?

The process of analyzing and preparing a user's search query before retrieving results.

Q41. What is query expansion?

Adding related words or synonyms to a query to improve retrieval.

Q42. What is query optimization?

Improving the execution of a search query to reduce processing time.

Q43. What is stemming in IR?

Reducing words to their stem to match different forms of the same word.

Q44. What is stop-word removal?

Removing frequently occurring words that generally provide little information for retrieval.
Unit III: Retrieval Evaluation and Performance
Q45. What is precision?

The proportion of retrieved documents that are relevant.
\[
Precision=\frac{Relevant\ Retrieved}{Total\ Retrieved}
\]

Q46. What is recall?

The proportion of all relevant documents that are retrieved.
\[
Recall=\frac{Relevant\ Retrieved}{Total\ Relevant}
\]

Q47. What is F-measure?

It combines precision and recall into one performance measure.
\[
F1=\frac{2PR}{P+R}
\]

Q48. What is accuracy?

The proportion of all correct predictions among total predictions.

Q49. What is a confusion matrix?

A table showing true positive, true negative, false positive and false negative predictions.

Q50. What is an ROC curve?

Receiver Operating Characteristic curve. It plots True Positive Rate against False Positive Rate at different classification thresholds.

Q51. What is AUC?

Area Under the ROC Curve. It measures how well a classifier distinguishes between classes.

Q52. What is MAP?

Mean Average Precision. It calculates the mean of average precision values across multiple queries.

Q53. What is MRR?

Mean Reciprocal Rank. It measures the average reciprocal rank of the first relevant result.

Q54. What is NDCG?

Normalized Discounted Cumulative Gain. It evaluates ranked search results by considering both relevance and ranking position.
Unit IV: Text Classification and Web Search
Q55. What is text categorization?

It is the process of assigning text documents to predefined categories.

Q56. Name some text classification algorithms.

Naive Bayes, SVM, Decision Tree, Logistic Regression and Random Forest.

Q57. What is a search engine?

A search engine is a software system that retrieves relevant information from a collection of web pages.

Q58. What are the main components of a search engine?
1. Crawler.
2. Indexer.
3. Search processor.
4. Ranking system.
5. Search interface.
Q59. What is a web crawler?

An automated program that visits and discovers web pages.

Q60. What is a web spider?

Another name for a web crawler.

Q61. What is web indexing?

The process of storing and organizing extracted webpage information for searching.

Q62. What is PageRank?

An algorithm originally developed by Google to estimate the importance of web pages based on their link structure.

Q63. What is anchor text?

The clickable text associated with a hyperlink.

Q64. What is a sitemap?

A file that lists important URLs of a website to help search engines discover pages.

Q65. What is robots.txt?

A file that specifies crawling rules for automated web crawlers.
_______________________________________________________________________________________________________
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

#OR#

import pandas as pd

df = pd.read_csv("human.csv")

sequence = df["sequence"][0]

sequence = sequence.upper()

print("DNA Sequence:")
print(sequence[:100])

# Calculate GC Content

gc_count = sequence.count("G") + sequence.count("C")

gc_content = (gc_count / len(sequence)) * 100

print("GC Content:", round(gc_content, 2), "%")

# Find motif

motif = "ATG"

positions = []

for i in range(len(sequence) - len(motif) + 1):

    if sequence[i:i+len(motif)] == motif:
        positions.append(i)

print("Motif:", motif)

print("Motif Positions:", positions[:20])

# Find ORFs in three forward reading frames

stop_codons = ["TAA", "TAG", "TGA"]

orfs = []

for frame in range(3):

    i = frame

    while i < len(sequence) - 2:

        if sequence[i:i+3] == "ATG":

            for j in range(i+3, len(sequence)-2, 3):

                codon = sequence[j:j+3]

                if codon in stop_codons:

                    orfs.append(sequence[i:j+3])
                    break

        i += 3

print("Number of ORFs:", len(orfs))

for orf in orfs[:10]:
    print("ORF:", orf[:60], "Length:", len(orf))
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

#OR#

import numpy as np
import pandas as pd

from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

np.random.seed(10)

genes = ["Gene_" + str(i) for i in range(1, 101)]

control = np.random.poisson(100, (100, 3))

treated = np.random.poisson(100, (100, 3))

# Simulate upregulated genes
treated[:10] = np.random.poisson(250, (10, 3))

# Simulate downregulated genes
treated[10:20] = np.random.poisson(35, (10, 3))

data = np.column_stack((control, treated))

df = pd.DataFrame(
    data,
    index=genes,
    columns=[
        "Control1", "Control2", "Control3",
        "Treated1", "Treated2", "Treated3"
    ]
)

print("Gene Expression Data:")
print(df.head())

# Calculate mean expression

control_mean = df.iloc[:, :3].mean(axis=1)

treated_mean = df.iloc[:, 3:].mean(axis=1)

# Calculate log2 Fold Change

log2fc = np.log2((treated_mean + 1) / (control_mean + 1))

# Calculate p-values

p_values = []

for i in range(len(df)):

    p = ttest_ind(
        df.iloc[i, :3],
        df.iloc[i, 3:],
        equal_var=False
    ).pvalue

    p_values.append(p)

# Multiple testing correction

adjusted_p = multipletests(
    p_values,
    method="fdr_bh"
)[1]

result = pd.DataFrame({
    "Control_Mean": control_mean,
    "Treated_Mean": treated_mean,
    "Log2_Fold_Change": log2fc,
    "P_Value": p_values,
    "Adjusted_P_Value": adjusted_p
})

result["Expression"] = "Not Significant"

result.loc[
    (result["Adjusted_P_Value"] < 0.05) &
    (result["Log2_Fold_Change"] > 1),
    "Expression"
] = "Upregulated"

result.loc[
    (result["Adjusted_P_Value"] < 0.05) &
    (result["Log2_Fold_Change"] < -1),
    "Expression"
] = "Downregulated"

print(result.head(20))

print(result["Expression"].value_counts())

result.to_csv("differential_expression_results.csv")

------------------------------------------------------------------------------------------
# BIF 3

#cell 1
import os
import sys
import glob
import subprocess
import sysconfig
import numpy as np
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO, Select
from scipy.spatial.distance import cdist

scripts = Path(sysconfig.get_path("scripts"))

print("Libraries imported successfully!")

#cell 2
class ProteinSelect(Select):
    def accept_chain(self, chain):
        return chain.id == "A"

    def accept_residue(self, residue):
        return residue.id[0] == " "

    def accept_atom(self, atom):
        return atom.get_altloc() in (" ", "A")

parser = PDBParser(QUIET=True)
structure = parser.get_structure("protein", "6m0j.pdb")

io = PDBIO()
io.set_structure(structure)
io.save("ACE2.pdb", ProteinSelect())

cmd = [
    str(scripts / "mk_prepare_receptor.exe"),
    "--read_pdb", "ACE2.pdb",
    "-o", "ACE2_receptor",
    "-p", "-j",
    "--default_altloc", "A"
]

result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout)
print(result.stderr)

#cell 3
cmd = [
    str(scripts / "mk_prepare_ligand.exe"),
    "-i", "Conformer3D_COMPOUND_CID_135398740.sdf",
    "-o", "ligand.pdbqt"
]

result = subprocess.run(cmd, capture_output=True, text=True)

print(result.stdout)
print("Ligand prepared:", os.path.exists("ligand.pdbqt"))

#cell 4
p = PDBParser(QUIET=True).get_structure("p", "6m0j.pdb")[0]

A = np.array([
    a.coord for a in p["A"].get_atoms()
    if a.element not in ("H", "D")
])

E = np.array([
    a.coord for a in p["E"].get_atoms()
    if a.element not in ("H", "D")
])

d = cdist(A, E)

center = A[np.any(d < 5, axis=1)].mean(axis=0)

print("Docking centre:", center)

#cell 5
cmd = [
    "vina.exe",
    "--receptor", "ACE2_receptor.pdbqt",
    "--ligand", "ligand.pdbqt",
    "--center_x", str(center[0]),
    "--center_y", str(center[1]),
    "--center_z", str(center[2]),
    "--size_x", "24",
    "--size_y", "24",
    "--size_z", "24",
    "--exhaustiveness", "8",
    "--num_modes", "5",
    "--out", "docked_pose.pdbqt"
]

result = subprocess.run(cmd, capture_output=True, text=True)

print(result.stdout)
print(result.stderr)

#cell 6
with open("docked_pose.pdbqt") as f:
    for line in f:
        if "VINA RESULT" in line:
            print(line.strip())

#cell 7
results = []

for file in glob.glob("ligands/*.sdf"):

    name = Path(file).stem

    subprocess.run([
        str(scripts / "mk_prepare_ligand.exe"),
        "-i", file,
        "-o", name + ".pdbqt"
    ], capture_output=True, text=True, check=True)

    subprocess.run([
        "vina.exe",
        "--receptor", "ACE2_receptor.pdbqt",
        "--ligand", name + ".pdbqt",
        "--center_x", str(center[0]),
        "--center_y", str(center[1]),
        "--center_z", str(center[2]),
        "--size_x", "24",
        "--size_y", "24",
        "--size_z", "24",
        "--out", name + "_docked.pdbqt"
    ], capture_output=True, text=True, check=True)

    with open(name + "_docked.pdbqt") as f:
        for line in f:
            if "VINA RESULT" in line:
                score = float(line.split()[3])
                results.append([name, score])
                break

print("Virtual screening completed!")

import pandas as pd

df = pd.DataFrame(results, columns=["Ligand", "Binding Affinity"])
df = df.sort_values("Binding Affinity")

display(df)

#OR#

Iconda tos accept --override-channels-channel https://repo.anaconda.com/pkgs/main accepted Terms of Service for https://repo.anaconda.com/pkgs/main

↑↓古早

ENG

[4]:

Iconda tos accept --override-channels--channel https://repo.anaconda.com/pkgs/r accepted Terms of Service for https://repo.anaconda.com/pkgs/r

[5]:

Iconda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2 accepted Terms of Service for https://repo.anaconda.com/pkgs/msys2

[1]: import urllib.request

url = "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.7/vina_1.2.7_win.exe" urllib.request.urlretrieve(url, "vina.exe")

print("Vina downloaded successfully!")

Vina downloaded successfully!

[2]: import os

print(os.path.exists("vins.exe"))

True
[3]: import subprocess

result subprocess.run(

["vina.exe", "version"], capture_output=True,

text=True

print(result.stdout)

print(result.stderr)

AutoDock Vina v1.2.7

[4]: import os

print(os.listdir())

['antigravity-ide', 'cache', '.claude', 'claude.json', 'corda, copilot', 'gemini', 'git-for-windows-updater', '.ipynb_checkpoints', 'ipython", jupyter', 'matplotlib', '.vscode', 'vscode-shared', 'om@j.pdb", "AppData', 'awesome-11m-apps', 'car_evaluation.csv', 'CC GENERAL.csv', 'Cisco Packet Tracer 8.2.2', 'city_day.csv', 'city_hour.csv", "Claude', 'Cleaned Delhi_AQI_Data.csv', 'Cleaned_RealEstate_Data.csv', 'Cleaned Retail_Sales_Data.csv", *Cleaned_Sales_Data.csv', 'Cleaned_Telecom_Data.csv", "Clustered_Customers.csv", "Conformer3D_COMPOUND CID_135398748.sdf", "Contacts', 'CrossDevice', 'D esktop', 'differential_expression_results.csv', 'document.txt", "Documents', 'Downloads', 'face recognition', 'Favorites', 'heart disease.csv", "human.c sv', 'Intel Graphics Profiles', 'Iris.csv', 'Iris_Clustered.csv', 'LeetCode", "Links", "Mall_Customers.csv', 'Music', 'myenv', 'NTUSER.DAT', 'ntuser.dat. LOG1', 'ntuser.dat.LOG2', 'NTUSER.DAT (@df87728-6060-11f8-a4fa-000c295c6f9a).TM.bLF', 'NTUSER.DAT(edf87728-6064-11f0-a4fa-000c295c6f9a). TMContainer@@@@@@ 00000000000001.regtrans-ms', 'NTUSER.DAT{edfe7720-6062-11f0-a4fa-000c295c6f9a).TMContainer@@@@@@@@@@@@@@@@@@2.regtrans-ms', 'ntuser.ini", "OneDrive", 'PCA_Output.csv', 'products.csv', 'realestate_prices.csv', 'Retail Sales Data.csv', 'sales.csv', 'sales data.csv', 'sales_data.json', 'sales_data.xlsx', "Saved Games', 'Searches', 'spam.csv', 'spam_ham_dataset.csv', 'stations.csv', 'station_day.csv', 'station_hour.csv", 'telecom customer churn.csv", "tel ecom_data_dictionary.csv', 'telecom_zipcode_population.csv', 'tiya', 'tiyal', 'uber.csv', 'Untitled.ipynb', 'Untitled1.ipynb", "venv", "Videos", 'vina.e xe', 'Wine.csv', 'pycache']
[8]:

import sys

![sys.executable) -m pip install gemmi

Collecting gemmi

Downloading gemmi-0.7.5-cp313-cp313-win_amd64.whl.metadata (2.3 kB)

Downloading gemmi-0.7.5-cp313-cp313-win_amd64.whl (2.3 MB)

0.0/2.3 MB ? eta:

0.5/2.3 MB 3.0 MB/s eta 0:00:01

1.0/2.3 MB 4.7 MB/s eta 0:00:01

1.0/2.3 MB 4.7 MB/s eta 0:00:01

1.0/2.3 MB 4.7 MB/s eta 0:00:01

1.6/2.3 MB 1.4 MB/s eta 0:00:01

2.3/2.3 MB 1.7 MB/s eta 0:00:00

Installing collected packages: gemmi

Successfully installed gemmi-0.7.5
[2] import subprocess

import sysconfig

from pathlib import Path

scripts Path (sysconfig.get_path("scripts"))

cmd = [

str(scripts/"mk_prepare_receptor.exe"), "--read_pdb", "6m0j.pdb", "-o", "receptor", "-p", "j", "-d", "A:903,904,905,906, E:601", "--default_altloc", "A"

result = subprocess.run(cmd, capture_output=True, text=True)

print("Return code:", result.returncode)

print(result.stdout)

print(result.stderr)

Return code: e

Template padding will be used.

gasteiger charges will be read from template file

Files written:

receptor.json parameterized receptor

receptor.pdbqt static (i.e., rigid) receptor input file
import os

print("Receptor PDBQT:", os.path.exists("receptor.pdbqt"))

print("Receptor JSON:", os.path.exists("receptor.json"))

Receptor PDBQT: True

Receptor 350N: True

[4]: import subprocess

import sysconfig from pathlib import Path

scripts Path(sysconfig.get_path("scripts"))

cmd = [

str(scripts/ "mk_prepare_ligand.exe"), "-1", "Conformer3D_COMPOUND_CID_135398740.sdf", "-o", "ligand.pdbqt"

result = subprocess.run(cmd, capture_output=True, text=True)

print("Return code:", result.returncode)

print(result.stdout)

print(result.stderr)

Return code: 6

Input molecules processed: 1, skipped:

PDBQT files written: 1

PDBQT files not written due to error:

Input molecules with errors: 8

リ

125°C
[5]: import os

print("Ligand prepared:", os.path.exists("ligand.pdbqt"))

Ligand prepared: True

[

6]: from Bio.PDB import PDBParser from scipy.spatial.distance import cdist import numpy as np

p= PDBParser (QUIET=True).get_structure("p", "6m0j.pdb")[0]

A= np.array([a.coord for a in p["A"].get_atoms() if a.element not in ("H", "0")])

E= np.array([a.coord for a in p["E"].get_atoms() if a.element not in ("H", "D")])

dcdist(A, Ε)

center= A[np.any(d< 5, axis=1)].mean(axis=0)

print("Docking centre:", center)

Docking centre: [-35.34925 29.908133 -1.0004]
[7]: import subprocess

cmd = [

"vina.exe",

"--receptor", "receptor.pdbqt",

"--ligand", "ligand.pdbqt",

"--center_x", str(center[0]),

"--center y", str(center[1]),

"--center_z", str(center[2]),

"-size x", "24",

"--size y", "24",

"size_z", "24",

"--exhaustiveness", "8",

"--num_modes", "5",

"--out", "docked_pose.pdbqt"

1

result = subprocess.run(cmd, capture_output=True, text=True)

print(result.stdout)

print(result.stderr)
AutoDock Vina v1.2.7

#If you used AutoDock Vina in your work, please cite:

#

#3. Eberhardt, D. Santos-Martins, A. F. Tillack, and S. Forli #

AutoDock Vina 1.2.0: New Docking Methods, Expanded Force

# Field, and Python Bindings, J. Chem. Inf. Model. (2021)

# DOI 10.1021/acs.jcim.1c00203

#0. Trott, A. J. Olson,

# AutoDock Vina: improving the speed and accuracy of docking # with a new scoring function, efficient optimization and

#multithreading, J. Comp. Chem. (2010)

#DOI 10.1002/jcc.21334

#

# Please see https://github.com/ccsb-scripps/AutoDock-Vina for

# more information.

Scoring function: vina

Rigid receptor: receptor.pdbqt

Ligand: ligand.pdbqt

Grid center: X -35.3492 Y 29.9081 2 -1.0004

Grid size X 24 Y 24 Z 24

Grid space: 0.375

Exhaustiveness: 8

CPU: 0

Verbosity: 1

#

#

#

Computing Vina grid done.

Performing docking (random seed: -1273144815)

0% 40 50 60 70 80 90 100%

10 20 30

mode

affinity | (kcal/mol) |

dist from best mode

rmsd l.b.

rmsd u.b.

1

-5.734

0

0

2

2.501

5.535

3

-5.607

3.172

5.654

4

-5.479

15.08

16.14

5

-5.24

-5.364

1.975

2.994

WARNING: At low exhaustiveness, it may be impossible to utilize all CPUs.
8]: with open("docked_pose.pdbqt") as f:

for line in f:

if "VINA RESULT" in line:

print(line.strip())

I

REMARK VINA RESULT:

-5.734

0.000

0.000

REMARK VINA RESULT:

-5.607

2.501

5.535

REMARK VINA RESULT:

-5.479

3.172

5.654

REMARK VINA RESULT:

-5.364

15.080

16.141

REMARK VINA R[9]:

import glob

import os

for file in glob.glob("ligands/*.sdf"): name= os.path.basename(file).replace(".sdf", "")

subprocess.run([

"mk_prepare_ligand.exe", "-1", file, "-o", name + ".pdbqt" ], capture_output=True, text=True, check=True)

subprocess.run([

"vina.exe", "--receptor", "receptor.pdbqt", "--ligand", name + ".pdbqt", "--center_x", str (center[0]), "--center_y", str (center[1]), "--center_z", str (center[2]), --size_x", "24", "--size_y", "24", "--size_z", "24", "--out", name + "_docked.pdbqt"

1, capture_output=True, text=True, check=True)

print("Virtual screening completed!")

Virtual screening completed!

[]:ESULT:

-5.240

1.975

2.994

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

#OR#

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("human.csv")

X = df["sequence"].astype(str).str.upper()

y = df["class"]

# Extract DNA 3-mer features

vectorizer = CountVectorizer(
    analyzer="char",
    ngram_range=(3, 3)
)

X = vectorizer.fit_transform(X)

# Split data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train Random Forest

model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Prediction

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Classification Report:")

print(classification_report(y_test, y_pred))
-----------------------------------------------------------------------
_________________________________________________________________________________________
Viva Questions with algos

2. BIOINFORMATICS (BIF) PRACTICALS
Practical 1: DNA Sequence Analysis
Aim: To analyze a DNA sequence and perform sequence manipulation, motif identification, GC content calculation and coding region identification.
A. Theory to write
DNA (Deoxyribonucleic Acid) is the hereditary material of living organisms.

DNA consists of four nitrogenous bases:
- A – Adenine
- T – Thymine
- G – Guanine
- C – Cytosine
DNA sequence analysis involves studying nucleotide sequences to identify important biological information.

Operations performed:
1. Sequence manipulation: Changing, extracting or modifying DNA sequences.
2. Motif identification: Finding specific patterns of nucleotides.
3. GC content: Calculating the percentage of G and C bases.
4. ORF identification: Finding possible protein-coding regions.
GC content formula:
\[
GC\%=\frac{G+C}{A+T+G+C}\times100
\]

Example:

DNA sequence: ATGCGCAT
- A = 2
- T = 2
- G = 2
- C = 2
\[
GC\%=\frac{2+2}{8}\times100=50\%
\]

ORF: An Open Reading Frame is a sequence of DNA that starts with a start codon and continues until a stop codon in the same reading frame.
- Start codon: ATG
- Stop codons: TAA, TAG, TGA.
B. One-line algorithm
1. Start.
2. Import Biopython.
3. Take a DNA sequence as input.
4. Calculate sequence length.
5. Count A, T, G and C bases.
6. Calculate GC percentage.
7. Search for a specific motif.
8. Identify possible ORFs using start and stop codons.
9. Display the results.
10. Stop.
C. Practical viva questions and answers
Q1. What is bioinformatics?

Bioinformatics is the application of computer science, mathematics and statistics to analyze biological data.

Q2. What is DNA?

DNA is the hereditary material that stores genetic information.

Q3. What is the full form of DNA?

Deoxyribonucleic Acid.

Q4. What are the four bases of DNA?

Adenine, Thymine, Guanine and Cytosine.

Q5. Which bases are complementary in DNA?
- A pairs with T.
- G pairs with C.
Q6. What is a nucleotide?

A nucleotide is the basic building block of DNA and RNA.

Q7. What is GC content?

It is the percentage of Guanine and Cytosine bases in a DNA sequence.

Q8. Why is GC content important?

It helps understand DNA composition, stability and certain genomic characteristics.

Q9. What is a motif?

A motif is a short, recurring sequence pattern that may have biological significance.

Q10. What is an ORF?

An Open Reading Frame is a continuous nucleotide sequence that could potentially encode a protein.

Q11. What is a codon?

A codon is a group of three nucleotides that specifies an amino acid or stop signal.

Q12. What is a start codon?

ATG is the usual DNA start codon. It corresponds to AUG in mRNA.

Q13. Name the three stop codons.

TAA, TAG and TGA.

Q14. What is a mutation?

A mutation is a change in the nucleotide sequence of DNA.

Q15. What is reverse complement?

It is the reverse of the complementary DNA sequence.

Q16. What is transcription?

The process of converting DNA information into RNA.

Q17. What is translation?

The process of converting mRNA information into a protein.

Q18. What is Biopython?

Biopython is an open-source Python library used for computational biology and bioinformatics.

Q19. What is Seq in Biopython?

Seq is a Biopython object used to represent and manipulate biological sequences.

Q20. What is FASTA format?

FASTA is a text-based format used to store DNA, RNA and protein sequences.

Q21. What is the difference between DNA and RNA?
DNA	RNA
Deoxyribose sugar	Ribose sugar
Usually double-stranded	Usually single-stranded
Contains Thymine	Contains Uracil
Stores genetic information	Helps in protein synthesis

Q22. What is a gene?

A gene is a segment of DNA that contains information to produce a functional product, such as a protein or RNA.

Q23. What is a genome?

The complete genetic material of an organism.

Q24. What is a reading frame?

A reading frame is the way a nucleotide sequence is divided into groups of three bases.

Q25. How many reading frames are present in double-stranded DNA?

Six possible reading frames: three on each strand.

Practical 2: RNA-Seq Data Analysis and Differential Gene Expression
Aim: To analyze RNA-Seq data and identify differentially expressed genes.
A. Theory to write
RNA-Seq stands for RNA Sequencing. It is a high-throughput sequencing technique used to study gene expression.

It helps identify which genes are active and how much they are expressed under different conditions.

Differential Gene Expression (DGE):

It is the process of identifying genes whose expression levels differ between two or more conditions.

For example:
- Control group: Healthy cells.
- Treatment group: Drug-treated cells.
By comparing both groups, we can identify genes whose expression increases or decreases after treatment.

Important terms:
1. Read counts: Number of sequencing reads assigned to a gene.
2. Normalization: Adjusting data for differences in sequencing depth and other technical factors.
3. Fold change: Measures how much gene expression changes.
4. p-value: Measures statistical evidence against the null hypothesis.
5. Adjusted p-value: Corrects for multiple hypothesis testing.
6. Upregulated genes: Genes with increased expression.
7. Downregulated genes: Genes with decreased expression.
Fold change formula:
\[
FC=\frac{\text{Expression in Treatment}}{\text{Expression in Control}}
\]

Log2 Fold Change:
\[
log_2FC=\log_2\left(\frac{Treatment}{Control}\right)
\]

Example:

Control expression = 10

Treatment expression = 40
\[
FC=40/10=4
\]

\[
log_2FC=2
\]

This means the gene expression increased four times.
B. One-line algorithm
1. Start.
2. Import the required libraries.
3. Load the RNA-Seq count dataset.
4. Perform data cleaning.
5. Normalize the expression values.
6. Separate control and treatment groups.
7. Calculate fold change and log2 fold change.
8. Calculate statistical significance and adjust p-values.
9. Identify upregulated and downregulated genes.
10. Visualize the results.
11. Stop.
C. Practical viva questions and answers
Q1. What is RNA-Seq?

RNA Sequencing is a technique used to measure and study gene expression.

Q2. What is the full form of RNA?

Ribonucleic Acid.

Q3. What is gene expression?

The process through which information from a gene is used to produce RNA or a protein.

Q4. What is differential gene expression?

It identifies genes that show different expression levels between experimental conditions.

Q5. What is a control group?

A reference group used for comparison with the treatment group.

Q6. What is a treatment group?

A group that receives a particular treatment or experimental condition.

Q7. What is fold change?

It is the ratio of gene expression between two conditions.

Q8. What is log2 fold change?

It is the base-2 logarithm of fold change, used to represent increases and decreases symmetrically.

Q9. What does positive log2FC indicate?

Positive log2FC indicates increased expression in the treatment group compared with the control group.

Q10. What does negative log2FC indicate?

Negative log2FC indicates decreased expression in the treatment group compared with the control group.

Q11. What does log2FC = 0 mean?

There is no change in expression between the two conditions.

Q12. What does log2FC = 1 mean?

Expression has doubled.

Q13. What does log2FC = -1 mean?

Expression has reduced to half.

Q14. What is an upregulated gene?

A gene whose expression increases under a particular condition.

Q15. What is a downregulated gene?

A gene whose expression decreases under a particular condition.

Q16. What is a p-value?

It measures how compatible the observed data are with the null hypothesis, assuming that hypothesis is true.

Q17. What is an adjusted p-value?

A p-value corrected for multiple statistical tests.

Q18. What is FDR?

False Discovery Rate. It controls the expected proportion of false discoveries among identified significant results.

Q19. What is a commonly used significance threshold?

Adjusted p-value or FDR less than 0.05 is commonly used.

Q20. What is normalization?

It adjusts expression data to make samples more comparable.

Q21. Why is normalization important?

Because different samples may have different sequencing depths and technical variations.

Q22. What is a volcano plot?

A graph used to visualize differentially expressed genes using log2 fold change and statistical significance.

Q23. What is a heatmap?

A graphical representation of data using different colours to show expression patterns.

Q24. What is a count matrix?

A table containing genes as rows, samples as columns and read counts as values.

Q25. What is DESeq2?

An R/Bioconductor package used for differential gene expression analysis of RNA-Seq count data.

Q26. What is the difference between RNA-Seq and microarray?
- RNA-Seq uses sequencing technology.
- Microarray uses probes to measure gene expression.
Q27. What is functional annotation?

Assigning biological information and functions to genes.

Q28. What is GO?

Gene Ontology. It describes gene functions using biological process, molecular function and cellular component categories.

Q29. What is KEGG?

Kyoto Encyclopedia of Genes and Genomes. It provides information about biological pathways and molecular interactions.

Q30. What is biological interpretation?

Understanding the biological meaning of the identified differentially expressed genes.

Practical 3: Molecular Docking and Virtual Screening
Aim: To perform molecular docking between a protein target and a small molecule ligand and identify potential drug candidates.
A. Theory to write
Molecular docking is a computational technique used to predict how a small molecule (ligand) binds to a target protein.

It helps understand the binding interaction and estimate the binding affinity.

Important terms:
- Protein: Biological molecule that performs important cellular functions.
- Ligand: Small molecule that binds to a protein.
- Receptor: Target protein to which a ligand binds.
- Binding site: Region of the protein where the ligand binds.
- Binding affinity: Strength of interaction between ligand and protein.
- Docking score: A numerical estimate of binding favourability.
Virtual screening:

Virtual screening is a computational process used to examine a large collection of chemical compounds and identify potential drug candidates.

Docking workflow:
1. Obtain the protein structure from the Protein Data Bank.
2. Prepare the protein by removing unnecessary molecules and adding required atoms.
3. Obtain the ligand structure.
4. Prepare the ligand.
5. Define the binding site or search space.
6. Perform molecular docking.
7. Analyze binding scores and poses.
8. Identify potential lead compounds.
Important software:
- AutoDock Vina.
- AutoDock Tools.
- PyMOL.
- RDKit.
- Biopython.
B. One-line algorithm
1. Start.
2. Download the target protein structure.
3. Obtain the ligand structure.
4. Prepare the protein and ligand.
5. Convert the required structures into PDBQT format.
6. Define the docking grid or search box.
7. Perform docking using AutoDock Vina.
8. Obtain docking scores and binding poses.
9. Compare the results.
10. Identify potential lead compounds.
11. Stop.
C. Practical viva questions and answers
Q1. What is molecular docking?

It is a computational technique used to predict the binding position and interaction between a ligand and a protein.

Q2. What is a ligand?

A small molecule that binds to a target protein.

Q3. What is a receptor?

A biological target, usually a protein, that binds to a ligand.

Q4. What is a protein?

A biological macromolecule made up of amino acids.

Q5. What is PDB?

Protein Data Bank. It is a database containing 3D structures of proteins and other biological macromolecules.

Q6. What is PDB ID?

A unique identifier assigned to a structure stored in the Protein Data Bank.

Q7. What is PDBQT?

A file format used by AutoDock Vina that stores molecular coordinates along with atom types and partial charges.

Q8. What is AutoDock Vina?

An open-source molecular docking software used to predict ligand-protein binding poses and scores.

Q9. What is RDKit?

An open-source cheminformatics toolkit used to process, analyze and manipulate chemical structures.

Q10. What is Biopython?

A Python library used for biological sequence and structure-related computational tasks.

Q11. What is binding affinity?

It describes how strongly a ligand binds to a protein.

Q12. What is the unit of docking score in AutoDock Vina?

kcal/mol.

Q13. What does a negative docking score mean?

It represents favourable predicted binding according to the docking scoring function.

Q14. Is a more negative Vina score generally better?

Yes. It generally indicates more favourable predicted binding, but it does not prove that a compound is an effective drug.

Q15. What is binding pose?

The predicted position and orientation of a ligand inside a protein binding site.

Q16. What is a binding pocket?

The region of a protein where a ligand can bind.

Q17. What is a hydrogen bond?

An attractive interaction involving a hydrogen atom and an electronegative atom such as oxygen or nitrogen.

Q18. What are hydrophobic interactions?

Interactions between nonpolar regions of molecules that tend to avoid contact with water.

Q19. What is virtual screening?

Computational screening of many compounds to identify potential drug candidates.

Q20. What is drug discovery?

The process of identifying and developing potential new medicines.

Q21. What is CADD?

Computer-Aided Drug Design. It uses computational methods to support drug discovery and development.

Q22. What is the difference between docking and virtual screening?
- Docking predicts the binding pose and score of a ligand.
- Virtual screening examines many compounds to find promising candidates, often using docking as one of its steps.
Q23. What is an RMSD?

Root Mean Square Deviation. It measures the difference between two sets of corresponding atomic coordinates.

Q24. What is ADMET?

Absorption, Distribution, Metabolism, Excretion and Toxicity.

Q25. Why do we remove water molecules from a protein structure?

Unnecessary water molecules may interfere with docking. However, structurally important water molecules should sometimes be retained.

Q26. What is a conformer?

A different spatial arrangement of the same molecule caused by rotation around chemical bonds.

Q27. What is a lead compound?

A promising molecule that can be further optimized during drug development.

Q28. What is a grid box?

A defined three-dimensional search region in which the docking algorithm searches for ligand binding poses.

Q29. What is exhaustiveness in AutoDock Vina?

It controls the computational search effort used to explore possible binding poses.

Q30. Can docking prove that a drug will work in humans?

No. Docking provides computational predictions that require experimental validation.

Practical 4: Machine Learning for Genomic Data
Aim: To apply machine learning algorithms such as Random Forest and SVM to classify genomic data.
A. Theory to write
Machine learning for genomic data involves applying machine learning algorithms to biological and genetic information to identify patterns and make predictions.

It can be used to classify diseases, identify genetic markers and predict biological conditions.

Algorithms:
1. Random Forest: An ensemble learning algorithm that combines multiple decision trees.
2. Support Vector Machine: A supervised learning algorithm that finds an optimal separating hyperplane between classes.
Workflow:
1. Load the genomic dataset.
2. Perform data preprocessing.
3. Select important features.
4. Divide the dataset into training and testing sets.
5. Train the ML model.
6. Predict the classes.
7. Evaluate the performance.
B. One-line algorithm
1. Start.
2. Import the required libraries.
3. Load the genomic dataset.
4. Clean the dataset.
5. Separate features and target labels.
6. Encode categorical variables if required.
7. Split the data into training and testing sets.
8. Train Random Forest or SVM.
9. Predict the test data.
10. Calculate accuracy, precision, recall and F1-score.
11. Display the results.
12. Stop.
C. Practical viva questions and answers
Q1. What is genomic data?

Genomic data contains information about the DNA and genetic material of an organism.

Q2. What is machine learning?

Machine learning is a branch of AI that enables computers to learn patterns from data and make predictions.

Q3. What is genomic classification?

It is the process of assigning genomic samples to predefined classes using their genetic features.

Q4. What is a feature?

A feature is an input variable used by a machine learning model.

Q5. What is a label?

A label is the output category that the model tries to predict.

Q6. What is Random Forest?

Random Forest is an ensemble algorithm that combines multiple decision trees to improve prediction.

Q7. Why is it called Random Forest?

Because it creates multiple decision trees using randomly selected samples and features.

Q8. What is a decision tree?

A tree-like model that makes decisions using conditions on input features.

Q9. What is SVM?

Support Vector Machine is a supervised learning algorithm used for classification and regression.

Q10. What is a hyperplane?

A decision boundary that separates different classes in a feature space.

Q11. What are support vectors?

The data points closest to the separating hyperplane. They help determine the decision boundary.

Q12. What is a kernel?

A mathematical function that helps SVM handle nonlinear relationships by implicitly mapping data into a higher-dimensional feature space.

Q13. Name the types of SVM kernels.
- Linear.
- Polynomial.
- RBF.
- Sigmoid.
Q14. What is an ensemble learning algorithm?

An algorithm that combines multiple models to improve prediction performance.

Q15. What is feature selection?

Selecting the most useful input features for building a model.

Q16. What is feature extraction?

Transforming raw data into useful numerical features.

Q17. What is a 3-mer?

A sequence of three consecutive nucleotides.

Example: ATG, GCA, TTC.

Q18. Why are k-mers used in genomic classification?

They convert DNA sequences into numerical features that machine learning algorithms can process.

Q19. What is cross-validation?

A technique used to evaluate a model by dividing data into multiple training and validation subsets.

Q20. What is overfitting?

When the model memorizes training data and performs poorly on unseen data.

Q21. What is accuracy?

The proportion of correctly classified samples.

Q22. What is a confusion matrix?

A table showing actual and predicted class values.

Q23. What is precision?

The proportion of correctly predicted positive cases among all predicted positive cases.

Q24. What is recall?

The proportion of actual positive cases correctly identified by the model.

Q25. What is F1-score?

The harmonic mean of precision and recall.

Q26. What is the difference between Random Forest and SVM?
Random Forest	SVM
Uses multiple decision trees	Uses a separating hyperplane
Ensemble algorithm	Margin-based algorithm
Handles nonlinear data well	Uses kernels for nonlinear data
Can provide feature importance	Depends on kernel and model interpretation

Q27. What is a genetic marker?

A measurable DNA sequence variation associated with a biological characteristic.

Q28. What is classification accuracy?

It measures how many samples the model predicts correctly out of all tested samples.

Q29. Which library is used for machine learning in Python?

Scikit-learn.

Q30. What are the applications of genomic machine learning?

Disease prediction, cancer classification, gene identification and genomic pattern recognition.
___________________________________________________________________________________________________
Theory Questions
__________________
PART C: BIF THEORY QUESTIONS AND ANSWERS
This section covers the main Elective III concepts: biological databases, sequence analysis, sequence alignment, protein structure, molecular modelling, docking, drug design, RNA-Seq, systems biology and genomic machine learning.
Unit I: Introduction to Bioinformatics and Biological Databases
Q1. What is Bioinformatics?

Bioinformatics combines biology, computer science, mathematics and statistics to store, analyze and interpret biological data.

Q2. What are the applications of Bioinformatics?
1. DNA sequence analysis.
2. Protein structure prediction.
3. Drug discovery.
4. Disease identification.
5. Genome analysis.
6. Gene expression analysis.
Q3. What is a biological database?

A biological database is an organized collection of biological information such as DNA, RNA, protein sequences and structures.

Q4. Name important biological databases.
- GenBank.
- EMBL/ENA.
- DDBJ.
- NCBI.
- UniProt.
- PDB.
- KEGG.
- GEO.
Q5. What is NCBI?

National Center for Biotechnology Information. It provides biological databases and bioinformatics tools.

Q6. What is GenBank?

GenBank is a public database of nucleotide sequences.

Q7. What is UniProt?

UniProt is a database containing protein sequence and functional information.

Q8. What is PDB?

Protein Data Bank. It stores three-dimensional structures of proteins and other biological macromolecules.

Q9. What is EMBL?

European Molecular Biology Laboratory. In sequence database contexts, EMBL/ENA refers to the European Nucleotide Archive.

Q10. What is DDBJ?

DNA Data Bank of Japan. It stores nucleotide sequence information.

Q11. What is GEO?

Gene Expression Omnibus. It is a public database for gene expression and other functional genomics data.

Q12. What is FASTA format?

A text-based format used to represent nucleotide or protein sequences.

Example:
>Sequence1
ATGCGTACGTAG


Q13. What is FASTQ format?

A format used to store biological sequences along with their sequencing quality scores.

Q14. What is the difference between FASTA and FASTQ?
FASTA	FASTQ
Stores sequences	Stores sequences and quality scores
Simple format	Contains four lines per read in standard representation
Starts with >	Starts with @

Q15. What is a genome?

The complete genetic material of an organism.

Q16. What is a proteome?

The complete set of proteins expressed by a cell, tissue or organism under particular conditions.

Q17. What is a transcriptome?

The complete set of RNA transcripts expressed by a cell or organism under a particular condition.

Q18. What is a gene?

A segment of DNA that contains instructions to produce a functional RNA or protein product.
Unit II: Sequence Analysis and Alignment
Q19. What is sequence alignment?

Sequence alignment is the process of arranging DNA, RNA or protein sequences to identify similarities and differences.

Q20. Why is sequence alignment performed?
- To identify similar sequences.
- To study evolutionary relationships.
- To identify conserved regions.
- To predict biological functions.
Q21. What are the types of sequence alignment?
1. Pairwise sequence alignment.
2. Multiple sequence alignment.
Q22. What is pairwise sequence alignment?

It compares two biological sequences to identify similarities.

Q23. What is Multiple Sequence Alignment (MSA)?

It aligns three or more biological sequences to identify conserved regions and similarities.

Q24. What is global alignment?

Global alignment aligns two sequences over their entire length.

Q25. What is local alignment?

Local alignment identifies the most similar regions within two sequences.

Q26. Difference between global and local alignment?
Global alignment	Local alignment
Compares entire sequences	Compares similar regions
Suitable for similar-length sequences	Suitable for sequences with local similarities
Needleman-Wunsch	Smith-Waterman

Q27. What is BLAST?

Basic Local Alignment Search Tool. It identifies similar sequences in biological databases.

Q28. What is FASTA in sequence searching?

FASTA is both a sequence file format and a family of tools used for sequence similarity searching.

Q29. What is the difference between BLAST and FASTA?

Both are used for sequence similarity searching. BLAST uses heuristic search methods, while FASTA uses its own heuristic approach based on sequence matches and refinement.

Q30. What is a scoring matrix?

A matrix used to assign scores to matches and mismatches during sequence alignment.

Q31. What is PAM?

Point Accepted Mutation. A substitution matrix used in protein sequence alignment.

Q32. What is BLOSUM?

Blocks Substitution Matrix. It is used to score amino acid substitutions in protein sequence alignment.

Q33. What is an amino acid?

An amino acid is the basic building block of a protein.

Q34. How many standard amino acids are used to build proteins?

20 standard amino acids.

Q35. What is a gap in sequence alignment?

A gap represents an insertion or deletion introduced to improve sequence alignment.

Q36. What is gap penalty?

A score penalty applied for introducing gaps during sequence alignment.

Q37. What is sequence similarity?

The degree of resemblance between two biological sequences.

Q38. What is sequence identity?

The percentage of exactly matching positions in an alignment.

Q39. What is homology?

Homology means that sequences share a common evolutionary origin.

Important: Homology is not expressed as a percentage. We say two sequences are homologous or not homologous.

Q40. What is E-value in BLAST?

E-value is the expected number of matches with a similar score that could occur by chance in a database search.

A lower E-value generally indicates a more statistically significant match.
Unit III: Protein Structure, Modelling and Molecular Docking
Q41. What is a protein?

A protein is a biological macromolecule made up of amino acids connected by peptide bonds.

Q42. What are the four levels of protein structure?
1. Primary.
2. Secondary.
3. Tertiary.
4. Quaternary.
Q43. Explain primary structure.

The linear sequence of amino acids in a protein.

Q44. Explain secondary structure.

Local folding patterns of proteins, mainly alpha helices and beta sheets.

Q45. Explain tertiary structure.

The overall three-dimensional folding of a single protein chain.

Q46. Explain quaternary structure.

The arrangement of two or more protein subunits into a functional protein complex.

Q47. What is protein structure prediction?

It is the process of predicting a protein's three-dimensional structure from its amino acid sequence or related structural information.

Q48. What is homology modelling?

A method of predicting a protein's 3D structure using a structurally known homologous protein as a template.

Q49. What are the steps in homology modelling?
1. Obtain the target sequence.
2. Search for a similar template.
3. Align target and template sequences.
4. Build the structural model.
5. Refine the model.
6. Validate the model.
Q50. What is a template?

A known protein structure used as a reference to predict another protein structure.

Q51. What is threading?

A protein structure prediction technique that fits a target amino acid sequence onto known protein structural folds.

Q52. What is ab initio modelling?

Predicting a protein's structure from its sequence using physical and computational principles without relying on a closely related template.

Q53. Differentiate homology modelling and threading.
Homology modelling	Threading
Uses a homologous protein template	Fits sequence onto known structural folds
Relies on sequence similarity	Can work even with low sequence similarity
Builds a model from alignment	Evaluates compatibility with structural frameworks

Q54. What is molecular modelling?

The computational representation and study of molecular structures and properties.

Q55. What is molecular docking?

Predicting how a ligand binds to a target protein.

Q56. What are the applications of molecular docking?
- Drug discovery.
- Identifying lead compounds.
- Studying protein-ligand interactions.
- Virtual screening.
Q57. What is CADD?

Computer-Aided Drug Design. It uses computational techniques to support drug discovery.

Q58. What are the two major approaches of CADD?
1. Structure-Based Drug Design (SBDD).
2. Ligand-Based Drug Design (LBDD).
Q59. What is Structure-Based Drug Design?

Drug design based on the 3D structure of the target protein.

Q60. What is Ligand-Based Drug Design?

Drug design based on the known properties and structures of active molecules.

Q61. What is pharmacophore modelling?

Identifying the essential chemical features required for a molecule to interact with a biological target.

Q62. What is virtual screening?

Computationally screening a large chemical library to identify promising drug candidates.

Q63. What is ADMET?

It describes the absorption, distribution, metabolism, excretion and toxicity properties of a drug candidate.

Q64. What is a lead molecule?

A molecule showing promising biological activity that can be optimized to develop a drug.

Q65. What is a binding pocket?

The specific region of a protein where a ligand binds.

Q66. What is molecular dynamics simulation?

A computational method used to study the movement and interactions of atoms and molecules over time.
Unit IV: RNA-Seq, Systems Biology and Machine Learning
Q67. What is gene expression?

The process by which information in a gene is used to produce a functional RNA or protein.

Q68. What is RNA-Seq?

A sequencing-based method used to study RNA and measure gene expression.

Q69. What is differential gene expression analysis?

It identifies genes whose expression changes significantly between different biological conditions.

Q70. What is fold change?

It is the ratio of expression in one condition to expression in another condition.

Q71. What is log2 fold change?

The base-2 logarithm of the fold change, used to represent expression differences.

Q72. What is an upregulated gene?

A gene showing increased expression in the condition being studied.

Q73. What is a downregulated gene?

A gene showing decreased expression in the condition being studied.

Q74. What is a p-value?

A statistical measure used to assess evidence against a null hypothesis.

Q75. What is FDR?

False Discovery Rate. It controls the expected proportion of false-positive discoveries among identified significant results.

Q76. What is systems biology?

Systems biology studies biological systems as interacting networks of genes, proteins and other components.

Q77. What is a biological network?

A representation of relationships between biological components.

Q78. What are the types of biological networks?
1. Gene regulatory networks.
2. Protein-protein interaction networks.
3. Metabolic networks.
4. Signalling networks.
Q79. What is a gene regulatory network?

A network showing how genes regulate the expression of other genes.

Q80. What is a protein-protein interaction network?

A network representing interactions between different proteins.

Q81. What is a metabolic network?

A network showing the biochemical reactions and metabolic pathways in an organism.

Q82. What is a node?

A node represents an individual component in a network, such as a gene or protein.

Q83. What is an edge?

An edge represents a relationship or interaction between two nodes.

Q84. What is centrality?

A measure of the importance of a node within a network.

Q85. What is machine learning in bioinformatics?

Using machine learning algorithms to analyze biological data and make predictions.

Q86. What is genomic data classification?

Classifying genetic samples into categories based on their genomic features.

Q87. What is Random Forest?

An ensemble learning algorithm that combines multiple decision trees.

Q88. What is SVM?

Support Vector Machine. It classifies data by finding an optimal separating boundary.

Q89. What is feature selection in genomic data?

Selecting important genes or genetic markers for model training.
_____________________________________________________________________________________________________
PART D: IMPORTANT FULL FORMS – IR AND BIF
These are very common viva questions. Learn these properly because examiners often ask direct full forms.
1. Information Retrieval (IR)
Short form	Full form
IR	Information Retrieval
NLP	Natural Language Processing
TF	Term Frequency
IDF	Inverse Document Frequency
TF-IDF	Term Frequency–Inverse Document Frequency
NLTK	Natural Language Toolkit
NLP	Natural Language Processing
HTML	HyperText Markup Language
HTTP	HyperText Transfer Protocol
HTTPS	HyperText Transfer Protocol Secure
URL	Uniform Resource Locator
XML	Extensible Markup Language
API	Application Programming Interface
WWW	World Wide Web
SEO	Search Engine Optimization
SVD	Singular Value Decomposition
LSI	Latent Semantic Indexing
PRP	Probability Ranking Principle
MAP	Mean Average Precision
MRR	Mean Reciprocal Rank
NDCG	Normalized Discounted Cumulative Gain
ROC	Receiver Operating Characteristic
AUC	Area Under the Curve
TP	True Positive
TN	True Negative
FP	False Positive
FN	False Negative
CSS	Cascading Style Sheets
JSON	JavaScript Object Notation
CSV	Comma-Separated Values

2. Bioinformatics (BIF)
Short form	Full form
BIF	Bioinformatics
DNA	Deoxyribonucleic Acid
RNA	Ribonucleic Acid
mRNA	Messenger RNA
tRNA	Transfer RNA
rRNA	Ribosomal RNA
cDNA	Complementary DNA
NCBI	National Center for Biotechnology Information
PDB	Protein Data Bank
BLAST	Basic Local Alignment Search Tool
MSA	Multiple Sequence Alignment
ORF	Open Reading Frame
GC	Guanine-Cytosine
PAM	Point Accepted Mutation
BLOSUM	Blocks Substitution Matrix
FASTA	FASTA sequence format (name derived from FAST-All)
FASTQ	FASTA-like sequence and quality format
SRA	Sequence Read Archive
GEO	Gene Expression Omnibus
ENA	European Nucleotide Archive
DDBJ	DNA Data Bank of Japan
GO	Gene Ontology
KEGG	Kyoto Encyclopedia of Genes and Genomes
RNA-Seq	RNA Sequencing
DGE	Differential Gene Expression
FC	Fold Change
log2FC	Log Base 2 Fold Change
FDR	False Discovery Rate
DESeq2	Differential Expression analysis based on the Negative Binomial distribution (DESeq2 package)
CADD	Computer-Aided Drug Design
SBDD	Structure-Based Drug Design
LBDD	Ligand-Based Drug Design
ADMET	Absorption, Distribution, Metabolism, Excretion and Toxicity
RMSD	Root Mean Square Deviation
SVM	Support Vector Machine
RF	Random Forest
ML	Machine Learning
AI	Artificial Intelligence
PDBQT	Protein Data Bank, Partial Charge (Q) and Atom Type (T) format
Vina	AutoDock Vina molecular docking software
3D	Three-Dimensional
NGS	Next-Generation Sequencing
SNP	Single Nucleotide Polymorphism
PCR	Polymerase Chain Reaction
CDS	Coding DNA Sequence
RPKM	Reads Per Kilobase of transcript per Million mapped reads
FPKM	Fragments Per Kilobase of transcript per Million mapped fragments
TPM	Transcripts Per Million
GO	Gene Ontology
PPI	Protein-Protein Interaction


PART E: IMPORTANT NAME THE FOLLOWING QUESTIONS
These are especially useful for quick viva preparation.
IR – Name the following
Question	Answer
Name the three Boolean operators.	AND, OR, NOT
Name two stemming algorithms.	Porter Stemmer, Snowball Stemmer
Name two Python NLP libraries.	NLTK, spaCy
Name two text vectorization techniques.	CountVectorizer, TF-IDF
Name two classification algorithms.	Naive Bayes, SVM
Name two clustering algorithms.	K-Means, Agglomerative Clustering
Name four types of linkage.	Single, Complete, Average, Ward
Name two types of hierarchical clustering.	Agglomerative, Divisive
Name two web scraping libraries.	BeautifulSoup, Scrapy
Name a Python library for HTTP requests.	Requests
Name a library for data manipulation.	Pandas
Name two search engines.	Google, Bing
Name two retrieval evaluation metrics.	Precision, Recall
Name two ranking models.	Vector Space, Probabilistic
Name the matrix decomposition technique used in LSI.	SVD
Name the three Boolean operations.	Intersection (AND), Union (OR), Difference (NOT)
Name the data structure used for fast document search.	Inverted Index
Name a method for phrase searching.	Positional Index
Name a common text classification dataset.	Spam/Ham email dataset
Name the main components of a search engine.	Crawler, Indexer, Query Processor, Ranker

BIF – Name the following
Question	Answer
Name the four DNA bases.	A, T, G, C
Name the four RNA bases.	A, U, G, C
Name the start codon.	ATG in DNA, AUG in mRNA
Name the three stop codons.	TAA, TAG, TGA in DNA
Name the four protein structural levels.	Primary, Secondary, Tertiary, Quaternary
Name two protein secondary structures.	Alpha helix, Beta sheet
Name two sequence alignment methods.	Global, Local
Name two sequence alignment algorithms.	Needleman-Wunsch, Smith-Waterman
Name two sequence search tools.	BLAST, FASTA
Name two protein scoring matrices.	PAM, BLOSUM
Name two protein databases.	UniProt, PDB
Name two nucleotide databases.	GenBank, ENA
Name three major nucleotide databases.	GenBank, ENA, DDBJ
Name two docking software tools.	AutoDock Vina, AutoDock
Name two cheminformatics tools.	RDKit, Open Babel
Name two protein visualization tools.	PyMOL, UCSF Chimera
Name two ML algorithms for genomic classification.	Random Forest, SVM
Name three differential expression outputs.	Fold Change, p-value, Adjusted p-value
Name three GO categories.	Biological Process, Molecular Function, Cellular Component
Name four biological networks.	Gene Regulatory, PPI, Metabolic, Signalling
Name two RNA-Seq analysis tools.	DESeq2, edgeR
Name three drug properties considered in ADMET.	Absorption, Metabolism, Toxicity (among five)
Name two types of drug design.	Structure-Based, Ligand-Based
Name two DNA sequence file formats.	FASTA, FASTQ


PART F: VERY IMPORTANT DIFFERENCES
These questions are frequently useful in theory papers and viva.
1. IR differences
Topic	Difference
IR vs Data Retrieval	IR retrieves relevant information; data retrieval finds exact data.
Stemming vs Lemmatization	Stemming cuts word endings; lemmatization finds meaningful dictionary forms.
Crawling vs Scraping	Crawling discovers pages; scraping extracts webpage data.
Precision vs Recall	Precision measures correctness of retrieved results; recall measures coverage of relevant results.
Boolean vs Vector Model	Boolean uses exact logical matching; Vector uses numerical similarity and ranking.
TF vs IDF	TF measures word frequency in a document; IDF measures its rarity across documents.
Forward vs Inverted Index	Forward maps documents to terms; inverted maps terms to documents.
Classification vs Clustering	Classification uses labelled data; clustering uses unlabelled data.
Agglomerative vs Divisive	Agglomerative merges clusters; divisive splits clusters.
Single vs Complete Linkage	Single uses minimum distance; complete uses maximum distance.

2. BIF differences
Topic	Difference
DNA vs RNA	DNA contains thymine; RNA contains uracil.
DNA vs Protein	DNA stores genetic information; proteins perform biological functions.
Genotype vs Phenotype	Genotype is genetic makeup; phenotype is observable characteristics.
Gene vs Genome	Gene is a DNA segment; genome is the complete genetic material.
Global vs Local Alignment	Global compares entire sequences; local finds similar regions.
BLAST vs FASTA	Both perform sequence similarity searches using different heuristic approaches.
Homology vs Similarity	Homology indicates common evolutionary origin; similarity is measurable sequence resemblance.
Primary vs Secondary Protein Structure	Primary is amino acid sequence; secondary is local folding.
Docking vs Virtual Screening	Docking predicts binding; screening searches compound libraries for promising candidates.
Upregulation vs Downregulation	Increased gene expression vs decreased gene expression.
Random Forest vs SVM	Multiple decision trees vs optimal separating hyperplane.
Structure-Based vs Ligand-Based Drug Design	Uses target protein structure vs known active ligand information.


PART G: IMPORTANT FORMULAS FOR IR AND BIF
Very important for numerical questions
1. IR formulas
1. Term Frequency
\[
TF(t,d)=\text{Frequency of term }t\text{ in document }d
\]

2. Inverse Document Frequency
\[
IDF(t)=\log\frac{N}{DF(t)}
\]

3. TF-IDF
\[
TFIDF=TF\times IDF
\]

4. Cosine Similarity
\[
Cosine(A,B)=\frac{A\cdot B}{|A||B|}
\]

5. Precision
\[
Precision=\frac{TP}{TP+FP}
\]

6. Recall
\[
Recall=\frac{TP}{TP+FN}
\]

7. F1-score
\[
F1=\frac{2PR}{P+R}
\]

8. Accuracy
\[
Accuracy=\frac{TP+TN}{TP+TN+FP+FN}
\]

2. BIF formulas
1. GC Content
\[
GC\%=\frac{G+C}{A+T+G+C}\times100
\]

2. Fold Change
\[
FC=\frac{Treatment}{Control}
\]

3. Log2 Fold Change
\[
log_2FC=\log_2(FC)
\]

4. Protein Sequence Identity
\[
Identity\%=\frac{\text{Number of identical positions}}{\text{Total aligned positions}}\times100
\]

5. Sequence Similarity

It is commonly expressed as the percentage of aligned positions considered similar according to a scoring matrix. It is not necessarily the same as sequence identity.

6. RMSD
\[
RMSD=\sqrt{\frac{1}{N}\sum_{i=1}^{N}d_i^2}
\]

Where \(d_i\) is the distance between corresponding atoms after structural superposition.

7. Precision
\[
Precision=\frac{TP}{TP+FP}
\]

8. Recall
\[
Recall=\frac{TP}{TP+FN}
\]

9. F1-score
\[
F1=\frac{2PR}{P+R}
\]


PART H: PRACTICAL CODE EXPLANATION – WHAT TO WRITE IN THEORY
This is particularly important because the examiner may show you a line from your Jupyter Notebook and ask what it does.
IR practicals: Important Python functions
Function / Library	Purpose
import nltk	Imports NLTK
word_tokenize()	Splits text into tokens
stopwords.words()	Gets stop words
PorterStemmer()	Creates a stemmer
stem()	Applies stemming
lower()	Converts text to lowercase
TfidfVectorizer()	Converts text into TF-IDF features
CountVectorizer()	Converts text into word counts
train_test_split()	Splits data into training and testing
MultinomialNB()	Creates a Naive Bayes classifier
fit()	Trains the model
predict()	Predicts output
accuracy_score()	Calculates accuracy
confusion_matrix()	Creates confusion matrix
precision_score()	Calculates precision
recall_score()	Calculates recall
f1_score()	Calculates F1-score
AgglomerativeClustering()	Creates hierarchical clusters
dendrogram()	Displays a dendrogram
linkage()	Calculates hierarchical linkage
requests.get()	Sends HTTP GET request
BeautifulSoup()	Parses HTML
find_all()	Finds multiple HTML elements
get()	Retrieves an HTML attribute
DataFrame()	Creates a Pandas table
to_csv()	Saves data in CSV format

BIF practicals: Important Python functions
Function / Library	Purpose
from Bio.Seq import Seq	Imports sequence handling
Seq()	Creates a biological sequence
len()	Calculates sequence length
count()	Counts bases or motifs
complement()	Finds complementary DNA
reverse_complement()	Finds reverse complement
transcribe()	Converts DNA into RNA
translate()	Converts coding sequence into protein
gc_fraction()	Calculates GC fraction
SeqIO.read()	Reads a biological sequence file
SeqIO.parse()	Reads multiple sequences
pandas.read_csv()	Loads a dataset
dropna()	Removes missing values
fillna()	Fills missing values
log2()	Calculates base-2 logarithm
RandomForestClassifier()	Creates Random Forest
SVC()	Creates SVM classifier
StandardScaler()	Standardizes features
classification_report()	Displays classification metrics
confusion_matrix()	Displays actual vs predicted classes
accuracy_score()	Calculates accuracy
PDBParser()	Reads PDB structures
MoleculePreparation	Prepares molecules for docking in supported preparation workflows
Vina()	Initializes AutoDock Vina
set_receptor()	Sets the receptor
set_ligand_from_file()	Loads the ligand
compute_vina_maps()	Computes docking maps
dock()	Performs docking
score()	Calculates docking score

Q90. What is a biomarker?

A measurable biological indicator that can provide information about a biological condition or disease.
________________________________________________________________________________________________________________
Code explaination :

# IR and BIF Practical Viva: Explanation of the `#OR#` Codes

I’ll explain the second/alternative code for each practical in simple viva language. The aim is to help you explain not just what each line does, but also why we use it when the external examiner asks.

Use the bold Viva answer as your ready-to-speak explanation.

# Part A: Information Retrieval (IR)

## IR 1: Text preprocessing using stop-word removal and stemming

Code used: NLTK + regular expressions (`re`).

| Code                                         | Explanation                                                 |
| -------------------------------------------- | ----------------------------------------------------------- |
| `import nltk`                                | Imports the Natural Language Toolkit library.               |
| `import re`                                  | Imports regular expressions for extracting words from text. |
| `nltk.download('stopwords')`                 | Downloads the English stop-word list.                       |
| `stopwords.words('english')`                 | Gets common English words such as is, are, the, and.        |
| `PorterStemmer()`                            | Creates an object to perform stemming.                      |
| `text.lower()`                               | Converts the complete text into lowercase.                  |
| `re.findall(r'\b[a-zA-Z]+\b', text.lower())` | Extracts individual alphabetic words from the text.         |
| `set(stopwords.words('english'))`            | Stores stop words in a set for faster checking.             |
| `filtered_words`                             | Removes words present in the stop-word list.                |
| `stemmer.stem(word)`                         | Reduces words to their root-like form.                      |
| `print()`                                    | Displays the original text and each preprocessing stage.    |

Example:

Input:\
`Information retrieval systems are retrieving useful documents from large collections.`

After stop-word removal:\
`information retrieval systems retrieving useful documents large collections`

After stemming:\
`inform retriev system retriev use document larg collect`

Viva answer:\
This program performs text preprocessing. First, it converts text into lowercase and extracts words. Then it removes common stop words and applies Porter stemming to reduce words to their root forms. This helps reduce unnecessary words and improves information retrieval.

Remember: Stemming does not always produce a grammatically correct English word. It produces a shortened root-like form.

## IR 2: Document retrieval using an inverted index

Code used: `defaultdict` from the `collections` module.

| Code                                  | Explanation                                                                 |
| ------------------------------------- | --------------------------------------------------------------------------- |
| `from collections import defaultdict` | Imports a dictionary that automatically creates an empty list for new keys. |
| `documents = {...}`                   | Stores documents with their IDs and text.                                   |
| `defaultdict(list)`                   | Creates an empty inverted index.                                            |
| `documents.items()`                   | Iterates through each document ID and its text.                             |
| `text.lower().split()`                | Converts text to lowercase and separates it into words.                     |
| `set(words)`                          | Removes duplicate words within the same document.                           |
| `inverted_index[word].append(doc_id)` | Stores the document ID against each word.                                   |
| `inverted_index.items()`              | Iterates through all words and their document IDs.                          |
| `input()`                             | Accepts a search word from the user.                                        |
| `inverted_index[query]`               | Retrieves documents containing the searched word.                           |

Example inverted index:

```
information : [D1]
retrieval   : [D1, D3]
documents   : [D1, D4]
useful      : [D2]
```

Viva answer:\
An inverted index is a data structure that stores each unique word along with the IDs of documents in which it appears. Instead of searching every document individually, we directly search the index. This makes document retrieval faster.

Important difference:

- Normal index: Document → Words
- Inverted index: Word → Documents

## IR 3: Email spam classification using Naive Bayes

Code used: CSV dataset + TF-IDF + Multinomial Naive Bayes.

| Code                                          | Explanation                                                                   |
| --------------------------------------------- | ----------------------------------------------------------------------------- |
| `import pandas as pd`                         | Imports Pandas for handling tabular data.                                     |
| `train_test_split`                            | Divides data into training and testing sets.                                  |
| `TfidfVectorizer`                             | Converts text into numerical TF-IDF features.                                 |
| `MultinomialNB`                               | Imports the Naive Bayes classification algorithm.                             |
| `accuracy_score`                              | Calculates the accuracy of the model.                                         |
| `classification_report`                       | Displays precision, recall and F1-score.                                      |
| `pd.read_csv("spam.csv", encoding="latin-1")` | Reads the email dataset. Latin-1 helps handle special characters in the file. |
| `df[["Category", "Message"]]`                 | Selects only the category and email message columns.                          |
| `X = df["Message"]`                           | Stores email text as input.                                                   |
| `y = df["Category"]`                          | Stores spam/ham labels as output.                                             |
| `test_size=0.2`                               | Uses 20% data for testing and 80% for training.                               |
| `random_state=42`                             | Ensures the same split each time.                                             |
| `fit_transform(X_train)`                      | Learns vocabulary and converts training text into TF-IDF values.              |
| `transform(X_test)`                           | Converts testing text using the same learned vocabulary.                      |
| `model.fit()`                                 | Trains the Naive Bayes model.                                                 |
| `model.predict()`                             | Predicts whether an email is spam or ham.                                     |

Viva answer:\
This program classifies emails as spam or ham using the Multinomial Naive Bayes algorithm. First, it loads an email dataset and separates messages from their labels. TF-IDF converts text into numerical features. The dataset is split into training and testing sets. Finally, the model is trained, tested and evaluated using accuracy and a classification report.

Important viva questions:

- What is TF-IDF? It measures the importance of a word in a document compared with the complete collection of documents.
- What is Naive Bayes? It is a probabilistic classification algorithm based on Bayes' theorem.
- What is ham? A legitimate email.
- Why Multinomial Naive Bayes? It is commonly used for text classification with word-based features.

## IR 4: Agglomerative hierarchical clustering

Your pasted IR 4 section has a mistake: its second `#OR#` code is actually an email spam classification program, not clustering. Do not explain that as IR 4 in the viva.

The correct clustering alternative is the Mall Customers program using `StandardScaler`, `linkage`, `dendrogram` and `AgglomerativeClustering`.

| Code                                                    | Explanation                                                       |
| ------------------------------------------------------- | ----------------------------------------------------------------- |
| `import pandas as pd`                                   | Imports Pandas to read and handle the dataset.                    |
| `StandardScaler`                                        | Standardizes numerical features.                                  |
| `AgglomerativeClustering`                               | Imports the hierarchical clustering algorithm.                    |
| `dendrogram, linkage`                                   | Imports functions to build and display a hierarchical tree.       |
| `pd.read_csv("Mall_Customers.csv")`                     | Loads the customer dataset.                                       |
| `df[["Annual Income (k$)", "Spending Score (1-100)"]]`  | Selects annual income and spending score.                         |
| `StandardScaler()`                                      | Creates the scaling object.                                       |
| `fit_transform(X)`                                      | Scales the selected features.                                     |
| `linkage(X_scaled, method="ward")`                      | Calculates hierarchical cluster combinations using Ward's method. |
| `dendrogram(Z)`                                         | Displays the tree-like structure of clusters.                     |
| `AgglomerativeClustering(n_clusters=5, linkage="ward")` | Creates a model to form five clusters using Ward linkage.         |
| `fit_predict(X_scaled)`                                 | Fits the model and assigns a cluster label to each customer.      |
| `plt.scatter()`                                         | Displays customers in a scatter plot, coloured by cluster.        |

Viva answer:\
Agglomerative clustering is a bottom-up hierarchical clustering algorithm. Initially, every data point is treated as a separate cluster. The algorithm repeatedly merges the closest clusters until the required number of clusters is obtained. A dendrogram visually represents this merging process.

Why do we use StandardScaler?\
To bring features to a similar scale so that a feature with larger numerical values does not dominate the clustering.

What is a dendrogram?\
A tree-like diagram that shows how individual data points or clusters are merged at different distances.

What is Ward linkage?\
It merges clusters in a way that minimizes the increase in within-cluster variance.

## IR 5: Web crawler for e-commerce product information

Code used: Requests + BeautifulSoup + Pandas.

| Code                                          | Explanation                                      |
| --------------------------------------------- | ------------------------------------------------ |
| `import requests`                             | Imports the library used to send HTTP requests.  |
| `import pandas as pd`                         | Imports Pandas to store product information.     |
| `BeautifulSoup`                               | Imports the HTML parser.                         |
| `urljoin`                                     | Converts relative links into complete URLs.      |
| `requests.get(url)`                           | Sends a request to the website.                  |
| `response.text`                               | Gets the HTML content of the webpage.            |
| `BeautifulSoup(response.text, "html.parser")` | Parses HTML into a searchable structure.         |
| `products = []`                               | Creates an empty list to store product details.  |
| `find_all("article", class_="product_pod")`   | Finds all product containers on the page.        |
| `book.h3.a["title"]`                          | Extracts the product title.                      |
| `find("p", class_="price_color").text`        | Extracts the product price.                      |
| `book.h3.a["href"]`                           | Extracts the product's relative link.            |
| `urljoin(url, link)`                          | Converts the relative link into a full URL.      |
| `products.append({...})`                      | Stores title, price and link in a dictionary.    |
| `pd.DataFrame(products)`                      | Converts the product list into a DataFrame.      |
| `df.to_csv("products.csv", index=False)`      | Saves the extracted information into a CSV file. |

Viva answer:\
This program is a basic web crawler. It sends an HTTP request to an e-commerce website, retrieves its HTML content and parses it using BeautifulSoup. It extracts product titles, prices and links, stores them in a list and converts the results into a CSV file.

Important viva questions:

- What is web crawling? Automatically visiting web pages and collecting their information.
- What is BeautifulSoup? A Python library used to parse and extract data from HTML and XML.
- What is Requests? A Python library used to send HTTP requests.
- What is `urljoin()`? It combines a base URL and a relative URL to create a complete URL.
- What is HTML parsing? Converting HTML source code into a structured format that can be searched.

# Part B: Bioinformatics (BIF)

## BIF 1: DNA sequence analysis

Code used: Pandas dataset + motif search + ORF detection.

| Code                                    | Explanation                                          |
| --------------------------------------- | ---------------------------------------------------- |
| `import pandas as pd`                   | Imports Pandas to read the DNA sequence dataset.     |
| `pd.read_csv("human.csv")`              | Loads the CSV file containing DNA sequences.         |
| `df["sequence"][0]`                     | Extracts the first DNA sequence from the dataset.    |
| `sequence.upper()`                      | Converts the sequence into uppercase.                |
| `sequence[:100]`                        | Displays the first 100 characters of the sequence.   |
| `sequence.count("G")`                   | Counts guanine bases.                                |
| `sequence.count("C")`                   | Counts cytosine bases.                               |
| `(gc_count / len(sequence)) * 100`      | Calculates GC content as a percentage.               |
| `motif = "ATG"`                         | Defines the DNA pattern to search for.               |
| `range(len(sequence) - len(motif) + 1)` | Checks all possible starting positions of the motif. |
| `sequence[i:i+len(motif)]`              | Extracts a small section of DNA for comparison.      |
| `positions.append(i)`                   | Stores the position where the motif occurs.          |
| `stop_codons = ["TAA", "TAG", "TGA"]`   | Defines the three stop codons.                       |
| `for frame in range(3)`                 | Checks the three possible forward reading frames.    |
| `sequence[i:i+3]`                       | Extracts one codon containing three nucleotides.     |
| `if codon == "ATG"`                     | Checks for the start codon.                          |
| `if codon in stop_codons`               | Checks whether a stop codon is found.                |
| `orfs.append(sequence[i:j+3])`          | Stores the identified open reading frame.            |
| `len(orf)`                              | Calculates the length of an ORF.                     |

Viva answer:\
This program performs DNA sequence analysis using a CSV dataset. It calculates sequence length and GC content, searches for a specific motif and identifies possible open reading frames. ORF detection checks for a start codon, ATG, followed by an in-frame stop codon such as TAA, TAG or TGA.

Important viva questions:

- What is DNA? Deoxyribonucleic acid, which stores genetic information.
- What are the four DNA bases? Adenine, Thymine, Guanine and Cytosine.
- What is GC content? The percentage of G and C bases in a DNA sequence.
- What is a motif? A short, recurring sequence pattern that may have biological significance.
- What is an ORF? An open reading frame is a sequence that begins with a start codon and ends with an in-frame stop codon.
- What is ATG? The common DNA start codon.

One important point: This code checks only the three forward reading frames. Complete ORF analysis can also examine the reverse strand.

## BIF 2: RNA-Seq analysis and differential gene expression

Code used: NumPy + Pandas + statistical testing + FDR correction.

| Code                                               | Explanation                                                                       |
| -------------------------------------------------- | --------------------------------------------------------------------------------- |
| `import numpy as np`                               | Imports NumPy for numerical calculations.                                         |
| `import pandas as pd`                              | Imports Pandas for tabular data handling.                                         |
| `ttest_ind`                                        | Imports the independent two-sample t-test.                                        |
| `multipletests`                                    | Imports a function for correcting multiple statistical tests.                     |
| `np.random.seed(10)`                               | Ensures reproducible random data.                                                 |
| `genes = ["Gene_" + str(i) ...]`                   | Creates names for 100 genes.                                                      |
| `np.random.poisson(100, (100, 3))`                 | Generates simulated expression counts for 100 genes across three control samples. |
| `treated = ...`                                    | Generates simulated expression counts for treated samples.                        |
| `treated[:10]`                                     | Makes the first 10 genes highly expressed in treated samples.                     |
| `treated[10:20]`                                   | Makes the next 10 genes less expressed in treated samples.                        |
| `np.column_stack((control, treated))`              | Combines control and treated data column-wise.                                    |
| `pd.DataFrame(...)`                                | Organizes the gene expression data into a table.                                  |
| `mean(axis=1)`                                     | Calculates the average expression of each gene across samples.                    |
| `np.log2((treated_mean + 1) / (control_mean + 1))` | Calculates log2 fold change. Adding 1 avoids division by zero.                    |
| `ttest_ind()`                                      | Compares control and treated expression values for each gene.                     |
| `equal_var=False`                                  | Uses Welch's t-test, which does not assume equal variances.                       |
| `p_values.append(p)`                               | Stores the p-value for each gene.                                                 |
| `multipletests(..., method="fdr_bh")`              | Applies Benjamini-Hochberg correction to control the false discovery rate.        |
| `Adjusted_P_Value < 0.05`                          | Selects genes with statistically significant adjusted p-values.                   |
| `Log2_Fold_Change > 1`                             | Identifies genes with more than twofold higher expression in treatment.           |
| `Log2_Fold_Change < -1`                            | Identifies genes with more than twofold lower expression in treatment.            |
| `value_counts()`                                   | Counts upregulated, downregulated and non-significant genes.                      |
| `to_csv()`                                         | Saves the results into a CSV file.                                                |

Viva answer:\
This program performs differential gene expression analysis between control and treated samples. It generates a simulated gene expression dataset, calculates average expression and log2 fold change, and performs a statistical t-test. It then applies Benjamini-Hochberg correction to identify significantly upregulated and downregulated genes.

Important viva questions:

- What is RNA-Seq? A sequencing-based technique used to study gene expression.
- What is gene expression? The process through which information in a gene is used to produce a functional product.
- What is upregulation? An increase in gene expression.
- What is downregulation? A decrease in gene expression.
- What is fold change? The ratio of expression in treated samples to expression in control samples.
- What is log2 fold change? The base-2 logarithm of the fold change.
- What is a p-value? It measures how compatible the observed data are with the null hypothesis.
- Why adjust p-values? Because thousands of genes may be tested, increasing the chance of false positives.

Remember: This is a simplified demonstration using simulated data and t-tests. Real RNA-Seq differential expression analysis generally uses raw read counts with tools such as DESeq2 or edgeR, along with appropriate biological replicates and normalization.

## BIF 3: Molecular docking and virtual screening

Code used: Alternative setup using AutoDock Vina, receptor preparation and ligand preparation.

This practical has several notebook cells. Think of it as a sequence of stages rather than one large Python program.

| Code / command                                 | Explanation                                                                  |
| ---------------------------------------------- | ---------------------------------------------------------------------------- |
| `conda tos accept ...`                         | Accepts the Anaconda channel Terms of Service so packages can be downloaded. |
| `import urllib.request`                        | Imports a module for downloading files from URLs.                            |
| `urllib.request.urlretrieve(url, "vina.exe")`  | Downloads the AutoDock Vina executable and saves it as `vina.exe`.           |
| `os.path.exists("vina.exe")`                   | Checks whether the downloaded executable exists.                             |
| `subprocess.run(["vina.exe", "version"], ...)` | Runs Vina and checks its installed version.                                  |
| `sysconfig.get_path("scripts")`                | Finds the Python environment's Scripts directory.                            |
| `Path(...)`                                    | Handles file and folder paths conveniently.                                  |
| `mk_prepare_receptor.exe`                      | Prepares the protein receptor for molecular docking.                         |
| `--read_pdb`                                   | Specifies the input protein PDB file.                                        |
| `-o receptor`                                  | Specifies the output file name prefix.                                       |
| `-p`                                           | Requests generation of the receptor PDBQT file.                              |
| `-j`                                           | Requests a JSON output file.                                                 |
| `--default_altloc A`                           | Selects alternate atom location A when applicable.                           |
| `os.path.exists("receptor.pdbqt")`             | Checks whether receptor preparation succeeded.                               |
| `mk_prepare_ligand.exe`                        | Prepares the ligand molecule for docking.                                    |
| `-i`                                           | Specifies the input ligand file.                                             |
| `-o ligand.pdbqt`                              | Specifies the output ligand file.                                            |
| `PDBParser()`                                  | Reads the protein's PDB structure.                                           |
| `cdist(A, E)`                                  | Calculates distances between two sets of atomic coordinates.                 |
| `center = ...mean(axis=0)`                     | Calculates the approximate centre of the docking region.                     |
| `vina.exe`                                     | Executes the molecular docking program.                                      |
| `--receptor`                                   | Specifies the prepared receptor file.                                        |
| `--ligand`                                     | Specifies the prepared ligand file.                                          |
| `--center_x`, `--center_y`, `--center_z`       | Specify the centre of the docking search box.                                |
| `--size_x`, `--size_y`, `--size_z`             | Specify the dimensions of the search box.                                    |
| `--exhaustiveness 8`                           | Controls the search effort used by Vina.                                     |
| `--num_modes 5`                                | Requests up to five possible binding poses.                                  |
| `--out`                                        | Specifies the output docking file.                                           |
| `"VINA RESULT"`                                | Identifies the lines containing docking affinity results.                    |
| `glob.glob("ligands/*.sdf")`                   | Finds all SDF ligand files in the ligands folder.                            |
| `Path(file).stem`                              | Extracts the ligand filename without its extension.                          |
| `results.append([name, score])`                | Stores the ligand name and its binding score.                                |
| `pd.DataFrame(results, ...)`                   | Creates a table of screening results.                                        |
| `sort_values("Binding Affinity")`              | Sorts ligands by binding affinity.                                           |

Viva answer:\
This practical performs molecular docking and virtual screening using AutoDock Vina. First, the protein receptor and ligand molecules are prepared in PDBQT format. Then, the binding site is selected and a docking grid box is defined. Vina predicts possible ligand binding poses and calculates binding affinity scores. Virtual screening repeats this process for multiple ligands to identify promising candidates.

Important viva questions:

- What is molecular docking? A computational technique that predicts how a ligand binds to a protein.
- What is a receptor? The target protein with which a ligand interacts.
- What is a ligand? A small molecule that binds to a target protein.
- What is PDB? Protein Data Bank format, used to store three-dimensional molecular structures.
- What is PDBQT? A PDB-derived format used by AutoDock tools that includes atom types and partial charges.
- What is AutoDock Vina? Software used for molecular docking and binding affinity estimation.
- What is binding affinity? A measure of how strongly a ligand is predicted to bind to a target.
- What does a negative binding score mean? It represents favourable predicted binding according to Vina's scoring function. A more negative score is generally considered more favourable.
- What is virtual screening? Computationally testing multiple molecules against a target protein to shortlist potential candidates.
- What is a docking grid box? The three-dimensional search region where Vina looks for ligand binding poses.



Important correction

&#x20;Your pasted alternative code contains a few typing or formatting mistakes, such as `--center y` instead of `--center_y`, and `-size x` instead of `--size_x`. The correct Vina options use underscores and two hyphens, for example `--center_y` and `--size_x`. Also, the ligand preparation command should use the correct `-i` input option.



How to explain the result:\
For example, if the best Vina score is `-5.734 kcal/mol`, say: The program predicted a binding affinity of -5.734 kcal/mol for the best docking pose. A more negative score generally indicates more favourable predicted binding, but it does not prove actual biological activity.

## BIF 4: Machine learning for genomic classification

Code used: DNA sequence classification using Random Forest and 3-mer features.

| Code                                     | Explanation                                                                      |
| ---------------------------------------- | -------------------------------------------------------------------------------- |
| `import pandas as pd`                    | Imports Pandas to read the DNA dataset.                                          |
| `CountVectorizer`                        | Converts DNA sequences into numerical features.                                  |
| `train_test_split`                       | Divides the dataset into training and testing sets.                              |
| `RandomForestClassifier`                 | Imports the Random Forest classification algorithm.                              |
| `accuracy_score`                         | Calculates prediction accuracy.                                                  |
| `pd.read_csv("human.csv")`               | Loads the DNA sequence dataset.                                                  |
| `df["sequence"].astype(str).str.upper()` | Converts all sequences into uppercase strings.                                   |
| `y = df["class"]`                        | Stores the class labels to be predicted.                                         |
| `analyzer="char"`                        | Tells the vectorizer to process individual characters and character patterns.    |
| `ngram_range=(3, 3)`                     | Extracts groups of three consecutive characters called 3-mers.                   |
| `fit_transform(X)`                       | Learns the 3-mer vocabulary and converts sequences into numerical features.      |
| `test_size=0.2`                          | Uses 80% data for training and 20% for testing.                                  |
| `random_state=42`                        | Makes the data split reproducible.                                               |
| `stratify=y`                             | Maintains approximately the same class proportions in training and testing data. |
| `n_estimators=150`                       | Creates 150 decision trees in the Random Forest.                                 |
| `class_weight="balanced"`                | Gives additional weight to underrepresented classes.                             |
| `model.fit()`                            | Trains the Random Forest model.                                                  |
| `model.predict()`                        | Predicts the class of test sequences.                                            |
| `accuracy_score()`                       | Calculates the percentage of correct predictions.                                |
| `classification_report()`                | Displays precision, recall, F1-score and support for each class.                 |

Viva answer:\
This program performs genomic sequence classification using a Random Forest algorithm. First, it loads DNA sequences and their class labels. It extracts 3-mer features, which are groups of three consecutive DNA characters. These features are converted into numerical values and supplied to the Random Forest classifier. Finally, the model predicts the sequence classes and evaluates its performance using accuracy and a classification report.

Important viva questions:

- What is genomic classification? Categorizing DNA or genomic data into predefined classes using computational methods.
- What is Random Forest? An ensemble learning algorithm that combines multiple decision trees to make predictions.
- What is a 3-mer? A sequence of three consecutive nucleotides.
- Why use 3-mers? They capture short DNA sequence patterns that may help distinguish genomic classes.
- What is a feature? A measurable input characteristic used by a machine learning model.
- What is a class label? The category that the model is trained to predict.
- What is overfitting? When a model learns training data too closely and performs poorly on unseen data.
- Why use multiple decision trees? Combining predictions from many trees generally improves stability and reduces overfitting compared with relying on one tree.

# Last-minute revision: Remember these functions

These are the functions an external examiner may point to and ask you to explain.

| Practical | Most important functions                                                                  |
| --------- | ----------------------------------------------------------------------------------------- |
| IR 1      | `re.findall()`, `stopwords.words()`, `PorterStemmer()`, `stem()`                          |
| IR 2      | `defaultdict()`, `split()`, `set()`, `append()`                                           |
| IR 3      | `read_csv()`, `TfidfVectorizer()`, `fit_transform()`, `MultinomialNB()`, `predict()`      |
| IR 4      | `StandardScaler()`, `linkage()`, `dendrogram()`, `fit_predict()`                          |
| IR 5      | `requests.get()`, `BeautifulSoup()`, `find_all()`, `urljoin()`, `to_csv()`                |
| BIF 1     | `count()`, `find()`, `len()`, `append()`, `upper()`                                       |
| BIF 2     | `mean()`, `np.log2()`, `ttest_ind()`, `multipletests()`                                   |
| BIF 3     | `PDBParser()`, `subprocess.run()`, `cdist()`, `glob.glob()`                               |
| BIF 4     | `CountVectorizer()`, `RandomForestClassifier()`, `fit()`, `predict()`, `accuracy_score()` |

# 9 practicals: One-line answers for the external

Learn these nine answers. They are useful when the examiner simply asks, What did you perform?

| Practical | Answer                                                                                                                     |
| --------- | -------------------------------------------------------------------------------------------------------------------------- |
| IR 1      | I performed text preprocessing using stop-word removal and Porter stemming.                                                |
| IR 2      | I created an inverted index to map words to their corresponding document IDs and retrieve documents.                       |
| IR 3      | I classified emails as spam or ham using the Naive Bayes algorithm.                                                        |
| IR 4      | I grouped similar data points using Agglomerative Hierarchical Clustering and visualized the hierarchy using a dendrogram. |
| IR 5      | I extracted product names, prices and links from a website using Requests and BeautifulSoup.                               |
| BIF 1     | I analyzed DNA sequences by calculating GC content, finding motifs and identifying ORFs.                                   |
| BIF 2     | I compared gene expression between control and treated samples to identify upregulated and downregulated genes.            |
| BIF 3     | I performed molecular docking and virtual screening using AutoDock Vina to estimate ligand binding affinity.               |
| BIF 4     | I classified genomic DNA sequences using 3-mer features and a Random Forest classifier.                                    |

Three corrections to remember before your practical exam:

1. IR 4: Your second alternative code is a spam classification program, not clustering. Use the Mall Customers clustering code for IR 4.
2. BIF 2: Your second code uses simulated expression data. Explain it as a demonstration of differential expression analysis, not actual experimental RNA-Seq results.
3. BIF 3: Your alternative notebook has some copied command errors. Remember the correct Vina argument names and explain the actual docking workflow.

My suggestion: Don't memorize every Python statement word for word. For each practical, remember the purpose, input, main library, important functions, algorithm and output. If you can explain those six things confidently, you can handle most external viva questions.
---------------------------------------------------------------
Here are the advantages, disadvantages and applications of all 9 IR and Bioinformatics practical topics, in simple SPPU viva-ready points. You can use these directly in your external practical exam.

# Information Retrieval (IR)

## 1. Text Preprocessing

Advantages

1. Removes unnecessary words and symbols.
2. Reduces text size and complexity.
3. Improves the efficiency of text analysis.
4. Helps improve search and classification results.

Disadvantages

1. May remove words that are important in context.
2. Stemming can produce incorrect root words.
3. Language-specific preprocessing is required.

Applications

1. Search engines.
2. Spam detection.
3. Sentiment analysis.
4. Text classification and document retrieval.

## 2. Inverted Indexing

Advantages

1. Provides faster document searching.
2. Reduces search time.
3. Efficiently stores word-to-document mappings.
4. Useful for searching large collections of documents.

Disadvantages

1. Requires extra storage.
2. Index creation takes time.
3. Index must be updated when documents change.
4. May require additional processing for complex queries.

Applications

1. Google and other search engines.
2. Digital libraries.
3. Document management systems.
4. E-commerce product searches.

## 3. Email Spam Classification

Advantages

1. Automatically identifies spam emails.
2. Saves users' time.
3. Reduces unwanted and potentially harmful emails.
4. Machine learning models can learn from labelled examples.

Disadvantages

1. May classify genuine emails as spam.
2. New spam patterns can be difficult to detect.
3. Requires labelled training data.
4. Accuracy depends on data quality.

Applications

1. Gmail spam filters.
2. Email security systems.
3. Phishing detection.
4. Business email management.

## 4. Agglomerative Hierarchical Clustering

Advantages

1. Does not require labelled data.
2. Shows hierarchical relationships between data points.
3. Dendrogram helps visualize clusters.
4. Useful for discovering hidden patterns.

Disadvantages

1. Computationally expensive for large datasets.
2. Sensitive to noise and outliers.
3. Once clusters are merged, they cannot be separated.
4. Results depend on the distance and linkage method.

Applications

1. Customer segmentation.
2. Document grouping.
3. Gene expression analysis.
4. Image segmentation.

## 5. Web Crawler

Advantages

1. Automatically collects information from websites.
2. Saves time compared to manual data collection.
3. Can collect product names, prices and links.
4. Helps gather large amounts of web data.

Disadvantages

1. Website structure changes can break the crawler.
2. Some websites block automated requests.
3. Large-scale crawling requires time and resources.
4. Must follow website terms, robots.txt and applicable data rules.

Applications

1. Search engines.
2. E-commerce price comparison.
3. Product information collection.
4. Website monitoring and research.

# Bioinformatics (BIF)

## 6. DNA Sequence Analysis

Advantages

1. Helps identify important DNA patterns.
2. Calculates GC content.
3. Identifies motifs and open reading frames.
4. Helps understand genetic information.

Disadvantages

1. Large DNA sequences require more processing.
2. Sequence errors can affect results.
3. Simple methods may miss complex biological patterns.
4. Identifying an ORF does not confirm that it is a functional gene.

Applications

1. Gene identification.
2. Genome analysis.
3. Disease research.
4. Genetic engineering and biotechnology.

## 7. RNA-Seq Differential Gene Expression

Advantages

1. Identifies genes with different expression levels.
2. Helps compare treated and control samples.
3. Can identify upregulated and downregulated genes.
4. Helps understand biological responses.

Disadvantages

1. Requires high-quality RNA and sequencing data.
2. Data analysis can be computationally expensive.
3. Results can be affected by biological variation.
4. Requires statistical analysis and proper normalization.

Applications

1. Cancer research.
2. Drug discovery.
3. Disease diagnosis research.
4. Studying gene expression under different conditions.

## 8. Molecular Docking and Virtual Screening

Advantages

1. Predicts how a ligand binds to a protein.
2. Helps identify potential drug candidates.
3. Saves time and cost in early drug discovery.
4. Allows screening of multiple molecules.

Disadvantages

1. Docking scores are predictions, not experimental proof.
2. Accuracy depends on protein structure and preparation.
3. May not fully represent real biological conditions.
4. Requires specialized software and computational resources.

Applications

1. Drug discovery.
2. Protein-ligand interaction studies.
3. Drug repurposing.
4. Virtual screening of chemical compounds.

## 9. Genomic Classification Using Machine Learning

Advantages

1. Automatically classifies genomic sequences.
2. Can identify patterns in DNA data.
3. Handles large datasets.
4. Random Forest can reduce overfitting compared with a single decision tree.

Disadvantages

1. Requires sufficient and good-quality training data.
2. Model performance depends on selected features.
3. Training can be computationally expensive for large datasets.
4. May not generalize well to unseen genomic data.

Applications

1. Disease prediction research.
2. Gene and DNA sequence classification.
3. Cancer genomics.
4. Identification of biological patterns.

# Quick revision table

| Practical                     | Main advantage                       | Main disadvantage                        |
| ----------------------------- | ------------------------------------ | ---------------------------------------- |
| IR 1: Text preprocessing      | Reduces text complexity              | May remove useful words                  |
| IR 2: Inverted index          | Faster searching                     | Requires extra storage                   |
| IR 3: Spam classification     | Automatic spam detection             | False classification                     |
| IR 4: Clustering              | Finds hidden groups                  | Sensitive to outliers                    |
| IR 5: Web crawler             | Automatic data collection            | Website changes can break it             |
| BIF 1: DNA analysis           | Identifies DNA patterns              | Simple methods have limitations          |
| BIF 2: RNA-Seq                | Finds differentially expressed genes | Requires quality data and statistics     |
| BIF 3: Molecular docking      | Helps identify drug candidates       | Predictions need experimental validation |
| BIF 4: Genomic classification | Classifies DNA patterns              | Depends on training data                 |

Viva tip: If the external asks for only one advantage, disadvantage and application, remember the first point under each heading. Those are the easiest and most relevant answers for your practical topics.