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
