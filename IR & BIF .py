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
