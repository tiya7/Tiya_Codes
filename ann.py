
# 1
import numpy as np
import matplotlib.pyplot as plt

# x values
x = np.linspace(-10, 10, 100)

# Activation functions
linear = x
step = np.where(x >= 0, 1, 0)
sigmoid = 1 / (1 + np.exp(-x))
relu = np.maximum(0, x)
tanh = np.tanh(x)

# Create 5 subplots (3 rows, 2 columns)
plt.figure(figsize=(10, 8))

plt.subplot(3, 2, 1)
plt.plot(x, linear)
plt.title("Linear")
plt.grid()

plt.subplot(3, 2, 2)
plt.plot(x, step)
plt.title("Step")
plt.grid()

plt.subplot(3, 2, 3)
plt.plot(x, sigmoid)
plt.title("Sigmoid")
plt.grid()

plt.subplot(3, 2, 4)
plt.plot(x, relu)
plt.title("ReLU")
plt.grid()

plt.subplot(3, 2, 5)
plt.plot(x, tanh)
plt.title("Tanh")
plt.grid()

plt.tight_layout()
plt.show()

---------------------------------------------------------------------------------------------------------------
# 1
import numpy as np
import matplotlib.pyplot as plt

# Take input from user
start = float(input("Enter start value: "))
end = float(input("Enter end value: "))
points = int(input("Enter number of points: "))

# Create input array
x = np.linspace(start, end, points)

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x):
    return np.maximum(0.01 * x, x)

# Plot graphs
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.plot(x, sigmoid(x))
plt.title("Sigmoid")

plt.subplot(2, 2, 2)
plt.plot(x, tanh(x))
plt.title("Tanh")

plt.subplot(2, 2, 3)
plt.plot(x, relu(x))
plt.title("ReLU")

plt.subplot(2, 2, 4)
plt.plot(x, leaky_relu(x))
plt.title("Leaky ReLU")

plt.tight_layout()
plt.show()

-----------------------------------------------------------------------------------------------------------------

# 2

# McCulloch-Pitts ANDNOT implementation

# weights
w1 = 1      # for x1
w2 = -1     # for x2 (NOT effect)
threshold = 1

# inputs
inputs = [(0,0), (0,1), (1,0), (1,1)]

print("x1 x2 Output")

for x1, x2 in inputs:
    # weighted sum
    net = x1*w1 + x2*w2
    
    # activation (step function)
    if net >= threshold:
        output = 1
    else:
        output = 0
    
    print(x1, x2, " ", output)

--------------------------------------------------------------------------------
# 2
# McCulloch-Pitts neuron for ANDNOT

def mp_neuron(A, B):
    w1 = 1
    w2 = -1
    threshold = 1

    net = A*w1 + B*w2
    return 1 if net >= threshold else 0


# Take input
A = int(input("Enter value of A (0 or 1): "))
B = int(input("Enter value of B (0 or 1): "))

# Validate
if (A not in [0,1]) or (B not in [0,1]):
    print("Please enter only 0 or 1")
else:
    print("A B Output")
    print(A, B, mp_neuron(A, B))

# McCulloch-Pitts neuron for ANDNOT

def mp_neuron(A, B):
    w1 = 1     # weight for A
    w2 = -1    # weight for B
    threshold = 1

    net = A*w1 + B*w2

    if net >= threshold:
        return 1
    else:
        return 0


# Test all combinations
inputs = [(0,0), (0,1), (1,0), (1,1)]

print("A B Output")
for A, B in inputs:
    print(A, B, mp_neuron(A, B))

-----------------------------------------------------------------------------------------

# 3
# input: last bit only (even=0, odd=1)
X = [0,1,0,1,0,1,0,1,0,1]
y = [0,1,0,1,0,1,0,1,0,1]

w = 0
b = 0
lr = 0.1

def activation(x):
    return 1 if x >= 0 else 0

# training
for epoch in range(10):
    for i in range(len(X)):
        net = w * X[i] + b
        output = activation(net)
        error = y[i] - output
        
        w += lr * error * X[i]
        b += lr * error

# user input
num = int(input("Enter number: "))
x = num % 2   # key idea

result = activation(w * x + b)

if result == 0:
    print("Even")
else:
    print("Odd")

-------------------------------------------------------------------------------------------
# 3
# Perceptron-style Even/Odd recognition for digits 0–9

def perceptron(x):
    w = 1
    b = 0

    net = x * w + b

    # Even/Odd check
    if net % 2 == 0:
        return 0   # Even
    else:
        return 1   # Odd


print("Number  Type")
for num in range(10):
    result = perceptron(num)
    if result == 0:
        print(num, "     EVEN")
    else:
        print(num, "     ODD")

""OR""
# Step function
def step_function(net):
    return 1 if net >= 0 else 0


# Perceptron-style function
def perceptron(num):
    # target using correct logic
    target = 1 if num % 2 != 0 else -1   # ODD = 1, EVEN = -1

    # simple weights (demo)
    w = 1
    net = w * target

    output = step_function(net)

    return "ODD" if output == 1 else "EVEN"


# Menu
print("Choose an option:")
print("1. Show all numbers (0–9)")
print("2. Check a single number")

choice = int(input("Enter your choice (1 or 2): "))


# Option 1: Show all
if choice == 1:
    print("\nNumber  Type")
    for num in range(10):
        print(num, "     ", perceptron(num))


# Option 2: Single input
elif choice == 2:
    num = int(input("Enter a number (0–9): "))

    if num < 0 or num > 9:
        print("Please enter between 0 and 9")
    else:
        print("\nNumber  Type")
        print(num, "     ", perceptron(num))


else:
    print("Invalid choice")
------------------------------------------------------------------------------------

# 4
import matplotlib.pyplot as plt

# data (AND gate)
x1 = [0, 0, 1, 1]
x2 = [0, 1, 0, 1]
target = [0, 0, 0, 1]

# weights and bias
w1 = 0
w2 = 0
b = 0
lr = 0.1

# training
for _ in range(10):
    for i in range(4):
        net = w1*x1[i] + w2*x2[i] + b
        output = 1 if net >= 0 else 0
        
        error = target[i] - output
        
        w1 += lr * error * x1[i]
        w2 += lr * error * x2[i]
        b += lr * error

# plot points
for i in range(4):
    if target[i] == 0:
        plt.scatter(x1[i], x2[i])
    else:
        plt.scatter(x1[i], x2[i], marker='x')

# decision boundary: w1*x + w2*y + b = 0
x = [-1, 2]
y = [-(w1*x[0] + b)/w2, -(w1*x[1] + b)/w2]

plt.plot(x, y)

plt.title("Perceptron (AND)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid()
plt.show()

-------------------------------------------------------------------------------------------
#4
# Step function
def step_function(net):
    return 1 if net >= 0 else 0


# Perceptron-style function
def perceptron(num):
    # target using correct logic
    target = 1 if num % 2 != 0 else -1   # ODD = 1, EVEN = -1

    # simple weights (demo)
    w = 1
    net = w * target

    output = step_function(net)

    return "ODD" if output == 1 else "EVEN"


# Menu
print("Choose an option:")
print("1. Show all numbers (0–9)")
print("2. Check a single number")

choice = int(input("Enter your choice (1 or 2): "))


# Option 1: Show all
if choice == 1:
    print("\nNumber  Type")
    for num in range(10):
        print(num, "     ", perceptron(num))


# Option 2: Single input
elif choice == 2:
    num = int(input("Enter a number (0–9): "))

    if num < 0 or num > 9:
        print("Please enter between 0 and 9")
    else:
        print("\nNumber  Type")
        print(num, "     ", perceptron(num))


else:
    print("Invalid choice")

-----------------------------------------------------------------------------
#5 
import numpy as np

# -------- Step 1: Define pattern pairs --------
X1 = np.array([1, -1, 1])
Y1 = np.array([1, 1])

X2 = np.array([-1, 1, -1])
Y2 = np.array([-1, -1])

# -------- Step 2: Weight Matrix --------
W = np.outer(X1, Y1) + np.outer(X2, Y2)

print("Weight Matrix (W):\n")

# Print column headers
print("      y1   y2")

# Print rows with labels
for i in range(len(W)):
    print(f"x{i+1}   {W[i][0]:>3}  {W[i][1]:>3}")

# -------- Step 3: Activation function --------
def sign(x):
    return np.where(x >= 0, 1, -1)

# -------- Step 4: Recall --------
X_test = X1
Y_recalled = sign(np.dot(X_test, W))

print("\nInput Pattern X:", X_test)
print("Recalled Pattern Y:", Y_recalled)

Y_test = Y1
X_recalled = sign(np.dot(Y_test, W.T))

print("\nInput Pattern Y:", Y_test)
print("Recalled Pattern X:", X_recalled)

--------------------------------------------------------------------------------
# 5
import numpy as np

# Sign activation function
def sign(x):
    return np.where(x >= 0, 1, -1)


# ---- Number of pairs ----
n = int(input("Enter number of vector pairs: "))

X_list = []
Y_list = []

# ---- Take inputs ----
for i in range(n):
    print(f"\nPair {i+1}:")
    X = list(map(int, input("Enter X (use 1 and -1): ").split()))
    Y = list(map(int, input("Enter Y (use 1 and -1): ").split()))

    X_list.append(np.array(X))
    Y_list.append(np.array(Y))


# ---- Weight matrix ----
W = np.zeros((len(X_list[0]), len(Y_list[0])))

for i in range(n):
    W += np.outer(X_list[i], Y_list[i])

print("\nWeight Matrix:\n", W)


# ---- Recall functions ----
def recall_Y(X):
    return sign(np.dot(X, W))

def recall_X(Y):
    return sign(np.dot(Y, W.T))


# ---- Testing all pairs ----
print("\nForward Recall (X → Y):")
for i in range(n):
    print(f"X{i+1} ->", recall_Y(X_list[i]))

print("\nBackward Recall (Y → X):")
for i in range(n):
    print(f"Y{i+1} ->", recall_X(Y_list[i]))

--------------------------------------------------------------------------
# 6
import numpy as np

# -------- Step 1: Input and Output --------
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

y = np.array([[0],
              [1],
              [1],
              [0]])   # XOR problem

# -------- Step 2: Initialize weights --------
np.random.seed(1)
w1 = np.random.rand(2, 2)   # input -> hidden
w2 = np.random.rand(2, 1)   # hidden -> output

# -------- Step 3: Activation function --------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# -------- Step 4: Training --------
for epoch in range(5000):

    # ---- Forward Propagation ----
    hidden = sigmoid(np.dot(X, w1))
    output = sigmoid(np.dot(hidden, w2))

    # ---- Error ----
    error = y - output

    # ---- Backpropagation ----
    d_output = error * sigmoid_derivative(output)
    d_hidden = np.dot(d_output, w2.T) * sigmoid_derivative(hidden)

    # ---- Update weights ----
    w2 += np.dot(hidden.T, d_output)
    w1 += np.dot(X.T, d_hidden)

# -------- Step 5: Output --------
print("Final Output:\n", output)

-----------------------------------------------------------------------------------------------
#6
import numpy as np

# sigmoid
def sig(x):
    return 1/(1+np.exp(-x))

# derivative
def dsig(x):
    return x*(1-x)


# training data (XOR)
X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([[0],[1],[1],[0]])

# weights
np.random.seed(1)
W1 = np.random.rand(2,2)
W2 = np.random.rand(2,1)

# training
for i in range(3000):
    h = sig(np.dot(X,W1))
    o = sig(np.dot(h,W2))

    e = Y - o

    d2 = e * dsig(o)
    d1 = d2.dot(W2.T) * dsig(h)

    W2 += h.T.dot(d2)
    W1 += X.T.dot(d1)


# input loop
while True:
    s = input("enter two binary inputs or q to quit: ")

    if s == 'q':
        break

    a,b = map(int, s.split())

    x = np.array([[a,b]])

    h = sig(np.dot(x,W1))
    o = sig(np.dot(h,W2))

    print("output:", 1 if o[0][0] > 0.5 else 0)

-----------------------------------------------------------------------------------------------------

# 7
import numpy as np

# -------- Step 1: Input and Output --------
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# -------- Step 2: Initialize weights --------
np.random.seed(1)
w1 = np.random.rand(2, 2)   # input -> hidden
w2 = np.random.rand(2, 1)   # hidden -> output

# -------- Step 3: Activation function --------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivative(x):
    return x * (1 - x)

# -------- Step 4: Training --------
for epoch in range(5000):

    # Forward Propagation
    hidden = sigmoid(np.dot(X, w1))
    output = sigmoid(np.dot(hidden, w2))

    # Error
    error = y - output

    # Backpropagation
    d_output = error * derivative(output)
    d_hidden = np.dot(d_output, w2.T) * derivative(hidden)

    # Update weights
    w2 += np.dot(hidden.T, d_output)
    w1 += np.dot(X.T, d_hidden)

# -------- Step 5: Result --------
print("Final Output:")
print(output)

---------------------------------------------------------------------------------------------
#7
import numpy as np

# Activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Derivative
def dsigmoid(x):
    return x*(1-x)


# ---- XOR Training Data ----
X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([[0],[1],[1],[0]])

# ---- Initialize weights ----
np.random.seed(1)
W1 = np.random.rand(2,2)
W2 = np.random.rand(2,1)

# ---- Training ----
for i in range(3000):
    # Forward
    hidden = sigmoid(np.dot(X, W1))
    output = sigmoid(np.dot(hidden, W2))

    # Error
    error = Y - output

    # Backpropagation
    d2 = error * dsigmoid(output)
    d1 = d2.dot(W2.T) * dsigmoid(hidden)

    # Update weights
    W2 += hidden.T.dot(d2)
    W1 += X.T.dot(d1)


# ---- User Input ----
while True:
    s = input("Enter two binary inputs or q to quit: ")

    if s.lower() == 'q':
        break

    try:
        a, b = map(int, s.split())

        if a not in [0,1] or b not in [0,1]:
            print("Enter only 0 or 1")
            continue

        inp = np.array([[a, b]])

        # Forward pass
        hidden = sigmoid(np.dot(inp, W1))
        out = sigmoid(np.dot(hidden, W2))

        result = 1 if out[0][0] > 0.5 else 0

        print("Output:", result)

    except:
        print("Invalid input. Enter like: 1 0")

-------------------------------------------------------------------------------------------------

# 8
import numpy as np

# -------- Step 1: Sample Input and Output --------
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

y = np.array([[0],
              [1],
              [1],
              [0]])   # sample dataset

# -------- Step 2: Initialize weights and bias --------
np.random.seed(0)
w1 = np.random.rand(2, 3)   # input -> hidden (2x3)
b1 = np.zeros((1,3))

w2 = np.random.rand(3, 1)   # hidden -> output (3x1)
b2 = np.zeros((1,1))

# -------- Step 3: Activation function --------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivative(x):
    return x * (1 - x)

# -------- Step 4: Training --------
for epoch in range(5000):

    # ---- Forward Propagation ----
    hidden_input = np.dot(X, w1) + b1
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, w2) + b2
    final_output = sigmoid(final_input)

    # ---- Error ----
    error = y - final_output

    # ---- Backpropagation ----
    d_output = error * derivative(final_output)
    d_hidden = np.dot(d_output, w2.T) * derivative(hidden_output)

    # ---- Update weights and bias ----
    w2 += np.dot(hidden_output.T, d_output)
    b2 += np.sum(d_output, axis=0, keepdims=True)

    w1 += np.dot(X.T, d_hidden)
    b1 += np.sum(d_hidden, axis=0, keepdims=True)

# -------- Step 5: Output --------
print("Final Output:\n", final_output)

------------------------------------------------------------------------------------------------
#8
import numpy as np

# activation
def sig(x):
    return 1/(1+np.exp(-x))

def dsig(x):
    return x*(1-x)


# inputs
n = int(input("number of training samples: "))
inp = int(input("number of input neurons: "))
hid = int(input("number of hidden neurons: "))
out = int(input("number of output neurons: "))

print("enter input values:")
X = np.array([list(map(float, input().split())) for _ in range(n)])

print("enter output values:")
Y = np.array([list(map(float, input().split())) for _ in range(n)])

epochs = int(input("enter epochs: "))


# weights
W1 = np.random.rand(inp, hid)
W2 = np.random.rand(hid, out)


# training
for _ in range(epochs):
    h = sig(X @ W1)
    o = sig(h @ W2)

    e = Y - o

    W2 += h.T @ (e * dsig(o))
    W1 += X.T @ ((e * dsig(o)) @ W2.T * dsig(h))


print("\nfinal output:")
print(o)


# testing
test = np.array([list(map(float, input("enter test input: ").split()))])

h = sig(test @ W1)
o = sig(h @ W2)

print("predicted output:", o)

---------------------------------------------------------------------------------------------------------

# 9
import numpy as np

# -------- Step 1: Define 4 patterns --------
P1 = np.array([1, -1, 1, -1])
P2 = np.array([-1, 1, -1, 1])
P3 = np.array([1, 1, -1, -1])
P4 = np.array([-1, -1, 1, 1])

patterns = [P1, P2, P3, P4]

# -------- Step 2: Create Weight Matrix --------
W = np.zeros((4, 4))

for p in patterns:
    W += np.outer(p, p)

# Remove self-connections
np.fill_diagonal(W, 0)

print("Weight Matrix (W):\n", W)

# -------- Step 3: Activation function --------
def sign(x):
    return np.where(x >= 0, 1, -1)

# -------- Step 4: Test Recall --------
test = np.array([1, -1, 1, -1])   # try noisy input also

print("\nInput Pattern:", test)

# Update once (synchronous)
output = sign(np.dot(W, test))

print("Recalled Pattern:", output)

-------------------------------------------------------------------------------------------------
#9
import numpy as np

# sign function
def sign(x):
    return np.where(x >= 0, 1, -1)


# ---- Take input ----
n = int(input("Enter size of each vector: "))

patterns = []
print("Enter 4 vectors (use 1 and -1):")

for i in range(4):
    p = list(map(int, input(f"Vector {i+1}: ").split()))
    patterns.append(np.array(p))

# ---- Weight matrix ----
W = np.zeros((n, n))

for p in patterns:
    W += np.outer(p, p)

# remove self-connection
np.fill_diagonal(W, 0)

print("\nWeight Matrix:\n", W)


# ---- Recall ----
test = list(map(int, input("\nEnter test vector: ").split()))
test = np.array(test)

result = sign(np.dot(W, test))

print("Recalled Pattern:", result)

-----------------------------------------------------------------------------------------------

# 10
import cv2

# Load pre-trained face detector (simple CNN-based model)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Read image
img = cv2.imread(r"C:\Users\mayur\OneDrive\Pictures\image.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect objects (faces)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# Draw boxes
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

# Show result
cv2.imshow('Output', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

-----------------------------------------------------------------------------------------------------------------------


# 11
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# -------------------------------
# 1. Neural Network (XOR Problem)
# -------------------------------
print("Neural Network (XOR)")

X1 = np.array([[0,0],[0,1],[1,0],[1,1]])
y1 = np.array([0,1,1,0])

model1 = Sequential([
    Dense(4, input_dim=2, activation='relu'),
    Dense(1, activation='sigmoid')
])

model1.compile(optimizer='adam',
               loss='binary_crossentropy',
               metrics=['accuracy'])

model1.fit(X1, y1, epochs=1000, verbose=0)

print("Predictions:")
print(model1.predict(X1))


# -------------------------------
# 2. Logistic Regression
# -------------------------------
print("\nLogistic Regression")

X2 = np.array([[0],[1],[2],[3]])
y2 = np.array([0,0,1,1])

model2 = Sequential([
    Dense(1, input_dim=1, activation='sigmoid')
])

model2.compile(optimizer='adam',
               loss='binary_crossentropy',
               metrics=['accuracy'])

model2.fit(X2, y2, epochs=500, verbose=0)

# Evaluation
loss, acc = model2.evaluate(X2, y2, verbose=0)

print("Accuracy:", acc)

--------------------------------------------------------------------------------------------------
#11
import tensorflow as tf
import numpy as np

# Training data
X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6]
], dtype=float)

y = np.array([
    [0],
    [0],
    [0],
    [1],
    [1],
    [1]
], dtype=float)

# Create model
model = tf.keras.Sequential([
    
    tf.keras.Input(shape=(1,)),
    
    tf.keras.layers.Dense(
        1,
        activation='sigmoid'
    )
])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(X, y, epochs=100)

# Evaluate
loss, accuracy = model.evaluate(X, y)

print("Loss =", loss)
print("Accuracy =", accuracy)

# Prediction
prediction = model.predict(
    np.array([[7]], dtype=float)
)

print("Prediction for 7 =", prediction)

---------------------------------------------------------------------------------------------

# 12
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# -------- Step 1: Load dataset --------
(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

# Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

# Add channel dimension
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# -------- Step 2: Build CNN --------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# -------- Step 3: Compile --------
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# -------- Step 4: Train --------
model.fit(X_train, y_train, epochs=3)

# -------- Step 5: Evaluate --------
test_loss, test_acc = model.evaluate(X_test, y_test)

print("Test Accuracy:", test_acc)

------------------------------------------------------------------------------------------------
#12 
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# ---------------- LOAD DATA ----------------

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Reshape for CNN
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Normalize
train_images = train_images / 255.0
test_images = test_images / 255.0

# ---------------- CREATE CNN MODEL ----------------

model = models.Sequential([

    tf.keras.Input(shape=(28,28,1)),

    # Convolution Layer
    layers.Conv2D(32, (3,3), activation='relu'),

    # Pooling Layer
    layers.MaxPooling2D((2,2)),

    # Second Convolution Layer
    layers.Conv2D(64, (3,3), activation='relu'),

    # Pooling Layer
    layers.MaxPooling2D((2,2)),

    # Flatten
    layers.Flatten(),

    # Dense Layer
    layers.Dense(64, activation='relu'),

    # Dropout reduces overfitting
    layers.Dropout(0.5),

    # Output Layer
    layers.Dense(10, activation='softmax')
])

# ---------------- COMPILE MODEL ----------------

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ---------------- TRAIN MODEL ----------------

history = model.fit(
    train_images,
    train_labels,
    epochs=5,
    validation_data=(test_images, test_labels)
)

# ---------------- EVALUATE MODEL ----------------

loss, accuracy = model.evaluate(
    test_images,
    test_labels
)

print("Test Accuracy =", accuracy)

# ---------------- PREDICT ----------------

import numpy as np

predictions = model.predict(test_images)

predicted_digit = np.argmax(predictions[0])

print("Predicted Digit =", predicted_digit)
print("Actual Digit =", test_labels[0])

# ---------------- PLOT GRAPHS ----------------

# Accuracy Graph
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'])

plt.show()

# Loss Graph
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'])

plt.show()

-------------------------------------------------------------------------------------------------

# 13
import tensorflow as tf
from tensorflow.keras import layers, models

# Load data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize
X_train, X_test = X_train/255.0, X_test/255.0

# Reshape
X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)

# Model
model = models.Sequential([
    tf.keras.Input(shape=(28,28,1)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile & Train
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=3)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print("Accuracy:", acc)

-------------------------------------------------------------------------------------------------------
#13
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import numpy as np

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize values
x_train = x_train / 255.0
x_test = x_test / 255.0

# Create model
model = Sequential([

    Flatten(input_shape=(28, 28)),

    Dense(128, activation='relu'),

    Dense(10, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(
    x_train,
    y_train,
    epochs=5
)

# Evaluate model
loss, accuracy = model.evaluate(
    x_test,
    y_test
)

print("Accuracy =", accuracy)

# Predict
prediction = model.predict(x_test)

predicted_digit = np.argmax(prediction[0])

print("Predicted Digit =", predicted_digit)
print("Actual Digit =", y_test[0])
