
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
