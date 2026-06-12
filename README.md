# Neural Network Visualizer — From Scratch Using NumPy

An interactive visualization of a fully connected neural network implemented from scratch using only NumPy concepts and vanilla JavaScript.

The project demonstrates:

* Forward Propagation
* Binary Cross Entropy Loss
* Backpropagation
* Gradient Descent
* ReLU Activation
* Sigmoid Activation
* Live Training Visualization
* Loss Curve Tracking

No TensorFlow. No PyTorch. No high-level deep learning frameworks.

---

# Network Architecture

```text
Input Layer (4)
      ↓
Hidden Layer 1 (4 ReLU)
      ↓
Hidden Layer 2 (3 ReLU)
      ↓
Output Layer (1 Sigmoid)
```

Dataset Rule:

```text
Output = 1
if sum of the four inputs ≥ 3

otherwise Output = 0
```

Example:

```text
[1,1,1,0] → 1
[1,0,0,1] → 0
[1,1,1,1] → 1
```

---

# Weight and Bias Shapes

## Layer 1

```python
W1.shape = (4,4)
b1.shape = (1,4)
```

## Layer 2

```python
W2.shape = (3,4)
b2.shape = (1,3)
```

## Output Layer

```python
W3.shape = (1,3)
b3.shape = (1,1)
```

---

# Forward Propagation

The goal of forward propagation is to convert inputs into predictions.

---

## Step 1: Hidden Layer 1

### Equation

```math
Z_1 = XW_1^T + b_1
```

### Why?

Each neuron computes:

```math
z = w_1x_1+w_2x_2+w_3x_3+w_4x_4+b
```

For all samples and all neurons simultaneously, we use matrix multiplication.

### Activation

```math
A_1 = ReLU(Z_1)
```

ReLU:

```math
ReLU(x)=max(0,x)
```

Negative values become zero.

---

## Step 2: Hidden Layer 2

### Equation

```math
Z_2=A_1W_2^T+b_2
```

The output of Layer 1 becomes the input of Layer 2.

### Activation

```math
A_2=ReLU(Z_2)
```

Again, negative values are removed.

---

## Step 3: Output Layer

### Equation

```math
Z_3=A_2W_3^T+b_3
```

### Sigmoid Activation

```math
A_3=\sigma(Z_3)
```

where

```math
\sigma(x)=\frac{1}{1+e^{-x}}
```

This converts any number into a probability between 0 and 1.

Example:

```text
2.5 → 0.924
-3 → 0.047
```

---

# Loss Function

The network must know how wrong its predictions are.

Binary Cross Entropy:

```math
L=-\frac1m
\sum
[y\log(\hat y)+(1-y)\log(1-\hat y)]
```

where

```text
y      = actual label
ŷ      = prediction
m      = number of samples
```

Perfect prediction:

```text
Loss ≈ 0
```

Bad prediction:

```text
Loss is large
```

---

# Backpropagation

Backpropagation determines how every weight contributed to the loss.

The error starts at the output layer and moves backward.

---

## Output Layer Error

For Sigmoid + Binary Cross Entropy:

```math
dZ_3=A_3-y
```

This famous simplification comes from combining:

```math
\frac{\partial Loss}{\partial A_3}
```

and

```math
\frac{\partial A_3}{\partial Z_3}
```

---

## Output Layer Gradients

### Weight Gradient

```math
dW_3=\frac1m(dZ_3)^TA_2
```

Why?

Gradient = Error × Input

Current error:

```math
dZ_3
```

Input feeding this layer:

```math
A_2
```

Therefore:

```math
dW_3 = Error × Input
```

---

### Bias Gradient

```math
db_3=\frac1m\sum dZ_3
```

Bias affects every sample equally.

Therefore we simply sum all errors.

---

# Hidden Layer 2

---

## Step 1: Propagate Error Backward

```math
dA_2=dZ_3W_3
```

Meaning:

```text
How much did each neuron in A2
contribute to the final output error?
```

---

## Step 2: ReLU Derivative

Forward:

```math
A_2=ReLU(Z_2)
```

Derivative:

```math
ReLU'(x)=
\begin{cases}
1,&x>0\\
0,&x\le0
\end{cases}
```

Therefore:

```math
dZ_2=dA_2 \odot ReLU'(Z_2)
```

where

```text
⊙ = element-wise multiplication
```

Inactive neurons receive zero gradient.

---

## Step 3: Layer 2 Gradients

### Weight Gradient

```math
dW_2=\frac1m(dZ_2)^TA_1
```

Again:

```text
Gradient = Error × Input
```

Error:

```math
dZ_2
```

Input:

```math
A_1
```

---

### Bias Gradient

```math
db_2=\frac1m\sum dZ_2
```

---

# Hidden Layer 1

The same process repeats.

---

## Error Propagation

```math
dA_1=dZ_2W_2
```

---

## ReLU Backward

```math
dZ_1=dA_1 \odot ReLU'(Z_1)
```

---

## Gradients

```math
dW_1=\frac1m(dZ_1)^TX
```

```math
db_1=\frac1m\sum dZ_1
```

Notice that the input to Layer 1 is the original dataset:

```math
X
```

which is why it appears in the equation.

---

# Gradient Descent

Once gradients are computed:

```math
W=W-\eta dW
```

```math
b=b-\eta db
```

where

```text
η = learning rate
```

This moves parameters in the direction that reduces loss.

---

# Training Loop

```text
Forward Propagation
        ↓
Loss Calculation
        ↓
Backpropagation
        ↓
Gradient Descent
        ↓
Repeat
```

---

# What This Project Demonstrates

* Neural Networks from Scratch
* Matrix-Based Computation
* ReLU and Sigmoid Activations
* Binary Cross Entropy
* Chain Rule
* Backpropagation
* Gradient Descent
* Interactive Neural Network Visualization

This project was built to understand how neural networks work internally rather than relying on high-level deep learning frameworks.
