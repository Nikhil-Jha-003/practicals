# ML Practical 3: Gradient Descent
# Minimize y = (x + 3)^2 starting from x = 2.

import numpy as np
import matplotlib.pyplot as plt

# 1. Define the function and its derivative
def function(x):
    return (x + 3) ** 2

def derivative(x):
    return 2 * (x + 3)

# 2. Initialize parameters
x = 2.0
learning_rate = 0.1
iterations = 50

x_values = [x]
y_values = [function(x)]

# 3. Apply gradient descent
print("Iteration\t x\t\t y")
for i in range(iterations):
    gradient = derivative(x)
    x = x - learning_rate * gradient

    x_values.append(x)
    y_values.append(function(x))

    print(f"{i + 1}\t\t {x:.6f}\t {function(x):.6f}")

# 4. Display final minimum
print("\nMinimum point:")
print("x =", round(x, 6))
print("y =", round(function(x), 6))

# 5. Plot the function and the descent path
x_plot = np.linspace(-6, 4, 300)
y_plot = function(x_plot)

plt.figure(figsize=(8, 5))
plt.plot(x_plot, y_plot, label="y = (x + 3)^2")
plt.scatter(x_values, y_values, color="red", s=18, label="Gradient descent steps")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Gradient Descent")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
