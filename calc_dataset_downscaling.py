import numpy as np
import os
import matplotlib.pyplot as plt


x = np.array([22.8e3, 26.6e3])

y1 = 4.8818638644 * x + -31340.379702

y2= np.exp(3.5682497788757034e-05 * x + 10.179345731968857)


x = np.array([59.5e3, 98.2e3])
print("lol", 0.19263265557080517 * x + 1263.06712014996)

print(y1, y2)


entries = [
   [ 0.1, .11, 0.99,],
  [  0.2, 1, 0,],
    [0.3, 0.18, 0.99,],
    [0.4, 0.56, 0.92,],
    [0.5, 0.27, 0.99,],
    [0.6, 0.92, 0.8,],
    [0.7, 0.83, 0.92,],
    [0.8, 0.79, 0.94,],
    [0.9, 0.84, 0.93,],
  [  0.1, 0, 1,],
    [0.2, 0.042, 0.99,],
  [  0.3, 0, 1,],
    [0.4, 0.73, 0.76,],
    [0.5, 0.97, 0.57,],
    [0.6, 0.75, 0.94,],
    [0.7, 0.84, 0.9,],
    [0.8, 0.93, 0.77,],
    [0.9, 0.43, 0.98,],
    ]

entries = np.array(entries)

x = entries[:,0]
y = (entries[:,1] + entries[:,2]) / 2

# sigmoid regression
def sigmoid(x, a, b, c):
    return a / (1 + np.exp(-b * (x - c)))
from scipy.optimize import curve_fit
popt, pcov = curve_fit(sigmoid, x, y, p0=[0, 0, 0])
print(popt)
plt.plot(np.arange(0.1, 1, 0.01), sigmoid(np.arange(0.1, 1, 0.01), *popt), 'k--', alpha = 0.5, label = f"y = {popt[0]:.2f} / (1 + e^({popt[1]:.2f} * (x - {popt[2]:.2f}))")
plt.scatter(x, y)
plt.title("Sigmoid regression")
plt.xlabel("percentage of dataset")
plt.ylabel("average accuracy")
plt.legend()
plt.show()