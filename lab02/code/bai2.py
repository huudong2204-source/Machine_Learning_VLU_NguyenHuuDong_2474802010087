import matplotlib.pyplot as plt
import numpy as np
def sigmoid(z):
  return 1 / (1 + np.exp(-z))
z = np.linspace(-8, 8, 200)
y = sigmoid(z)
plt.figure(figsize=(8, 5))
plt.plot(z, y, label=r"$\sigma(z) = \frac{1}{1 + e^{-z}}$", color="blue")
plt.axhline(0.5, color="gray", linestyle="--", label="y = 0.5")
plt.axvline(0, color="gray", linestyle=":", label="z = 0")
plt.xlabel("z")
plt.ylabel(r"$\sigma(z)$")
plt.title("Đồ thị hàm Sigmoid")
plt.legend()
plt.grid(True)
plt.savefig("output/sigmoid.png")
plt.show()