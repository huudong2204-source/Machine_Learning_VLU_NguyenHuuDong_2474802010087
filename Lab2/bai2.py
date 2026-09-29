# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

z = np.linspace(-8, 8, 200)
sigma = 1 / (1 + np.exp(-z))

plt.figure(figsize=(8, 5))
plt.plot(z, sigma, label=r"$\sigma(z) = \frac{1}{1 + e^{-z}}$", color="blue", lw=2)
plt.axhline(0.5, color="gray", linestyle="--", label="Muc 0.5")
plt.axvline(0, color="red", linestyle=":", label="z = 0")

plt.title("Do thi ham Sigmoid")
plt.xlabel("z")
plt.ylabel(r"$\sigma(z)$")
plt.grid(True, alpha=0.3)
plt.legend()

plt.savefig("sigmoid.png", dpi=300)
print("Da luu do thi thanh tep sigmoid.png!")