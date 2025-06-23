import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# 1. Generate Sample Data (with noise)
np.random.seed(0)
X = np.sort(np.random.rand(30, 1) * 10, axis=0)   # 30 points between 0 and 10
y = np.sin(X).ravel() + np.random.randn(30) * 0.3 # sin curve with noise

# 2. Prepare models with different complexities (degrees)
degrees = [1, 4, 15]  # Linear, good fit, overfit

# 3. Plot them
plt.figure(figsize=(15, 4))
X_test = np.linspace(0, 10, 100).reshape(-1, 1)  # smooth test points

for i, d in enumerate(degrees):
    model = make_pipeline(PolynomialFeatures(degree=d), LinearRegression())
    model.fit(X, y)
    y_pred = model.predict(X_test)
    
    plt.subplot(1, 3, i + 1)
    plt.scatter(X, y, color='black', label="Training Data")
    plt.plot(X_test, y_pred, color='blue', label=f"Degree {d}")
    plt.title(f"Degree {d} (MSE: {mean_squared_error(y, model.predict(X)):.2f})")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()

plt.suptitle("Underfitting vs Good Fit vs Overfitting")
plt.tight_layout()
plt.show()
