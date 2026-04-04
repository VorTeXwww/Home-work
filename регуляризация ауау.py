import numpy as np

np.random.seed(42)
N = 60
X = np.linspace(-3, 3, N)
y_true = np.sin(X) * 2 + X * 0.5
y_hat = y_true + np.random.normal(0, 0.6, N)

b = 0
w1 = 0
w2 = 0
alpha = 0.1
lr = 0.1
epochs = 100
eps = 0.001

for epoch in range(epochs):
    y_pred = b  + w1 * np.sin(X) + w2 * X
    mse_start = np.mean((y_hat - y_pred) ** 2)

    ridge = mse_start + alpha * (w1 ** 2 + w2 ** 2)

    grad_b = 0
    grad1 = 0
    grad2 = 0

    error = y_pred - y_hat

    grad_b += (2 / N) * np.sum(error)
    grad1 += (2 / N) * np.sum(error * np.sin(X)) + alpha * w1 * 2
    grad2 += (2/ N) * np.sum(error * X )+ alpha * w2 * 2

    b = b - lr * grad_b
    w1 = w1 - lr * grad1
    w2 = w2 - lr * grad2

    y_pred_new = b + w1 * np.sin(X) + w2 * X
    mse_end = np.mean((y_hat - y_pred_new) ** 2)

    if abs(mse_start - mse_end) < eps:
        print("Stop! big raznica")
        break

    print(f"Epochs: {epoch:}")
    print(f"Start mse: {mse_start}")
    print(f"Ridge: {ridge}")
    print(f"End mse: {mse_end}")