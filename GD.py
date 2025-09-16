import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

ALPHA = None
N_ITER = 5000
TOL = 1e-9
LOG_EVERY = 50

df = pd.read_excel("Concrete_Data.xlsx")
df.columns = [c.strip() for c in df.columns]

y_col = "Concrete compressive strength(MPa, megapascals)"
X_cols = [c for c in df.columns if c != y_col]

test_start, test_end = 501, 630
test_idx = list(range(test_start, test_end + 1))
train_idx = [i for i in range(len(df)) if i < test_start or i > test_end]

train = df.iloc[train_idx].reset_index(drop=True)
test  = df.iloc[test_idx].reset_index(drop=True)

X_train = train[X_cols].to_numpy()
y_train = train[y_col].to_numpy()
X_test  = test[X_cols].to_numpy()
y_test  = test[y_col].to_numpy()

X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test_b  = np.c_[np.ones((X_test.shape[0], 1)),  X_test]

m, n = X_train_b.shape
XT_X = X_train_b.T @ X_train_b
eigvals = np.linalg.eigvals(XT_X).real
L = (2.0 / m) * eigvals.max()
alpha = ALPHA if ALPHA is not None else 0.9 / L

theta = np.zeros(n)
loss_hist = []
last_mse = None

for t in range(1, N_ITER + 1):
    y_pred = X_train_b @ theta
    grad = (2.0 / m) * (X_train_b.T @ (y_pred - y_train))
    theta = theta - alpha * grad

    if t % LOG_EVERY == 0 or t == 1 or t == N_ITER:
        mse_tr = mean_squared_error(y_train, X_train_b @ theta)
        loss_hist.append((t, mse_tr))
        if last_mse is not None:
            rel_improve = (last_mse - mse_tr) / max(1e-12, last_mse)
            if rel_improve < TOL and t > LOG_EVERY:
                break
        last_mse = mse_tr

b = theta[0]
coefs = theta[1:]

yhat_tr = X_train_b @ theta
yhat_te = X_test_b  @ theta

metrics = {
    "alpha (learning rate)": alpha,
    "iterations_run": loss_hist[-1][0] if loss_hist else 0,
    "Train MSE": mean_squared_error(y_train, yhat_tr),
    "Train R^2": r2_score(y_train, yhat_tr),
    "Test  MSE": mean_squared_error(y_test,  yhat_te),
    "Test  R^2": r2_score(y_test,  yhat_te),
}

coef_table = pd.DataFrame({
    "Predictor": ["Intercept"] + X_cols,
    "Coefficient": [b] + coefs.tolist()
})

loss_df = pd.DataFrame(loss_hist, columns=["iteration", "train_mse"])
coef_csv = "gd_raw_coefficients.csv"
metrics_csv = "gd_raw_metrics.csv"
loss_csv = "gd_raw_loss_history.csv"
coef_table.to_csv(coef_csv, index=False)
pd.DataFrame([metrics]).to_csv(metrics_csv, index=False)
loss_df.to_csv(loss_csv, index=False)
coef_csv, metrics_csv, loss_csv
