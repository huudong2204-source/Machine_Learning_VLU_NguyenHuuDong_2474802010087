import pandas as pd
from sklearn.model_selection import train_test_split
from src.regression_models import compare_housing_regression

# Bài 6: Dự đoán giá nhà và đánh giá khả năng ngoại suy
df = pd.read_csv("data/gia_nha.csv")
X = df[["dien_tich"]]
y = df["gia"]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

df_res, rf_pred, tree_mean = compare_housing_regression(X_tr, y_tr, X_te, y_te)
print("=== SO SÁNH MÔ HÌNH HỒI QUY GIÁ NHÀ ===")
print(df_res.to_string(index=False))
print(f"\nGiá 250m2 do RF dự đoán: {rf_pred:.4f}")
print(f"Trung bình dự đoán 200 cây: {tree_mean:.4f}")
