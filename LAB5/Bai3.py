import pandas as pd
from sklearn.model_selection import train_test_split
from src.model_evaluation import evaluate_depth_and_criteria

# Bài 3: Đánh giá tác động của max_depth & so sánh Gini vs Entropy
df = pd.read_csv("data/khach_vay.csv")
X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=17, stratify=y)

df_depths, df_comp = evaluate_depth_and_criteria(X_tr, y_tr, X_te, y_te)
print("=== KẾT QUẢ THEO ĐỘ SÂU ===")
print(df_depths.to_string(index=False))

print("\n=== SO SÁNH GINI VÀ ENTROPY ===")
print(df_comp.to_string(index=False))
