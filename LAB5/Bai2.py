import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from src.tree_utils import evaluate_best_feature_split

# Bài 2: Tự tính toán chỉ số Gini Gain & so sánh với Scikit-Learn
df = pd.read_csv("data/khach_vay.csv")
X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=17, stratify=y)

best_col, best_val, gini_init, gini_gain = evaluate_best_feature_split(X_tr, y_tr)
print(f"Câu hỏi tự tính: {best_col} <= {best_val:.4f}")
print(f"Gini gốc: {gini_init:.4f} | Gini giảm: {gini_gain:.4f}")

sk_tree = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X_tr, y_tr)
sk_col = X.columns[sk_tree.tree_.feature[0]]
sk_val = sk_tree.tree_.threshold[0]

print(f"Câu hỏi sklearn: {sk_col} <= {sk_val:.4f}")
print("Hai kết quả trùng nhau:", (best_col == sk_col) and np.isclose(best_val, sk_val))
