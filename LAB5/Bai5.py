import pandas as pd
from sklearn.model_selection import train_test_split
from src.feature_analysis import analyze_feature_importances

# Bài 5: Phân tích độ quan trọng thuộc tính MDI vs Permutation Importance
df = pd.read_csv("data/khach_vay.csv")
X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=17, stratify=y)

imp_df, test_acc = analyze_feature_importances(X_tr, y_tr, X_te, y_te)
print(f"Accuracy tập kiểm tra: {test_acc:.4f}\n")
print("=== ĐỘ QUAN TRỌNG CỦA NĂM CỘT ===")
print(imp_df.round(4).to_string())
