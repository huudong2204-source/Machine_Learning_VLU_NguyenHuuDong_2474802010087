import pandas as pd
from sklearn.model_selection import train_test_split
from src.model_evaluation import compare_pruning_strategies

# Bài 4: So sánh các phương pháp cắt tỉa min_samples_leaf vs max_depth
df = pd.read_csv("data/khach_vay.csv")
X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=17, stratify=y)

leaf_df, test_summary, best_leaf = compare_pruning_strategies(X_tr, y_tr, X_te, y_te)
print("=== KẾT QUẢ KIỂM ĐỊNH CHÉO MIN_SAMPLES_LEAF ===")
print(leaf_df.to_string(index=False))
print(f"\nMin_samples_leaf tốt nhất: {best_leaf}")

print("\n=== SO SÁNH HAI CÁCH CẮT TỈA ===")
print(test_summary.to_string(index=False))
