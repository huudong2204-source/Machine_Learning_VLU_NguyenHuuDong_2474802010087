import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from src.tree_utils import export_tree_visualization

# Bài 1: Cây 5 lá và Cây độ sâu = 2
df = pd.read_csv("data/khach_vay.csv")
X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=17, stratify=y)

print(f"Số khách tập học: {len(X_tr)}")
print(f"Số khách tập kiểm tra: {len(X_te)}")

model_5leaf = DecisionTreeClassifier(max_leaf_nodes=5, random_state=0).fit(X_tr, y_tr)
export_tree_visualization(model_5leaf, list(X.columns), ["Tre han", "Dung han"], "Cay quyet dinh nam la", "cay_nam_la.png")

model_depth2 = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X_tr, y_tr)
export_tree_visualization(model_depth2, list(X.columns), ["Tre han", "Dung han"], "Cay quyet dinh sau 2", "cay_sau_2.png")

left_node = model_depth2.tree_.children_left[0]
sub_leaves = [model_depth2.tree_.children_left[left_node], model_depth2.tree_.children_right[left_node]]

conclusions = []
for idx, leaf_id in enumerate(sub_leaves, start=1):
    counts = model_depth2.tree_.value[leaf_id][0]
    probs = counts / counts.sum()
    pred_label = model_depth2.classes_[np.argmax(probs)]
    conclusions.append(pred_label)
    print(f"Lá trái {idx}: class={pred_label}, tỷ lệ trễ hạn={probs[0]:.4f}, tỷ lệ đúng hạn={probs[1]:.4f}")

print("Hai lá cùng kết luận:", conclusions[0] == conclusions[1])
