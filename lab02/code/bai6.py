import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/sinh_vien.csv")
X = df[["gio_on"]]
y = df["qua_mon"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

mo_hinh = LogisticRegression()
mo_hinh.fit(X_train, y_train)
y_pred = mo_hinh.predict(X_test)

# Lớp dương là lớp 0 (Rớt môn)
precision_0 = precision_score(y_test, y_pred, pos_label=0)
recall_0 = recall_score(y_test, y_pred, pos_label=0)

# Lớp dương là lớp 1 (Qua môn)
precision_1 = precision_score(y_test, y_pred, pos_label=1)
recall_1 = recall_score(y_test, y_pred, pos_label=1)

print(f"Lớp 0 (Rớt môn): Precision = {precision_0:.4f}, Recall = {recall_0:.4f}")
print(f"Lớp 1 (Qua môn) : Precision = {precision_1:.4f}, Recall = {recall_1:.4f}")

#Giải thích: Hai bộ số khác nhau vì đối tượng "dương tính" (positive class) cần tính toán đã thay đổi:

#Với lớp 1 (Qua môn): Tử số là TP (đoán qua, thật sự qua).

#Với lớp 0 (Rớt môn): Lúc này đối tượng quan tâm thay đổi, TN (đoán rớt, thật sự rớt) trở thành điểm dương tính thực sự. 
#Do đó công thức tính Precision và Recall bị hoán đổi vai trò của các số hạng trong ma trận nhầm lẫn.  