import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/sinh_vien.csv")
X = df[["gio_on"]]
y = df["qua_mon"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

mo_hinh = LogisticRegression()
mo_hinh.fit(X_train, y_train)

p = mo_hinh.predict_proba(X_test)[:, 1]

nguong_list = np.arange(0.05, 1.0, 0.05)
f1_tot_nhat = -1
nguong_tot_nhat = None

print("Ngưỡng\tF1-Score")
for nguong in nguong_list:
  y_pred = (p >= nguong).astype(int)
  score = f1_score(y_test, y_pred, zero_division=0)
  print(f"{nguong:.2f}\t{score:.4f}")

  if score > f1_tot_nhat:
    f1_tot_nhat = score
    nguong_tot_nhat = nguong

print(
    f"\nNgưỡng cho F1 cao nhất: {nguong_tot_nhat:.2f} (F1 = {f1_tot_nhat:.4f})"
)
#Nhận xét:

#Ngưỡng tốt nhất theo chỉ số F1 không nhất thiết phải bằng đúng 0.5 vì chỉ số F1 phụ thuộc vào sự cân bằng giữa Precision và Recall.
#Tùy thuộc vào phân bố dữ liệu kiểm tra, ngưỡng tối ưu có thể bị lệch sang trái hoặc phải.   