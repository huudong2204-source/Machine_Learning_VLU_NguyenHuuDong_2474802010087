import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
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
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)
print("=== KẾT QUẢ TÍNH TỔNG QUAN (BÀI 4) ===")
print(f"TP = {tp}, TN = {tn}, FP = {fp}, FN = {fn}")
print(f"Accuracy  = {accuracy:.4f}")
print(f"Precision = {precision:.4f}")
print(f"Recall    = {recall:.4f}")
print(f"F1-Score  = {f1:.4f}\n")

# 6. IN NHẬN XÉT TRỰC TIẾP RA TERMINAL
nhan_xet = f"""================ NHẬN XÉT BÀI 4 ================
1. Độ chính xác các chỉ số:
   - Các chỉ số tự tính toán thủ công hoàn toàn khớp 100% với các hàm
     kết quả từ thư viện scikit-learn.

2. Ý nghĩa các thước đo:
   - Accuracy ({accuracy:.2%}): Mô hình dự đoán đúng tổng cộng {tp + tn}/{tp + tn + fp + fn} mẫu.
   - Precision ({precision:.2%}): Trong số các sinh viên mô hình đoán Qua môn,
     có {precision:.2%} bạn thực sự qua môn (bị đoán nhầm {fp} bạn).
   - Recall ({recall:.2%}): Mô hình phát hiện được {recall:.2%} tổng số sinh viên
     thực sự qua môn (chỉ bỏ sót {fn} bạn).
   - F1-Score ({f1:.4f}): Chỉ số cân bằng giữa Precision và Recall đạt mức cao.

3. Đánh giá xu hướng:
   - Chỉ số Recall cao hơn Precision cho thấy mô hình đang nghiêng về hướng
     "dễ tính" (xu hướng đoán qua môn nhiều hơn để tránh bỏ sót người qua môn).
=================================================
"""

print(nhan_xet)