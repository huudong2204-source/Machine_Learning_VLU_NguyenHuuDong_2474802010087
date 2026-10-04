import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression

# Giữ nguyên đường dẫn của bạn
path_csv = "data/sinh_vien.csv"
path_anh = "data/giaithichbai3.png"

# 1. Đọc dữ liệu và khớp mô hình một biến
df = pd.read_csv(path_csv)
X = df[["gio_on"]]
y = df["qua_mon"]

mo_hinh = LogisticRegression()
mo_hinh.fit(X, y)


# 2. Hàm dự đoán du_doan(gio)
def du_doan(gio):
  w = float(mo_hinh.coef_[0][0])
  b = float(mo_hinh.intercept_[0])

  z = w * gio + b
  df_input = pd.DataFrame({"gio_on": [gio]})
  p = mo_hinh.predict_proba(df_input)[0, 1]
  c_hat = int(p >= 0.5)

  print(
      f"Số giờ ôn: {gio:5.2f} | z: {z:6.4f} | Xác suất qua môn: {p:.4f} | Nhãn:"
      f" {c_hat}"
  )
print("Dự đoán kết quả theo số giờ ôn:")
for g in [3, 8, 12.89, 18, 26]:
  du_doan(g)

# 4. Kiểm tra và vẽ ảnh nếu chưa tồn tại
if not os.path.exists(path_anh):
  os.makedirs(os.path.dirname(path_anh), exist_ok=True)

  z_vals = np.linspace(-6, 6, 200)
  p_vals = 1 / (1 + np.exp(-z_vals))

  plt.figure(figsize=(7, 4))
  plt.plot(z_vals, p_vals, label="Sigmoid", color="blue")
  plt.axvline(0, color="red", linestyle="--", label="z = 0 (p = 0.5)")
  plt.axhline(0.5, color="gray", linestyle=":")
  plt.title("Giải thích: Tại z = 0 thì xác suất p = 0.5")
  plt.xlabel("z = w*x + b")
  plt.ylabel("Xác suất (p)")
  plt.legend()
  plt.grid(True)
  plt.savefig(path_anh)
  plt.close()


if os.path.exists(path_anh):
  img = Image.open(path_anh)
  img.show()


  input("\nĐã mở ảnh thành công! Nhấn Enter trong Terminal để đóng...")