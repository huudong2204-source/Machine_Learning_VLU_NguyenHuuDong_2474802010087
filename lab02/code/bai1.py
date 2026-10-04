import pandas as pd
# Đọc bộ dữ liệu
df = pd.read_csv("data/sinh_vien.csv")
# Lọc nhóm sinh viên có điểm giữa kỳ từ 7 trở lên và nhóm còn lại
nhom_tren_7 = df[df["diem_giua_ky"] >= 7.0]
nhom_duoi_7 = df[df["diem_giua_ky"] < 7.0]
# 1. Số bạn có điểm giữa kỳ từ 7 trở lên
so_luong_tren_7 = len(nhom_tren_7)
# 2. Tỷ lệ qua môn của nhóm điểm >= 7
ty_le_tren_7 = nhom_tren_7["qua_mon"].mean()
# 3. Tỷ lệ qua môn của nhóm điểm < 7
ty_le_duoi_7 = nhom_duoi_7["qua_mon"].mean()
print(f"Số sinh viên có điểm giữa kỳ >= 7: {so_luong_tren_7}")
print(f"Tỷ lệ qua môn của nhóm có điểm giữa kỳ >= 7: {ty_le_tren_7:.4f}")
print(f"Tỷ lệ qua môn của nhóm có điểm giữa kỳ < 7: {ty_le_duoi_7:.4f}")
#Nhận xét:
#Điểm giữa kỳ phân biệt rất tốt giữa hai nhóm sinh viên. 
#Tỷ lệ qua môn của nhóm có điểm giữa kỳ từ 7 trở lên cao hơn vượt trội so với nhóm có điểm giữa kỳ dưới 7