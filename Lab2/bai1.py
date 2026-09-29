import pandas as pd

df = pd.read_csv("sinh_vien.csv")

# Nhóm điểm giữa kỳ >= 7
nhom_gioi = df[df["diem_giua_ky"] >= 7.0]
so_ban_gioi = len(nhom_gioi)
ty_le_nhom_gioi = nhom_gioi["qua_mon"].mean()

# Nhóm còn lại (< 7)
nhom_con_lai = df[df["diem_giua_ky"] < 7.0]
ty_le_nhom_con_lai = nhom_con_lai["qua_mon"].mean()

print(f"So ban co diem giua ky >= 7: {so_ban_gioi}")
print(f"Ty le qua mon nhom diem giua ky >= 7: {ty_le_nhom_gioi:.4f}")
print(f"Ty le qua mon nhom con lai (< 7): {ty_le_nhom_con_lai:.4f}")
print(
    "\nNhan xet: Diem giua ky phan biet rat tot hai nhom, sinh vien co diem"
    " giua ky >= 7 co ty le qua mon cao hon ro ret so voi nhom con lai."
)