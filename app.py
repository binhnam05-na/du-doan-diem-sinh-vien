import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Mô hình dự đoán điểm trúng tuyển đại học", layout="wide")
st.title("🎓 Mô hình dự đoán điểm trúng tuyển đại học")

# -------------------------
# Sidebar nhập điểm và số lượng sinh viên
# -------------------------
st.sidebar.header("📥 Nhập thông tin")

n_students = st.sidebar.slider("Số lượng sinh viên mô phỏng", 100, 2000, 500, step=50)

# Nhập điểm cá nhân
toan = st.sidebar.slider("Điểm Toán", 0.0, 10.0, 0.0, 0.05)
ly = st.sidebar.slider("Điểm Lý", 0.0, 10.0, 0.0, 0.05)
hoa = st.sidebar.slider("Điểm Hóa", 0.0, 10.0, 0.0, 0.05)

# -------------------------
# Dữ liệu điểm chuẩn mô phỏng qua các năm
# -------------------------
years = np.arange(2018, 2024)
nganh_list = ["CNTT","Kinh tế","Kỹ thuật","Luật","Mỹ thuật","Ngôn ngữ Anh",
              "Quản trị kinh doanh","Truyền thông","Tài chính","Y dược"]

np.random.seed(42)
data_chuan = pd.DataFrame({
    "Năm": np.repeat(years, len(nganh_list)),
    "Ngành": nganh_list*len(years),
    "DiemChuan": np.round(np.random.uniform(18,26,len(years)*len(nganh_list)),2)
})

# -------------------------
# Layout chính
# -------------------------
col1, col2 = st.columns([1,3])

# -------------------------
# Cột trái: điểm cá nhân
# -------------------------
with col1:
    st.subheader("📥 Điểm cá nhân")
    st.write(f"- Toán: {toan}")
    st.write(f"- Lý: {ly}")
    st.write(f"- Hóa: {hoa}")
    tong_user = toan + ly + hoa
    st.write(f"**Tổng điểm 3 môn: {tong_user:.2f}**")

# -------------------------
# Cột phải: biểu đồ điểm chuẩn qua các năm
# -------------------------
with col2:
    st.subheader("📈 Biến động điểm chuẩn theo ngành qua các năm")
    plt.figure(figsize=(10,5))
    sns.lineplot(data=data_chuan, x="Năm", y="DiemChuan", hue="Ngành", marker="o")
    
    # Thêm điểm cá nhân
    plt.axhline(y=tong_user, color='red', linestyle='--', label="Tổng điểm cá nhân")
    plt.text(years[-1]+0.1, tong_user, f"{tong_user:.2f}", color='red')
    
    plt.ylabel("Điểm chuẩn trung bình")
    plt.xlabel("Năm")
    plt.title("Biến động điểm chuẩn trung bình theo ngành")
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    st.pyplot(plt)
