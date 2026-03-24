import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Mô hình dự đoán điểm trúng tuyển đại học", layout="wide")
st.title("🎓 Mô hình dự đoán điểm trúng tuyển đại học")

# -------------------------
# Sidebar: điểm cá nhân & số lượng sinh viên mô phỏng
# -------------------------
st.sidebar.header("📥 Nhập thông tin")

n_students = st.sidebar.slider("Số lượng sinh viên mô phỏng", 100, 2000, 500, step=50)

# Điểm cá nhân
toan = st.sidebar.slider("Điểm Toán", 0.0, 10.0, 0.0, 0.05)
ly = st.sidebar.slider("Điểm Lý", 0.0, 10.0, 0.0, 0.05)
hoa = st.sidebar.slider("Điểm Hóa", 0.0, 10.0, 0.0, 0.05)

tong_user = toan + ly + hoa

# -------------------------
# Dữ liệu mô phỏng điểm chuẩn qua các năm
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
# Mô phỏng sinh viên
# -------------------------
np.random.seed(42)
df_students = pd.DataFrame({
    "Toan": np.random.uniform(0,10,n_students),
    "Ly": np.random.uniform(0,10,n_students),
    "Hoa": np.random.uniform(0,10,n_students)
})

df_students["TongDiem"] = df_students["Toan"] + df_students["Ly"] + df_students["Hoa"]

# Giả lập xác suất trúng tuyển theo ngành (cao hơn nếu tổng điểm > điểm chuẩn)
# Thêm noise ngẫu nhiên để phân bố sinh viên
probs = []
for ng in nganh_list:
    dc = data_chuan[data_chuan["Ngành"]==ng]["DiemChuan"].mean()
    prob = 1/(1+np.exp(-(df_students["TongDiem"]-dc)/2))  # logistic function
    probs.append(prob)

df_probs = pd.DataFrame(np.array(probs).T, columns=nganh_list)
df_students = pd.concat([df_students, df_probs], axis=1)

# -------------------------
# Layout
# -------------------------
col1, col2 = st.columns([1,3])

# -------------------------
# Cột trái: thông tin cá nhân
# -------------------------
with col1:
    st.subheader("📥 Điểm cá nhân")
    st.write(f"- Toán: {toan}")
    st.write(f"- Lý: {ly}")
    st.write(f"- Hóa: {hoa}")
    st.write(f"**Tổng điểm 3 môn: {tong_user:.2f}**")
    
    st.subheader("📊 Xác suất trúng tuyển dự kiến")
    for ng in nganh_list:
        dc = data_chuan[data_chuan["Ngành"]==ng]["DiemChuan"].mean()
        prob = 1/(1+np.exp(-(tong_user-dc)/2))
        st.write(f"- {ng}: {prob*100:.2f}%")

# -------------------------
# Cột phải: biểu đồ điểm chuẩn
# -------------------------
with col2:
    st.subheader("📈 Biến động điểm chuẩn qua các năm")
    plt.figure(figsize=(10,5))
    sns.lineplot(data=data_chuan, x="Năm", y="DiemChuan", hue="Ngành", marker="o")
    plt.axhline(y=tong_user, color='red', linestyle='--', label="Tổng điểm cá nhân")
    plt.text(years[-1]+0.1, tong_user, f"{tong_user:.2f}", color='red')
    plt.ylabel("Điểm chuẩn trung bình")
    plt.xlabel("Năm")
    plt.title("Biến động điểm chuẩn trung bình theo ngành")
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    st.pyplot(plt)

    # -------------------------
    # Biểu đồ phân cụm: xác suất sinh viên theo ngành
    # -------------------------
    st.subheader("📊 Phân bố xác suất trúng tuyển sinh viên mô phỏng")
    plt.figure(figsize=(10,5))
    for ng in nganh_list:
        plt.scatter(df_students["TongDiem"], df_students[ng], alpha=0.5, label=ng, s=20)
    plt.axvline(x=tong_user, color='red', linestyle='--', label="Tổng điểm cá nhân")
    plt.xlabel("Tổng điểm 3 môn")
    plt.ylabel("Xác suất trúng tuyển")
    plt.title("Xác suất trúng tuyển theo tổng điểm sinh viên mô phỏng")
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    st.pyplot(plt)
