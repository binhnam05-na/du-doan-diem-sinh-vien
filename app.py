import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.set_page_config(page_title="Mô hình dự đoán điểm trúng tuyển đại học", layout="wide")
st.title("🎓 Mô hình dự đoán điểm trúng tuyển đại học")

# ======================
# Sidebar: nhập điểm và số lượng sinh viên
# ======================
st.sidebar.header("📥 Thông tin nhập")
n_students = st.sidebar.slider("Số lượng sinh viên mô phỏng", 100, 2000, 500, step=50)
toan = st.sidebar.slider("Điểm Toán", 0.0, 10.0, 0.0, 0.05)
ly = st.sidebar.slider("Điểm Lý", 0.0, 10.0, 0.0, 0.05)
hoa = st.sidebar.slider("Điểm Hóa", 0.0, 10.0, 0.0, 0.05)
van = st.sidebar.slider("Điểm Văn", 0.0, 10.0, 0.0, 0.05)
anh = st.sidebar.slider("Điểm Anh", 0.0, 10.0, 0.0, 0.05)
tong_user = toan + ly + hoa  # ví dụ theo khối A00

# ======================
# Layout 4 cột
# ======================
col1, col2, col3, col4 = st.columns([1,1,1,1])

# ----------------------
# Cột 1: nhập điểm + hiển thị tổng điểm cá nhân
# ----------------------
with col1:
    st.subheader("📥 Điểm cá nhân")
    st.write(f"Tổng điểm (khối A00): {tong_user:.2f}")

# ----------------------
# Cột 2: phân bố tổng điểm sinh viên mô phỏng
# ----------------------
with col2:
    st.subheader("📊 Phân bố tổng điểm sinh viên")
    np.random.seed(42)
    df_students = pd.DataFrame({
        "Toan": np.random.uniform(0,10,n_students),
        "Ly": np.random.uniform(0,10,n_students),
        "Hoa": np.random.uniform(0,10,n_students)
    })
    df_students["TongDiem"] = df_students["Toan"] + df_students["Ly"] + df_students["Hoa"]

    fig, ax = plt.subplots(figsize=(4,3))
    sns.histplot(df_students["TongDiem"], bins=30, kde=True, ax=ax, color="skyblue")
    ax.axvline(tong_user, color='red', linestyle='--', label="Tổng điểm cá nhân")
    ax.set_xlabel("Tổng điểm 3 môn")
    ax.set_ylabel("Số sinh viên")
    ax.set_title("Histogram tổng điểm")
    ax.legend()
    st.pyplot(fig)

# ----------------------
# Cột 3: điểm chuẩn qua các năm
# ----------------------
with col3:
    st.subheader("📈 Điểm chuẩn qua các năm")
    years = np.arange(2018, 2024)
    nganh_list = ["CNTT","Kinh tế","Kỹ thuật","Luật","Mỹ thuật","Ngôn ngữ Anh",
                  "Quản trị kinh doanh","Truyền thông","Tài chính","Y dược"]
    np.random.seed(42)
    data_chuan = pd.DataFrame({
        "Năm": np.repeat(years, len(nganh_list)),
        "Ngành": nganh_list*len(years),
        "DiemChuan": np.round(np.random.uniform(18,26,len(years)*len(nganh_list)),2)
    })

    fig2, ax2 = plt.subplots(figsize=(4,3))
    sns.lineplot(data=data_chuan, x="Năm", y="DiemChuan", hue="Ngành", marker="o", ax=ax2)
    ax2.axhline(tong_user, color='red', linestyle='--', label="Tổng điểm cá nhân")
    ax2.set_ylabel("Điểm chuẩn")
    ax2.set_xlabel("Năm")
    ax2.set_title("Xu hướng điểm chuẩn")
    ax2.legend().set_visible(False)  # giấu legend để gọn
    st.pyplot(fig2)

# ----------------------
# Cột 4: phân cụm sinh viên theo điểm & ngành
# ----------------------
with col4:
    st.subheader("📊 Phân cụm sinh viên")
    df_probs = pd.DataFrame()
    for ng in nganh_list:
        dc = data_chuan[data_chuan["Ngành"]==ng]["DiemChuan"].mean()
        df_probs[ng] = 1/(1+np.exp(-(df_students["TongDiem"]-dc)/2))  # xác suất logistic

    kmeans = KMeans(n_clusters=3, random_state=42)
    df_students["Cluster"] = kmeans.fit_predict(df_probs)

    fig3, ax3 = plt.subplots(figsize=(4,3))
    for c in sorted(df_students["Cluster"].unique()):
        subset = df_students[df_students["Cluster"]==c]
        ax3.scatter(subset["TongDiem"], subset["Cluster"], alpha=0.5, s=20, label=f"Cluster {c}")
    ax3.axvline(tong_user, color='red', linestyle='--', label="Tổng điểm cá nhân")
    ax3.set_xlabel("Tổng điểm 3 môn")
    ax3.set_ylabel("Cluster")
    ax3.set_title("Phân cụm sinh viên theo xác suất trúng tuyển")
    ax3.legend()
    st.pyplot(fig3)
