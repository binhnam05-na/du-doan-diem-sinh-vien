import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

st.set_page_config(page_title="Mô hình dự đoán điểm trúng tuyển đại học", layout="wide")
st.title("🎓 Mô hình dự đoán điểm trúng tuyển đại học")

# -------------------------
# Sidebar: điểm cá nhân & số lượng sinh viên
# -------------------------
st.sidebar.header("📥 Nhập thông tin")
n_students = st.sidebar.slider("Số lượng sinh viên mô phỏng", 100, 2000, 500, step=50)
toan = st.sidebar.slider("Điểm Toán", 0.0, 10.0, 0.0, 0.05)
ly = st.sidebar.slider("Điểm Lý", 0.0, 10.0, 0.0, 0.05)
hoa = st.sidebar.slider("Điểm Hóa", 0.0, 10.0, 0.0, 0.05)
tong_user = toan + ly + hoa

# -------------------------
# 1️⃣ Phân bố điểm: histogram & boxplot
# -------------------------
st.header("1️⃣ Phân bố điểm sinh viên mô phỏng")
np.random.seed(42)
df_students = pd.DataFrame({
    "Toan": np.random.uniform(0,10,n_students),
    "Ly": np.random.uniform(0,10,n_students),
    "Hoa": np.random.uniform(0,10,n_students)
})
df_students["TongDiem"] = df_students["Toan"] + df_students["Ly"] + df_students["Hoa"]

# Histogram
fig_hist, ax_hist = plt.subplots(figsize=(8,3))
sns.histplot(df_students["TongDiem"], bins=30, kde=True, ax=ax_hist, color="skyblue")
ax_hist.axvline(tong_user, color='red', linestyle='--', label="Tổng điểm cá nhân")
ax_hist.set_xlabel("Tổng điểm 3 môn")
ax_hist.set_ylabel("Số sinh viên")
ax_hist.set_title("Histogram tổng điểm sinh viên mô phỏng")
ax_hist.legend()
st.pyplot(fig_hist)

# Boxplot
fig_box, ax_box = plt.subplots(figsize=(8,2))
sns.boxplot(x=df_students["TongDiem"], ax=ax_box, color="lightgreen")
ax_box.axvline(tong_user, color='red', linestyle='--', label="Tổng điểm cá nhân")
ax_box.set_xlabel("Tổng điểm 3 môn")
ax_box.set_title("Boxplot tổng điểm sinh viên mô phỏng")
ax_box.legend()
st.pyplot(fig_box)

st.markdown("---")  # khoảng cách

# -------------------------
# 2️⃣ Xu hướng theo năm: line chart điểm chuẩn
# -------------------------
st.header("2️⃣ Xu hướng điểm chuẩn trung bình qua các năm")
years = np.arange(2018, 2024)
nganh_list = ["CNTT","Kinh tế","Kỹ thuật","Luật","Mỹ thuật","Ngôn ngữ Anh",
              "Quản trị kinh doanh","Truyền thông","Tài chính","Y dược"]
np.random.seed(42)
data_chuan = pd.DataFrame({
    "Năm": np.repeat(years, len(nganh_list)),
    "Ngành": nganh_list*len(years),
    "DiemChuan": np.round(np.random.uniform(18,26,len(years)*len(nganh_list)),2)
})

fig_line, ax_line = plt.subplots(figsize=(10,4))
sns.lineplot(data=data_chuan, x="Năm", y="DiemChuan", hue="Ngành", marker="o", ax=ax_line)
ax_line.axhline(y=tong_user, color='red', linestyle='--', label="Tổng điểm cá nhân")
ax_line.text(years[-1]+0.1, tong_user, f"{tong_user:.2f}", color='red')
ax_line.set_ylabel("Điểm chuẩn trung bình")
ax_line.set_xlabel("Năm")
ax_line.set_title("Điểm chuẩn trung bình theo ngành")
ax_line.legend(bbox_to_anchor=(1.05,1), loc='upper left')
plt.tight_layout()
st.pyplot(fig_line)

st.markdown("---")  # khoảng cách

# -------------------------
# 3️⃣ Dự đoán trúng tuyển: k-means clustering
# -------------------------
st.header("3️⃣ Dự đoán trúng tuyển: phân cụm sinh viên theo xác suất")
# Giả lập xác suất trúng tuyển theo ngành
df_probs = pd.DataFrame()
for ng in nganh_list:
    dc = data_chuan[data_chuan["Ngành"]==ng]["DiemChuan"].mean()
    df_probs[ng] = 1/(1+np.exp(-(df_students["TongDiem"]-dc)/2))

# KMeans phân cụm
kmeans = KMeans(n_clusters=3, random_state=42)
df_students["Cluster"] = kmeans.fit_predict(df_probs)

# Biểu đồ scatter xác suất theo cluster
fig_cluster, ax_cluster = plt.subplots(figsize=(10,4))
for c in sorted(df_students["Cluster"].unique()):
    subset = df_students[df_students["Cluster"]==c]
    ax_cluster.scatter(subset["TongDiem"], subset["Cluster"], alpha=0.5, s=20, label=f"Cluster {c}")
ax_cluster.axvline(x=tong_user, color='red', linestyle='--', label="Tổng điểm cá nhân")
ax_cluster.set_xlabel("Tổng điểm 3 môn")
ax_cluster.set_ylabel("Cluster")
ax_cluster.set_title("Phân cụm sinh viên theo xác suất trúng tuyển")
ax_cluster.legend()
plt.tight_layout()
st.pyplot(fig_cluster)
