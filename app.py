import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dự đoán trúng tuyển đại học", layout="wide")
st.title("🎓 Mô phỏng trúng tuyển đại học (0–30 điểm, nhiều điểm lẻ)")

# =======================
# Khai báo ngành & điểm chuẩn
# =======================
nganh_list = ["CNTT","KinhTe","YDuoc","KyThuat"]
diem_chuan = {"CNTT":24, "KinhTe":22, "YDuoc":26, "KyThuat":23}

# =======================
# Khối thi (tổ hợp môn)
# =======================
khoi_dict = {
    "A00": ["Toan","Ly","Hoa"],
    "A01": ["Toan","Ly","Anh"],
    "C00": ["Van","Su","Dia"]
}

# =======================
# Tạo dữ liệu giả lập 500 sinh viên
# =======================
n_students = 500
np.random.seed(42)
data = pd.DataFrame({
    "Toan": np.round(np.random.uniform(0,10,n_students),2),
    "Ly": np.round(np.random.uniform(0,10,n_students),2),
    "Hoa": np.round(np.random.uniform(0,10,n_students),2),
    "Van": np.round(np.random.uniform(0,10,n_students),2),
    "Anh": np.round(np.random.uniform(0,10,n_students),2),
    "Su": np.round(np.random.uniform(0,10,n_students),2),
    "Dia": np.round(np.random.uniform(0,10,n_students),2),
    "Nganh": np.random.choice(nganh_list, n_students)
})

# =======================
# Tạo cột tổng điểm theo khối
# =======================
for khoi, mon_list in khoi_dict.items():
    data[f"Tong_{khoi}"] = data[mon_list].sum(axis=1) * 3 / len(mon_list)  # chuẩn hóa về 0-30

# =======================
# Tạo nhãn trúng tuyển từng ngành dựa trên A00
# =======================
data["TrungTuyen"] = (data["Tong_A00"] >= data["Nganh"].map(diem_chuan)).astype(int)

# =======================
# Train Logistic Regression dự đoán trúng tuyển
# =======================
features = ["Toan","Ly","Hoa","Van","Anh","Su","Dia"]
X = data[features]
y = data["TrungTuyen"]

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])
pipe.fit(X, y)

# =======================
# Layout 4 cột
# =======================
col1, col2, col3, col4 = st.columns(4)

# =======================
# Cột 1: Slider nhập điểm cá nhân
# =======================
with col1:
    st.header("📥 Nhập điểm cá nhân")
    khoi_chon = st.selectbox("Chọn khối", list(khoi_dict.keys()))
    mon1, mon2, mon3 = khoi_dict[khoi_chon]
    
    d1 = st.slider(f"{mon1}", 0.0, 10.0, 0.0, 0.05)
    d2 = st.slider(f"{mon2}", 0.0, 10.0, 0.0, 0.05)
    d3 = st.slider(f"{mon3}", 0.0, 10.0, 0.0, 0.05)

# =======================
# Cột 2: Tính tổng điểm và xác suất trúng tuyển từng ngành
# =======================
tong_user = []
prob_user = []

for ng in nganh_list:
    # chọn tổng điểm theo khối A00 cho tính toán ví dụ
    if ng in ["CNTT","KyThuat","YDuoc"]:
        tong = d1 + d2 + d3  # theo khối bạn chọn
    else:
        tong = d1 + d2 + d3  # đơn giản hiện tại
    tong_user.append(tong)

    # chuẩn bị input cho model
    X_input = np.zeros((1, len(features)))
    for idx, f in enumerate(features):
        if f in khoi_dict[khoi_chon]:
            X_input[0, idx] = [d1,d2,d3][khoi_dict[khoi_chon].index(f)]
        else:
            X_input[0, idx] = 0.0  # môn khác = 0
    try:
        prob = pipe.predict_proba(X_input)[0,1]
    except:
        prob = 0.0
    prob_user.append(prob)

df_user = pd.DataFrame({
    "Nganh": nganh_list,
    "TongDiem": tong_user,
    "XacSuat": prob_user,
    "DiemChuan": [diem_chuan[ng] for ng in nganh_list]
})

# =======================
# Hiển thị xác suất
# =======================
with col1:
    st.subheader("🚀 Xác suất trúng tuyển")
    for i, ng in enumerate(nganh_list):
        st.write(f"- {ng}: Tổng điểm = {tong_user[i]:.2f}, "
                 f"Điểm chuẩn = {diem_chuan[ng]}, "
                 f"Xác suất = {prob_user[i]*100:.2f}%")

# =======================
# Biểu đồ tổng điểm (0-30)
# =======================
with col2:
    st.subheader("📊 Tổng điểm theo ngành")
    fig, ax = plt.subplots(figsize=(4,3))
    sns.barplot(data=df_user, x="Nganh", y="TongDiem", palette="viridis", ax=ax)
    for i, row in df_user.iterrows():
        ax.axhline(row["DiemChuan"], color='red', linestyle='--')
        ax.text(i, row["DiemChuan"]+0.2, f"DC={row['DiemChuan']}", color='red', ha='center')
    ax.set_ylim(0,30)  # cố định trục y
    ax.set_ylabel("Tổng điểm")
    st.pyplot(fig)

# =======================
# Biểu đồ xác suất
# =======================
with col3:
    st.subheader("📊 Xác suất trúng tuyển")
    fig2, ax2 = plt.subplots(figsize=(4,3))
    sns.barplot(data=df_user, x="Nganh", y="XacSuat", palette="coolwarm", ax=ax2)
    for i, row in df_user.iterrows():
        ax2.text(i, row["XacSuat"]+0.02, f"{row['XacSuat']*100:.1f}%", ha='center')
    ax2.set_ylim(0,1)
    ax2.set_ylabel("Xác suất")
    st.pyplot(fig2)

# =======================
# Cột 4: Scatter plot phân bố sinh viên
# =======================
with col4:
    st.subheader("📊 Phân bố sinh viên")
    fig3, ax3 = plt.subplots(figsize=(4,3))
    # KMeans 3 nhóm theo tổng điểm A00
    kmeans = KMeans(n_clusters=3, random_state=42)
    data["Cluster"] = kmeans.fit_predict(data["Tong_A00"].values.reshape(-1,1))
    scatter = ax3.scatter(data["Tong_A00"], np.arange(len(data)), c=data["Cluster"], cmap="Set2")
    ax3.set_xlabel("Tổng điểm A00")
    ax3.set_ylabel("Sinh viên")
    st.pyplot(fig3)
    st.write("Màu theo phân cụm tổng điểm sinh viên (0–30)")
