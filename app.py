import streamlit as st
import numpy as np
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

MODEL_PATH = "logistic_trungtuyen.pkl"
st.set_page_config(page_title="Dự đoán trúng tuyển", layout="wide")

st.title("🎓 Dự đoán khả năng trúng tuyển đại học")

# Kiểm tra model
if not os.path.exists(MODEL_PATH):
    st.error("❌ Không tìm thấy model. Hãy chạy train_model.py trước!")
    st.stop()

pipe = joblib.load(MODEL_PATH)

# ======================
# Tạo layout 4 cột
# ======================
col1, col2, col3, col4 = st.columns(4)

# ======================
# INPUT - Slider cho từng môn ở cột 1
# ======================
with col1:
    st.header("📥 Nhập điểm từng môn")
    toan = st.slider("Điểm Toán", 0.0, 10.0, 0.0, 0.25)
    ly = st.slider("Điểm Lý", 0.0, 10.0, 0.0, 0.25)
    hoa = st.slider("Điểm Hóa", 0.0, 10.0, 0.0, 0.25)
    van = st.slider("Điểm Văn", 0.0, 10.0, 0.0, 0.25)
    anh = st.slider("Điểm Anh", 0.0, 10.0, 0.0, 0.25)

# ======================
# TÍNH TỔNG ĐIỂM VÀ XÁC SUẤT
# ======================
nganh_list = ["CNTT", "KinhTe", "YDuoc", "KyThuat"]
diem_chuan = {"CNTT": 24, "KinhTe": 22, "YDuoc": 26, "KyThuat": 23}

tong_diem_user = []
prob_user = []

for ng in nganh_list:
    if ng in ["CNTT","KyThuat","YDuoc"]:
        tong = toan + ly + hoa
    else:
        tong = toan + van + anh
    tong_diem_user.append(tong)
    
    X = np.array([[toan, ly, hoa, van, anh]])
    prob = pipe.predict_proba(X)[0,1]
    prob_user.append(prob)

df_user = pd.DataFrame({
    "Nganh": nganh_list,
    "TongDiem": tong_diem_user,
    "XacSuat": prob_user,
    "DiemChuan": [diem_chuan[ng] for ng in nganh_list]
})

# ======================
# Hiển thị xác suất trúng tuyển ở cột 1
# ======================
with col1:
    st.subheader("🚀 Xác suất trúng tuyển")
    for i, ng in enumerate(nganh_list):
        st.write(f"- {ng}: Tổng điểm = {tong_diem_user[i]:.2f}, "
                 f"Điểm chuẩn = {diem_chuan[ng]}, "
                 f"Xác suất = {prob_user[i]*100:.2f}%")

# ======================
# Biểu đồ tổng điểm ở cột 2
# ======================
with col2:
    st.subheader("📊 Tổng điểm theo ngành")
    fig, ax = plt.subplots(figsize=(4,3))
    sns.barplot(data=df_user, x="Nganh", y="TongDiem", palette="viridis", ax=ax)
    for i, row in df_user.iterrows():
        ax.axhline(row["DiemChuan"], color='red', linestyle='--')
        ax.text(i, row["DiemChuan"]+0.2, f"DC={row['DiemChuan']}", color='red', ha='center')
    ax.set_ylabel("Tổng điểm")
    st.pyplot(fig)

# ======================
# Biểu đồ xác suất ở cột 3
# ======================
with col3:
    st.subheader("📊 Xác suất trúng tuyển")
    fig2, ax2 = plt.subplots(figsize=(4,3))
    sns.barplot(data=df_user, x="Nganh", y="XacSuat", palette="coolwarm", ax=ax2)
    for i, row in df_user.iterrows():
        ax2.text(i, row["XacSuat"]+0.02, f"{row['XacSuat']*100:.1f}%", ha='center')
    ax2.set_ylabel("Xác suất")
    st.pyplot(fig2)

# ======================
# Cột 4 có thể để trống hoặc dùng để thêm info
# ======================
with col4:
    st.subheader("ℹ️ Thông tin")
    st.write("Kéo các slider để thay đổi điểm và xem ảnh hưởng đến khả năng trúng tuyển theo từng ngành.")
