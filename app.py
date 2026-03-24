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
# INPUT - Slider cho từng môn
# ======================
st.header("📥 Nhập điểm từng môn")

toan = st.slider("Điểm Toán", 0, 10, 7)
ly = st.slider("Điểm Lý", 0, 10, 7)
hoa = st.slider("Điểm Hóa", 0, 10, 7)
van = st.slider("Điểm Văn", 0, 10, 7)
anh = st.slider("Điểm Anh", 0, 10, 7)

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
# Hiển thị xác suất trúng tuyển
# ======================
st.subheader("🚀 Xác suất trúng tuyển theo ngành")
for i, ng in enumerate(nganh_list):
    st.write(f"- {ng}: Tổng điểm = {tong_diem_user[i]}, "
             f"Điểm chuẩn = {diem_chuan[ng]}, "
             f"Xác suất trúng tuyển = {prob_user[i]*100:.2f}%")

# ======================
# BIỂU ĐỒ
# ======================
st.subheader("📊 Minh họa tổng điểm theo ngành")

fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(data=df_user, x="Nganh", y="TongDiem", palette="viridis", ax=ax)
for i, row in df_user.iterrows():
    ax.axhline(row["DiemChuan"], color='red', linestyle='--')
    ax.text(i, row["DiemChuan"]+0.2, f"DC={row['DiemChuan']}", color='red', ha='center')
ax.set_ylabel("Tổng điểm")
ax.set_title("Tổng điểm của bạn và điểm chuẩn từng ngành")
st.pyplot(fig)

st.subheader("📊 Xác suất trúng tuyển theo ngành")
fig2, ax2 = plt.subplots(figsize=(8,5))
sns.barplot(data=df_user, x="Nganh", y="XacSuat", palette="coolwarm", ax=ax2)
for i, row in df_user.iterrows():
    ax2.text(i, row["XacSuat"]+0.02, f"{row['XacSuat']*100:.1f}%", ha='center')
ax2.set_ylabel("Xác suất trúng tuyển")
ax2.set_title("Khả năng trúng tuyển theo ngành")
st.pyplot(fig2)
