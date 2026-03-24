import streamlit as st
import numpy as np
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

MODEL_PATH = "logistic_trungtuyen.pkl"

st.title("🎓 Dự đoán khả năng trúng tuyển đại học")

# Kiểm tra model
if not os.path.exists(MODEL_PATH):
    st.error("❌ Không tìm thấy model. Hãy chạy train_model.py trước!")
    st.stop()

pipe = joblib.load(MODEL_PATH)

# ======================
# INPUT
# ======================
st.header("📥 Nhập điểm")

toan = st.number_input("Điểm Toán", 0, 10, 7)
ly = st.number_input("Điểm Lý", 0, 10, 7)
hoa = st.number_input("Điểm Hóa", 0, 10, 7)
van = st.number_input("Điểm Văn", 0, 10, 7)
anh = st.number_input("Điểm Anh", 0, 10, 7)

# ======================
# PREDICT
# ======================
if st.button("🚀 Dự đoán"):
    new_student = np.array([[toan, ly, hoa, van, anh]])
    prob = pipe.predict_proba(new_student)[0,1]

    st.success(f"🎯 Xác suất trúng tuyển: {prob*100:.2f}%")

# ======================
# BIỂU ĐỒ DEMO
# ======================
st.header("📊 Minh họa dữ liệu")

# tạo lại dữ liệu demo nhỏ (KHÔNG train lại model)
n = 200
np.random.seed(1)

data = pd.DataFrame({
    "TongDiem": np.random.randint(12, 30, n),
    "TrungTuyen": np.random.choice([0,1], n),
    "Nganh": np.random.choice(["CNTT", "KinhTe", "YDuoc", "KyThuat"], n)
})

# Histogram
fig1, ax1 = plt.subplots()
sns.histplot(data, x="TongDiem", hue="TrungTuyen", bins=15, kde=True, ax=ax1)
ax1.set_title("Phân bố tổng điểm")
st.pyplot(fig1)

# Countplot
fig2, ax2 = plt.subplots()
sns.countplot(data=data, x="Nganh", hue="TrungTuyen", ax=ax2)
ax2.set_title("Số lượng trúng tuyển theo ngành")
st.pyplot(fig2)
