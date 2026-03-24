import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# =======================
# Config trang
# =======================
st.set_page_config(page_title="Dự đoán trúng tuyển", layout="wide")
st.title("🎓 Dự đoán khả năng trúng tuyển đại học")

MODEL_PATH = "logistic_trungtuyen.pkl"

# =======================
# Nếu chưa có model → train
# =======================
if not os.path.exists(MODEL_PATH):
    st.info("🔄 Chưa có model, đang train model...")
    
    # Tạo dữ liệu float 0-10
    n = 500
    np.random.seed(42)
    data = pd.DataFrame({
        "DiemToan": np.round(np.random.uniform(0,10,n),2),
        "DiemLy": np.round(np.random.uniform(0,10,n),2),
        "DiemHoa": np.round(np.random.uniform(0,10,n),2),
        "DiemVan": np.round(np.random.uniform(0,10,n),2),
        "DiemAnh": np.round(np.random.uniform(0,10,n),2),
        "Nganh": np.random.choice(["CNTT","KinhTe","YDuoc","KyThuat"], n)
    })

    diem_chuan = {"CNTT":24, "KinhTe":22, "YDuoc":26, "KyThuat":23}
    data["DiemChuan"] = data["Nganh"].map(diem_chuan)

    # Tính tổng điểm theo ngành
    def tinh_tong(row):
        if row["Nganh"] in ["CNTT","KyThuat","YDuoc"]:
            return row["DiemToan"] + row["DiemLy"] + row["DiemHoa"]
        else:
            return row["DiemToan"] + row["DiemVan"] + row["DiemAnh"]

    data["TongDiem"] = data.apply(tinh_tong, axis=1)
    data["TrungTuyen"] = (data["TongDiem"] >= data["DiemChuan"]).astype(int)

    # Features & labels
    X = data[["DiemToan","DiemLy","DiemHoa","DiemVan","DiemAnh"]]
    y = data["TrungTuyen"]

    # Train pipeline
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression())
    ])
    pipe.fit(X, y)

    # Lưu model
    joblib.dump(pipe, MODEL_PATH)
    st.success("✅ Đã train xong model!")

else:
    pipe = joblib.load(MODEL_PATH)

# =======================
# Layout 4 cột
# =======================
col1, col2, col3, col4 = st.columns(4)

# =======================
# Slider input
# =======================
with col1:
    st.header("📥 Nhập điểm")
    toan = st.slider("Điểm Toán", 0.0, 10.0, 0.0, 0.25)
    ly = st.slider("Điểm Lý", 0.0, 10.0, 0.0, 0.25)
    hoa = st.slider("Điểm Hóa", 0.0, 10.0, 0.0, 0.25)
    van = st.slider("Điểm Văn", 0.0, 10.0, 0.0, 0.25)
    anh = st.slider("Điểm Anh", 0.0, 10.0, 0.0, 0.25)

# =======================
# Tính tổng điểm & xác suất
# =======================
nganh_list = ["CNTT","KinhTe","YDuoc","KyThuat"]
diem_chuan = {"CNTT":24, "KinhTe":22, "YDuoc":26, "KyThuat":23}

tong_diem_user = []
prob_user = []

for ng in nganh_list:
    if ng in ["CNTT","KyThuat","YDuoc"]:
        tong = toan + ly + hoa
    else:
        tong = toan + van + anh
    tong_diem_user.append(tong)

    # Predict xác suất
    X_input = np.array([[toan, ly, hoa, van, anh]], dtype=np.float64)
    try:
        prob = pipe.predict_proba(X_input)[0,1]
    except Exception as e:
        prob = 0.0
    prob_user.append(prob)

df_user = pd.DataFrame({
    "Nganh": nganh_list,
    "TongDiem": tong_diem_user,
    "XacSuat": prob_user,
    "DiemChuan": [diem_chuan[ng] for ng in nganh_list]
})

# =======================
# Hiển thị xác suất
# =======================
with col1:
    st.subheader("🚀 Xác suất trúng tuyển")
    for i, ng in enumerate(nganh_list):
        st.write(f"- {ng}: Tổng điểm = {tong_diem_user[i]:.2f}, "
                 f"Điểm chuẩn = {diem_chuan[ng]}, "
                 f"Xác suất = {prob_user[i]*100:.2f}%")

# =======================
# Biểu đồ tổng điểm
# =======================
with col2:
    st.subheader("📊 Tổng điểm theo ngành")
    fig, ax = plt.subplots(figsize=(4,3))
    sns.barplot(data=df_user, x="Nganh", y="TongDiem", palette="viridis", ax=ax)
    for i, row in df_user.iterrows():
        ax.axhline(row["DiemChuan"], color='red', linestyle='--')
        ax.text(i, row["DiemChuan"]+0.2, f"DC={row['DiemChuan']}", color='red', ha='center')
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
    ax2.set_ylabel("Xác suất")
    st.pyplot(fig2)

# =======================
# Cột thông tin
# =======================
with col4:
    st.subheader("ℹ️ Thông tin")
    st.write("Kéo các slider để thay đổi điểm và xem ảnh hưởng đến khả năng trúng tuyển theo từng ngành.")
