import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Mô phỏng trúng tuyển", layout="wide")
st.title("🎓 Mô phỏng trúng tuyển đại học (0–30 điểm, nhiều điểm lẻ)")

# ======================
# Tùy chỉnh số sinh viên
# ======================
n_students = st.sidebar.slider("Số lượng sinh viên mô phỏng", 100, 2000, 500, step=50)

# ======================
# Khai báo ngành & điểm chuẩn
# ======================
nganh_list = ["CNTT","KinhTe","YDuoc","KyThuat"]
diem_chuan = {"CNTT":24, "KinhTe":22, "YDuoc":26, "KyThuat":23}

# ======================
# Tổ hợp môn
# ======================
khoi_dict = {
    "A00": ["Toan","Ly","Hoa"],
    "A01": ["Toan","Ly","Anh"],
    "A02": ["Toan","Ly","Sinh"],
    "A03": ["Toan","Ly","GDCD"],
    "B00": ["Toan","Hoa","Sinh"],
    "B01": ["Toan","Hoa","Ly"],
    "B02": ["Toan","Hoa","Anh"],
    "B03": ["Toan","Hoa","GDCD"],
    "C00": ["Van","Su","Dia"],
    "C01": ["Van","Su","GDCD"],
    "C02": ["Van","Su","Anh"],
    "C03": ["Van","Su","Toan"],
    "D01": ["Toan","Van","Anh"]
}

# ======================
# Sinh dữ liệu sinh viên giả lập
# ======================
np.random.seed(42)
môn_list = list({m for sublist in khoi_dict.values() for m in sublist})
data = pd.DataFrame({mon: np.round(np.random.uniform(0,10,n_students),2) for mon in môn_list})
data["Nganh"] = np.random.choice(nganh_list, n_students)

# ======================
# Tạo tổng điểm theo khối
# ======================
for khoi, mon_list in khoi_dict.items():
    data[f"Tong_{khoi}"] = data[mon_list].sum(axis=1) * 3 / len(mon_list)  # chuẩn hóa 0-30

# ======================
# Tạo nhãn trúng tuyển theo A00
# ======================
data["TrungTuyen"] = (data["Tong_A00"] >= data["Nganh"].map(diem_chuan)).astype(int)

# ======================
# Train logistic regression
# ======================
features = môn_list
X = data[features]
y = data["TrungTuyen"]

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])
pipe.fit(X, y)

# ======================
# Layout
# ======================
col1, col2, col3 = st.columns([1,1,1])

# ----------------------
# Cột 1: Slider điểm cá nhân
# ----------------------
with col1:
    st.header("📥 Nhập điểm cá nhân")
    khoi_chon = st.selectbox("Chọn khối", list(khoi_dict.keys()))
    mon1, mon2, mon3 = khoi_dict[khoi_chon]
    d1 = st.slider(mon1, 0.0, 10.0, 0.0, 0.05)
    d2 = st.slider(mon2, 0.0, 10.0, 0.0, 0.05)
    d3 = st.slider(mon3, 0.0, 10.0, 0.0, 0.05)

# ----------------------
# Cột 2: Biểu đồ tổng điểm
# ----------------------
tong_user = []
prob_user = []

for ng in nganh_list:
    tong = d1 + d2 + d3
    tong_user.append(tong)
    
    X_input = np.zeros((1,len(features)))
    for idx, f in enumerate(features):
        if f in khoi_dict[khoi_chon]:
            X_input[0, idx] = [d1,d2,d3][khoi_dict[khoi_chon].index(f)]
    prob = pipe.predict_proba(X_input)[0,1]
    prob_user.append(prob)

df_user = pd.DataFrame({
    "Nganh": nganh_list,
    "TongDiem": tong_user,
    "XacSuat": prob_user,
    "DiemChuan": [diem_chuan[ng] for ng in nganh_list]
})

with col2:
    st.subheader("📊 Tổng điểm & điểm chuẩn")
    fig, ax = plt.subplots(figsize=(4,3))
    sns.barplot(data=df_user, x="Nganh", y="TongDiem", palette="viridis", ax=ax)
    for i, row in df_user.iterrows():
        ax.axhline(row["DiemChuan"], color='red', linestyle='--')
        ax.text(i, row["DiemChuan"]+0.2, f"DC={row['DiemChuan']}", color='red', ha='center')
    ax.set_ylim(0,30)
    st.pyplot(fig)

# ----------------------
# Cột 3: Biểu đồ xác suất & hồi quy
# ----------------------
with col3:
    st.subheader("📈 Xác suất trúng tuyển (đường hồi quy)")
    fig2, ax2 = plt.subplots(figsize=(4,3))
    # đường hồi quy logistic
    x_line = np.linspace(0,30,100)
    for i, ng in enumerate(nganh_list):
        y_line = 1/(1 + np.exp(-(x_line - diem_chuan[ng])))
        ax2.plot(x_line, y_line, label=f"{ng}")
        # điểm người dùng
        ax2.scatter(tong_user[i], prob_user[i], color='red', zorder=5)
    ax2.set_xlabel("Tổng điểm 3 môn")
    ax2.set_ylabel("Xác suất")
    ax2.set_ylim(0,1)
    ax2.legend()
    st.pyplot(fig2)

# ----------------------
# Phân bố sinh viên
# ----------------------
st.subheader("📊 Phân bố sinh viên theo ngành & tổng điểm")
fig3, ax3 = plt.subplots(figsize=(10,4))
sns.scatterplot(data=data, x="Tong_A00", y=np.random.uniform(0,n_students,n_students), hue="Nganh", palette="Set2", ax=ax3)
ax3.set_xlabel("Tổng điểm 3 môn A00")
ax3.set_ylabel("Sinh viên (ngẫu nhiên)")
st.pyplot(fig3)
