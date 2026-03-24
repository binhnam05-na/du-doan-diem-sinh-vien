import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Mô hình dự đoán điểm trúng tuyển đại học", layout="wide")
st.title("🎓 Mô hình dự đoán điểm trúng tuyển đại học")

# -------------------------
# Sidebar: điểm cá nhân, số lượng sinh viên, chọn khối
# -------------------------
st.sidebar.header("📥 Nhập thông tin")
n_students = st.sidebar.slider("Số lượng sinh viên mô phỏng", 100, 2000, 500, step=50)

# Chọn khối
khoi_options = {
    "A00": ["Toan", "Ly", "Hoa"],
    "A01": ["Toan", "Ly", "Anh"],
    "A02": ["Toan", "Ly", "Sinh"],
    "B00": ["Toan", "Hoa", "Sinh"],
    "B01": ["Toan", "Hoa", "Ly"],
    "C00": ["Van", "Su", "Dia"],
    "C01": ["Van", "Su", "GDCD"],
    "D01": ["Toan", "Van", "Anh"]
}
khoi_selected = st.sidebar.selectbox("Chọn khối/tổ hợp 3 môn", list(khoi_options.keys()))
mon_selected = khoi_options[khoi_selected]

# Nhập điểm từng môn (chỉ các môn trong khối hiện tại)
diem_dict = {}
for mon in set([m for sublist in khoi_options.values() for m in sublist]):
    if mon in mon_selected:
        diem_dict[mon] = st.sidebar.slider(f"Điểm {mon}", 0.0, 10.0, 0.0, 0.05)
    else:
        diem_dict[mon] = 0.0  # các môn khác đặt =0

tong_user = sum([diem_dict[m] for m in mon_selected])

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

# Cho phép điều chỉnh điểm chuẩn qua các năm
st.sidebar.subheader("Điều chỉnh điểm chuẩn trung bình")
dc_adjust = st.sidebar.slider("Điểm chuẩn tăng/giảm", -5.0, 5.0, 0.0, 0.1)
data_chuan["DiemChuanAdj"] = data_chuan["DiemChuan"] + dc_adjust

# -------------------------
# Mô phỏng sinh viên
# -------------------------
np.random.seed(42)
# Sinh viên ngẫu nhiên cho tất cả môn (các môn không dùng = 0)
df_students = pd.DataFrame()
for mon in diem_dict.keys():
    df_students[mon] = np.random.uniform(0,10,n_students)

df_students["TongDiem"] = df_students[mon_selected].sum(axis=0)

# Giả lập xác suất trúng tuyển theo ngành
probs = []
for ng in nganh_list:
    dc = data_chuan[data_chuan["Ngành"]==ng]["DiemChuanAdj"].mean()
    prob = 1/(1+np.exp(-(df_students["TongDiem"]-dc)/2))
    probs.append(prob)
df_probs = pd.DataFrame(np.array(probs).T, columns=nganh_list)
df_students = pd.concat([df_students, df_probs], axis=1)

# -------------------------
# Layout 2 cột chính
# -------------------------
col1, col2 = st.columns([1,3])

# -------------------------
# Cột trái: thông tin cá nhân
# -------------------------
with col1:
    st.subheader("📥 Điểm cá nhân")
    for mon in mon_selected:
        st.write(f"- {mon}: {diem_dict[mon]:.2f}")
    st.write(f"**Tổng điểm 3 môn ({khoi_selected}): {tong_user:.2f}**")
    
    st.subheader("📊 Xác suất trúng tuyển dự kiến")
    for ng in nganh_list:
        dc = data_chuan[data_chuan["Ngành"]==ng]["DiemChuanAdj"].mean()
        prob = 1/(1+np.exp(-(tong_user-dc)/2))
        st.write(f"- {ng}: {prob*100:.2f}%")

# -------------------------
# Cột phải: biểu đồ
# -------------------------
with col2:
    # Biểu đồ điểm chuẩn qua các năm
    st.subheader("📈 Biến động điểm chuẩn theo ngành")
    fig, ax = plt.subplots(figsize=(10,4))
    sns.lineplot(data=data_chuan, x="Năm", y="DiemChuanAdj", hue="Ngành", marker="o", ax=ax)
    ax.axhline(y=tong_user, color='red', linestyle='--', label="Tổng điểm cá nhân")
    ax.text(years[-1]+0.1, tong_user, f"{tong_user:.2f}", color='red')
    ax.set_ylabel("Điểm chuẩn trung bình")
    ax.set_xlabel("Năm")
    ax.set_title(f"Điểm chuẩn trung bình theo ngành (Khối {khoi_selected})")
    ax.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")  # khoảng cách giữa các biểu đồ

    # Biểu đồ phân bố sinh viên và xác suất trúng tuyển
    st.subheader("📊 Phân bố xác suất trúng tuyển sinh viên")
    fig2, ax2 = plt.subplots(figsize=(10,4))
    for ng in nganh_list:
        ax2.scatter(df_students["TongDiem"], df_students[ng], alpha=0.5, label=ng, s=20)
    ax2.axvline(x=tong_user, color='red', linestyle='--', label="Tổng điểm cá nhân")
    ax2.set_xlabel(f"Tổng điểm 3 môn ({khoi_selected})")
    ax2.set_ylabel("Xác suất trúng tuyển")
    ax2.set_title("Xác suất trúng tuyển theo tổng điểm sinh viên mô phỏng")
    ax2.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    st.pyplot(fig2)
