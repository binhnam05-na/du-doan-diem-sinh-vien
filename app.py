import pandas as pd
import numpy as np

n = 500
np.random.seed(42)
data = pd.DataFrame({
    "DiemToan": np.random.randint(4, 10, n),
    "DiemLy": np.random.randint(4, 10, n),
    "DiemHoa": np.random.randint(4, 10, n),
    "DiemVan": np.random.randint(4, 10, n),
    "DiemAnh": np.random.randint(4, 10, n),
    "Nganh": np.random.choice(["CNTT", "KinhTe", "YDuoc", "KyThuat"], n)
})

diem_chuan = {"CNTT": 24, "KinhTe": 22, "YDuoc": 26, "KyThuat": 23}
data["DiemChuan"] = data["Nganh"].map(diem_chuan)

def tinh_tong(row):
    if row["Nganh"] in ["CNTT", "KyThuat", "YDuoc"]:
        return row["DiemToan"] + row["DiemLy"] + row["DiemHoa"]
    else:
        return row["DiemToan"] + row["DiemVan"] + row["DiemAnh"]

data["TongDiem"] = data.apply(tinh_tong, axis=1)
data["TrungTuyen"] = (data["TongDiem"] >= data["DiemChuan"]).astype(int)

print(data.head())


import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(data, x="TongDiem", hue="TrungTuyen", bins=15, kde=True)
plt.title("Phân bố tổng điểm theo trúng tuyển")
plt.show()

sns.countplot(data=data, x="Nganh", hue="TrungTuyen")
plt.title("Số lượng trúng tuyển theo ngành")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Feature & label
X = data[["DiemToan","DiemLy","DiemHoa","DiemVan","DiemAnh"]]
y = data["TrungTuyen"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline với scaler và LogisticRegression
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])
pipe.fit(X_train, y_train)

# Đánh giá
y_pred = pipe.predict(X_test)
print(classification_report(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.show()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# Tạo dữ liệu giả lập
n = 500
np.random.seed(42)
data = pd.DataFrame({
    "DiemToan": np.random.randint(4, 10, n),
    "DiemLy": np.random.randint(4, 10, n),
    "DiemHoa": np.random.randint(4, 10, n),
    "DiemVan": np.random.randint(4, 10, n),
    "DiemAnh": np.random.randint(4, 10, n),
    "Nganh": np.random.choice(["CNTT", "KinhTe", "YDuoc", "KyThuat"], n)
})

diem_chuan = {"CNTT": 24, "KinhTe": 22, "YDuoc": 26, "KyThuat": 23}
data["DiemChuan"] = data["Nganh"].map(diem_chuan)

def tinh_tong(row):
    if row["Nganh"] in ["CNTT", "KyThuat", "YDuoc"]:
        return row["DiemToan"] + row["DiemLy"] + row["DiemHoa"]
    else:  # KinhTe
        return row["DiemToan"] + row["DiemVan"] + row["DiemAnh"]

data["TongDiem"] = data.apply(tinh_tong, axis=1)
data["TrungTuyen"] = (data["TongDiem"] >= data["DiemChuan"]).astype(int)

# Features & labels
X = data[["DiemToan","DiemLy","DiemHoa","DiemVan","DiemAnh"]]
y = data["TrungTuyen"]

# Train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])
pipe.fit(X_train, y_train)

# Lưu mô hình
joblib.dump(pipe, "logistic_trungtuyen.pkl")
print("Đã lưu pipeline vào logistic_trungtuyen.pkl")


import streamlit as st
import numpy as np
import joblib
import os

MODEL_PATH = "logistic_trungtuyen.pkl"

if not os.path.exists(MODEL_PATH):
    st.error(f"Không tìm thấy file {MODEL_PATH}. Vui lòng chạy train_model.py trước!")
else:
    pipe = joblib.load(MODEL_PATH)

    st.title("Dự đoán khả năng trúng tuyển đại học")

    # Nhập điểm
    toan = st.number_input("Điểm Toán", min_value=0, max_value=10, step=1)
    ly = st.number_input("Điểm Lý", min_value=0, max_value=10, step=1)
    hoa = st.number_input("Điểm Hóa", min_value=0, max_value=10, step=1)
    van = st.number_input("Điểm Văn", min_value=0, max_value=10, step=1)
    anh = st.number_input("Điểm Anh", min_value=0, max_value=10, step=1)

    if st.button("Dự đoán"):
        new_student = np.array([[toan, ly, hoa, van, anh]])
        prob = pipe.predict_proba(new_student)[0,1]
        st.success(f"Xác suất trúng tuyển: {prob*100:.2f}%")