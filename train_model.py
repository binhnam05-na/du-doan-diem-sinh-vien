import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# Tạo dữ liệu giả
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

# Train model
X = data[["DiemToan","DiemLy","DiemHoa","DiemVan","DiemAnh"]]
y = data["TrungTuyen"]

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

pipe.fit(X, y)

# Lưu model
joblib.dump(pipe, "logistic_trungtuyen.pkl")
print("✅ Đã lưu model!")