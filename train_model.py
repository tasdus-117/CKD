import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Đọc dữ liệu
print(">> Đang tải dữ liệu...")
try:
    # Ưu tiên đọc file local nếu có
    df = pd.read_csv('kidney_disease.csv', na_values=['?', ' ', '\t'])
except FileNotFoundError:
    print("Không tìm thấy file local, vui lòng tải file kidney_disease.csv về máy.")
    exit()

# 2. Xử lý sơ bộ Target và Cột ID
if 'id' in df.columns:
    df = df.drop('id', axis=1)

# Xử lý biến mục tiêu (classification)
TARGET_COLUMN = 'classification'
df = df.dropna(subset=[TARGET_COLUMN])

# Làm sạch nhãn (y)
df[TARGET_COLUMN] = df[TARGET_COLUMN].str.strip().replace('ckd\t', 'ckd')
# Chuyển đổi nhãn: ckd -> 1, notckd -> 0
label_map = {'ckd': 1, 'notckd': 0}
df[TARGET_COLUMN] = df[TARGET_COLUMN].map(label_map)

# Kiểm tra lại nếu còn giá trị lạ chưa map được
df = df.dropna(subset=[TARGET_COLUMN])

X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN].astype(int)

# 3. Định nghĩa các nhóm cột (như code của bạn)
numerical_cols = [
    'age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc'
]

categorical_cols = [
    'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad',
    'appet', 'pe', 'ane'
]

# Chuyển đổi kiểu dữ liệu cho đúng chuẩn pipeline
for col in numerical_cols:
    X[col] = pd.to_numeric(X[col], errors='coerce')

for col in categorical_cols:
    X[col] = X[col].astype(str) # Đưa về string để OneHotEncoder xử lý

# 4. Xây dựng Pipeline Tiền xử lý (Code của bạn)
# Pipeline xử lý cột SỐ
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline xử lý cột CATEGORY
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # sparse_output=False để dễ debug nếu cần
])

# Kết hợp bằng ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ],
    remainder='passthrough'
)

# 5. Tạo Pipeline TỔNG (Bao gồm cả Model)
# Đây là bước quan trọng: Nhét cả Preprocessor và Model vào chung 1 ống
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# 6. Chia tập dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 7. Huấn luyện Pipeline
print(">> Đang huấn luyện Pipeline (Tiền xử lý + Model)...")
full_pipeline.fit(X_train, y_train)

# 8. Đánh giá
y_pred = full_pipeline.predict(X_test)
print("\n=== KẾT QUẢ HUẤN LUYỆN ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 9. Lưu toàn bộ Pipeline
# Khi lưu cái này, ta lưu luôn cả logic Scaler, Imputer, OneHot và Model
joblib.dump(full_pipeline, 'ckd_pipeline.pkl')
print("\n>> Đã lưu file 'ckd_pipeline.pkl'.")