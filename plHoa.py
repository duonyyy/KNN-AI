# Import các thư viện cần thiết
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# 1. Tải dữ liệu Iris
iris = load_iris()
X = iris.data  # Đặc trưng (features): chiều dài, chiều rộng đài hoa và cánh hoa
y = iris.target  # Nhãn (labels): 3 loại hoa Iris

# 2. Chia dữ liệu thành tập train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Khởi tạo và huấn luyện mô hình Logistic Regression
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 4. Dự đoán trên tập test
y_pred = model.predict(X_test)

# 5. Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
print(f"Độ chính xác của mô hình: {accuracy * 100:.2f}%")
print("\nBáo cáo phân loại chi tiết:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 6. Dự đoán một mẫu mới (ví dụ)
sample = np.array([[5.0, 3.4, 1.5, 0.2]])  # Một bông hoa với các đặc trưng
prediction = model.predict(sample)
print(f"Dự đoán loại hoa cho mẫu mới: {iris.target_names[prediction[0]]}")