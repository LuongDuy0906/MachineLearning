import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Định nghĩa hàm entropy để tính toán mức độ thông tin của dữ liệu
def entropy(y):
    """
    Tính toán entropy của vector nhãn y.
    
    Args:
        y: vector nhãn đầu ra (target) của tập dữ liệu.
    
    Returns:
        Giá trị entropy của y.
    """
    # Đếm số lượng mẫu của mỗi lớp trong y
    class_counts = np.bincount(y)
    # Loại bỏ các lớp có số lượng bằng 0
    probabilities = class_counts[class_counts > 0] / len(y)
    # Tính entropy theo công thức: - sum(p * log2(p))
    return -np.sum(probabilities * np.log2(probabilities))

def information_gain(X, y, feature_index):
    """
    Tính toán Information Gain của một đặc trưng.
    
    Args:
        X: Tập dữ liệu đầu vào (features).
        y: Tập nhãn đầu ra tương ứng với X.
        feature_index: Chỉ số của đặc trưng trong X mà chúng ta muốn tính Information Gain.
    
    Returns:
        Giá trị Information Gain của đặc trưng tại feature_index.
    """
    # Entropy của toàn bộ tập dữ liệu
    total_entropy = entropy(y)
    
    # Tính entropy có điều kiện cho từng giá trị của đặc trưng tại feature_index
    values, counts = np.unique(X[:, feature_index], return_counts=True)
    weighted_entropy = 0
    
    for value, count in zip(values, counts):
        # Chọn ra các mẫu trong X và y có giá trị đặc trưng bằng `value`
        subset_mask = X[:, feature_index] == value
        subset_y = y[subset_mask]
        
        # Tính entropy cho tập con
        subset_entropy = entropy(subset_y)
        
        # Tính trọng số cho entropy của tập con
        weighted_entropy += (count / len(y)) * subset_entropy
    
    # Information Gain là tổng entropy trừ đi weighted_entropy
    return total_entropy - weighted_entropy

# Hàm đánh giá mô hình
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, precision, recall, f1, report

def grid_search_cv(model, params, X_train, y_train, scoring='accuracy', cv=5):
    # Khởi tạo đối tượng GridSearchCV
    grid_search = GridSearchCV(model, params, cv=cv, scoring=scoring, n_jobs=-1)

    # Thực hiện tìm kiếm lưới với tập huấn luyện
    grid_search.fit(X_train, y_train)

    # Lấy mô hình tốt nhất và thông số tương ứng
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    return best_model, best_params, best_score

def evaluate_errors(model, X_train, y_train, X_test, y_test, cv=5):
    """
    Hàm tính toán Training Error, Validation Error và Test Error cho mô hình.

    Args:
        model: Mô hình cần đánh giá.
        X_train: Dữ liệu huấn luyện đầu vào.
        y_train: Nhãn huấn luyện.
        X_test: Dữ liệu kiểm tra đầu vào.
        y_test: Nhãn kiểm tra.
        cv: Số lần chia cross-validation (mặc định là 5).

    Returns:
        train_error: Lỗi trên tập huấn luyện.
        validation_error: Lỗi trung bình của cross-validation.
        test_error: Lỗi trên tập kiểm tra.
    """
    # Dự đoán trên tập huấn luyện
    y_train_pred = model.predict(X_train)
    train_error = 1 - accuracy_score(y_train, y_train_pred)

    # Sử dụng cross-validation để tính validation error
    validation_error = 1 - cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy').mean()

    # Dự đoán trên tập kiểm tra
    y_test_pred = model.predict(X_test)
    test_error = 1 - accuracy_score(y_test, y_test_pred)

    return train_error, validation_error, test_error

# Hàm khởi tạo và huấn luyện mô hình Perceptron
def train_perceptron(X_train, y_train, X_test, y_test):
    model = Perceptron(random_state=42)
    model.fit(X_train, y_train)
    return model

# Hàm khởi tạo và huấn luyện mô hình Decision Tree
def train_decision_tree(X_train, y_train, X_test, y_test):
    model = DecisionTreeClassifier(random_state=42, max_depth=3, min_samples_leaf=5)
    model.fit(X_train, y_train)
    return model

# Hàm khởi tạo và huấn luyện mô hình Neural Network
def train_neural_network(X_train, y_train, X_test, y_test):
    model = MLPClassifier(hidden_layer_sizes=(5, 5), max_iter=2000, activation="relu", random_state=42)
    model.fit(X_train, y_train)
    return model

# Hàm khởi tạo và huấn luyện mô hình Stacking Classifier
def train_stacking_classifier(X_train, y_train, X_test, y_test):
    base_models = [('perceptron', Perceptron(random_state=42)),
                   ('decision_tree', DecisionTreeClassifier(random_state=42, max_depth=3, min_samples_leaf=5))]
    model = StackingClassifier(estimators=base_models, final_estimator=MLPClassifier(hidden_layer_sizes=(5, 5), max_iter=1000, random_state=42, activation="relu"))
    model.fit(X_train, y_train)
    return model

# Vẽ ma trận nhầm lẫn
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'Confusion Matrix: {model_name}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    st.pyplot(fig)

# Vẽ Learning Curve
def plot_learning_curve(model, X, y, model_name):
    from sklearn.model_selection import learning_curve
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    fig, ax = plt.subplots()
    ax.plot(train_sizes, train_mean, 'o-', color='r', label='Training Score')
    ax.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-Validation Score')
    ax.set_title(f'Learning Curve: {model_name}')
    ax.set_xlabel('Training Examples')
    ax.set_ylabel('Score')
    ax.legend(loc='best')
    ax.grid()
    st.pyplot(fig)

# Tạo giao diện Streamlit
st.title("Raisin Dataset Prediction")

# Sử dụng bộ dữ liệu mặc định
data = pd.read_csv("Raisin_Dataset.csv")
data['Class'] = data['Class'].map({'Kecimen': 0, 'Besni': 1})

# Xử lý dữ liệu
X = data.drop('Class', axis=1)
y = data['Class']
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Chia tập dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Tạo form nhập liệu cho người dùng
st.write("## Input Data for Prediction")
input_data = []
for feature in data.columns[:-1]:
    value = st.number_input(f"Enter {feature}", min_value=0.0, max_value=1000000.0, value=1500.0, step=1.0)
    input_data.append(value)

# Chọn mô hình
st.write("### Choose a model for prediction")
model_choice = st.selectbox("Choose a model", ["Perceptron", "ID3 Decision Tree", "Neural Network", "Stacking Classifier"])

# Định nghĩa tham số cho GridSearchCV và khởi tạo mô hình


params = {}
if model_choice == "Perceptron":
    model = train_perceptron(X_train, y_train, X_test, y_test)
elif model_choice == "ID3 Decision Tree":
    model = train_decision_tree(X_train, y_train, X_test, y_test)
elif model_choice == "Neural Network":
    model = train_neural_network(X_train, y_train, X_test, y_test)
elif model_choice == "Stacking Classifier":
    model = train_stacking_classifier(X_train, y_train, X_test, y_test)

# Sử dụng GridSearchCV để tìm tham số tốt nhất
best_model, best_params, best_score = grid_search_cv(model, params, X_train, y_train, scoring='accuracy', cv=5)

# Huấn luyện mô hình với tham số tốt nhất
best_model.fit(X_train, y_train)

# Tính toán các chỉ số lỗi sau khi mô hình đã được huấn luyện
train_error, validation_error, test_error = evaluate_errors(best_model, X_train, y_train, X_test, y_test, cv=5)

# Tạo danh sách các chỉ số lỗi
error_types = ['Training Error', 'Validation Error', 'Test Error']
errors = [train_error, validation_error, test_error]

# Tạo biểu đồ cột
fig, ax = plt.subplots()
ax.bar(error_types, errors, color=['blue', 'orange', 'green'])

# Thêm nhãn cho biểu đồ
ax.set_xlabel('Error Type')
ax.set_ylabel('Error Value')
ax.set_title('Error Analysis for Model')
ax.set_ylim(0, 1)  # Đặt giới hạn trục y từ 0 đến 1 để hiển thị rõ ràng các lỗi

# Khi nhấn nút Dự đoán
if st.button("Predict"):
    input_data = np.array(input_data).reshape(1, -1)
    
    # Chuẩn hóa dữ liệu đầu vào
    input_data = scaler.transform(input_data)
    
    # Dự đoán với mô hình đã huấn luyện
    prediction = best_model.predict(input_data)
    result = "Kecimen" if prediction[0] == 0 else "Besni"
    st.write(f"### Predicted Class: {result}")

    # Đánh giá mô hình trên tập kiểm tra
    y_test_pred = best_model.predict(X_test)
    y_pred = model.predict(X_test)
    test_accuracy, test_precision, test_recall, test_f1, test_report = evaluate_model(y_test, y_test_pred)
    st.text("Classification Report:\n" + test_report)

    train_error, validation_error, test_error = evaluate_errors(best_model, X_train, y_train, X_test, y_test, cv=5)
    
    # Hiển thị giá trị lỗi trên các cột
    for i, v in enumerate(errors):
        ax.text(i, v + 0.02, f"{v:.4f}", ha='center', fontweight='bold')

    # Hiển thị biểu đồ trên giao diện Streamlit
    st.pyplot(fig)

    # Hiển thị ma trận nhầm lẫn
    plot_confusion_matrix(y_test, y_test_pred, model_choice)

    # Hiển thị Learning Curve
    plot_learning_curve(best_model, X, y, model_choice)
