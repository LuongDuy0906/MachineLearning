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

def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, precision, recall, f1, report

def grid_search_cv(model, params, X_train, y_train, scoring='accuracy', cv=5):
    grid_search = GridSearchCV(model, params, cv=cv, scoring=scoring, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    return best_model, best_params, best_score

def train_perceptron(X_train, y_train, X_test, y_test):
    model = Perceptron(random_state=42)
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train, X_test, y_test):
    model = DecisionTreeClassifier(criterion="entropy", random_state=42, max_depth=3, min_samples_leaf=5)
    model.fit(X_train, y_train)
    return model

def train_neural_network(X_train, y_train, X_test, y_test):
    model = MLPClassifier(hidden_layer_sizes=(5, 5), max_iter=2000, activation="relu", random_state=42)
    model.fit(X_train, y_train)
    return model

def train_stacking_classifier(X_train, y_train, X_test, y_test):
    base_models = [('perceptron', Perceptron(random_state=42)),
                   ('decision_tree', DecisionTreeClassifier(criterion="entropy", random_state=42, max_depth=3, min_samples_leaf=5))]
    model = StackingClassifier(estimators=base_models, final_estimator=MLPClassifier(hidden_layer_sizes=(5, 5), max_iter=2000, random_state=42, activation="relu"))
    model.fit(X_train, y_train)
    return model

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'Confusion Matrix: {model_name}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    st.pyplot(fig)

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

st.title("Raisin Dataset Prediction")

data = pd.read_csv("data/Raisin_Dataset.csv")
data['Class'] = data['Class'].map({'Kecimen': 0, 'Besni': 1})

# Xử lý dữ liệu
X = data.drop('Class', axis=1)
y = data['Class']
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

st.write("## Input Data for Prediction")
input_data = []
for feature in data.columns[:-1]:
    value = st.number_input(f"Enter {feature}", min_value=0.0, max_value=1000000.0, value=1500.0, step=1.0, format="%.7f")
    input_data.append(value)

st.write("### Choose a model for prediction")
model_choice = st.selectbox("Choose a model", ["Perceptron", "ID3 Decision Tree", "Neural Network", "Stacking Classifier"])

params = {}
if model_choice == "Perceptron":
    model = train_perceptron(X_train, y_train, X_test, y_test)
elif model_choice == "ID3 Decision Tree":
    model = train_decision_tree(X_train, y_train, X_test, y_test)
elif model_choice == "Neural Network":
    model = train_neural_network(X_train, y_train, X_test, y_test)
elif model_choice == "Stacking Classifier":
    model = train_stacking_classifier(X_train, y_train, X_test, y_test)

best_model, best_params, best_score = grid_search_cv(model, params, X_train, y_train, scoring='accuracy', cv=5)

best_model.fit(X_train, y_train)

if st.button("Predict"):
    input_data = np.array(input_data).reshape(1, -1)
   
    input_data = scaler.transform(input_data)
   
    prediction = best_model.predict(input_data)
    result = "Kecimen" if prediction[0] == 0 else "Besni"
    st.write(f"### Predicted Class: {result}")

    y_test_pred = best_model.predict(X_test)
    y_pred = model.predict(X_test)
    test_accuracy, test_precision, test_recall, test_f1, test_report = evaluate_model(y_test, y_test_pred)
    st.text("Classification Report:\n" + test_report)

    plot_confusion_matrix(y_test, y_test_pred, model_choice)

    plot_learning_curve(best_model, X, y, model_choice)
