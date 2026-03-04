import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
def run_dt_scenario():
    print("--- SCENARIO 2: DECISION TREE CLASSIFIER ---")
    data_path = r'c:\Users\kamal\Downloads\archive (9)\train_u6lujuX_CVtuZ9i (1).csv'
    df = pd.read_csv(data_path)
    print("\nInitial Info:")
    print(df.info())
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
    df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
    df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
    df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
    df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
    df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
    features = ['ApplicantIncome', 'LoanAmount', 'Credit_History', 'Education', 'Property_Area']
    X = df[features]
    y = df['Loan_Status']
    le = LabelEncoder()
    X['Education'] = le.fit_transform(X['Education'])
    X['Property_Area'] = le.fit_transform(X['Property_Area'])
    y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    depths = [2, 3, 5, 10, None]
    results = []
    for d in depths:
        dt = DecisionTreeClassifier(max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, dt.predict(X_train))
        test_acc = accuracy_score(y_test, dt.predict(X_test))
        results.append((d, train_acc, test_acc))
        print(f"Depth {d}: Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}")
    print("\nObservation on Overfitting:")
    print("If Training Accuracy is much higher than Testing Accuracy as depth increases, overfitting is occurring.")
    best_dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    best_dt.fit(X_train, y_train)
    y_pred = best_dt.
    print("\nFinal Model Performance (Depth=3):")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Rejected', 'Approved'], yticklabels=['Rejected', 'Approved'])
    plt.title('Confusion Matrix - Loan Prediction')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(r'c:\Users\kamal\Downloads\dt_confusion_matrix.png')
    plt.close()
    plt.figure(figsize=(20, 10))
    plot_tree(best_dt, feature_names=features, class_names=['Rejected', 'Approved'], filled=True)
    plt.title('Decision Tree Structure (Max Depth 3)')
    plt.savefig(r'c:\Users\kamal\Downloads\dt_tree_structure.png')
    plt.close()
    importance = best_dt.feature_importances_
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance, y=features)
    plt.title('Feature Importance Plot')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.savefig(r'c:\Users\kamal\Downloads\dt_feature_importance.png')
    plt.close()
    print("\nVisualizations saved to c:\\Users\\kamal\\Downloads")

if __name__ == "__main__":
    run_dt_scenario()

