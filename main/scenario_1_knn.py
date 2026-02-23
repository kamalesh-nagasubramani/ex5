import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from matplotlib.colors import ListedColormap

# student roll numbers (Placeholder - You can replace this)
# Roll Number: 24BAD000 (Example)

def run_knn_scenario():
    print("--- SCENARIO 1: K-NEAREST NEIGHBORS (KNN) ---")
    
    # 1. Load the Breast Cancer dataset
    data_path = r'c:\Users\kamal\Downloads\archive (10)\breast-cancer.csv'
    df = pd.read_csv(data_path)
    
    # 2. Data inspection & preprocessing
    print("\nDataset Info:")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Selecting requested features
    features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']
    X = df[features]
    y = df['diagnosis']
    
    # 3. Encode target labels
    le = LabelEncoder()
    y = le.fit_transform(y)  # M -> 1, B -> 0
    
    # 4. Apply feature scaling (very important for KNN)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 5. Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # 6. Train a KNN classifier and Experiment with different values of K
    k_values = range(1, 21)
    accuracies = []
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
    
    # 7. Analyze model sensitivity to K (Accuracy vs K Plot)
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, marker='o', linestyle='--', color='b')
    plt.title('Accuracy vs K Value')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy')
    plt.xticks(k_values)
    plt.grid()
    plt.savefig(r'c:\Users\kamal\Downloads\knn_accuracy_vs_k.png')
    plt.close()
    
    # 8. Train the best KNN (let's pick K=5 or best K from accuracies)
    best_k = k_values[np.argmax(accuracies)]
    print(f"\nBest K identified: {best_k}")
    knn_best = KNeighborsClassifier(n_neighbors=best_k)
    knn_best.fit(X_train, y_train)
    y_pred_best = knn_best.predict(X_test)
    
    # 9. Evaluate performance
    print("\nFinal Model Performance (Best K):")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_best):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_best):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred_best):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred_best):.4f}")
    
    # 10. Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix (K={best_k})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(r'c:\Users\kamal\Downloads\knn_confusion_matrix.png')
    plt.close()
    
    # 11. Identify misclassified cases
    misclassified = np.where(y_test != y_pred_best)[0]
    print(f"\nNumber of misclassified cases: {len(misclassified)}")
    if len(misclassified) > 0:
        print("Indices of misclassified cases:", misclassified[:10])
    
    # 12. Decision Boundary (using two features: radius_mean and texture_mean)
    # Re-train on 2 features for visualization
    X_vis = X_scaled[:, :2]
    knn_vis = KNeighborsClassifier(n_neighbors=best_k)
    knn_vis.fit(X_vis, y)
    
    # Create mesh grid
    h = .02
    x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
    y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = knn_vis.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    
    plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    plt.title(f"3-Class classification (k = {best_k}, weights = 'uniform')")
    plt.xlabel('Radius Mean (scaled)')
    plt.ylabel('Texture Mean (scaled)')
    plt.title('Decision Boundary using Radius Mean and Texture Mean')
    plt.savefig(r'c:\Users\kamal\Downloads\knn_decision_boundary.png')
    plt.close()
    
    print("\nVisualizations saved to c:\\Users\\kamal\\Downloads")

if __name__ == "__main__":
    run_knn_scenario()
