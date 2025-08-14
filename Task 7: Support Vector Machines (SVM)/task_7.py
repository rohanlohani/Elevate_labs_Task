import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

# 1. Load and prepare dataset
df = pd.read_csv("/home/rohan/Desktop/Elevate Labs/Task 7: Support Vector Machines (SVM)/brca.csv")
df = df.drop(columns=["Unnamed: 0"])  # drop index col
X = df.drop(columns=["y"])
y = df["y"].map({"B": 0, "M": 1})  # encode target

# We'll choose only 2 features for decision boundary visualization
feature_pair = ("x.radius_mean", "x.texture_mean")
X_pair = X[list(feature_pair)]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_pair, y, test_size=0.3, random_state=42, stratify=y
)

# 2. Define pipelines for linear and RBF SVM
pipe_linear = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="linear"))
])
pipe_rbf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf"))
])

# 3. Hyperparameter tuning (C, gamma)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_grid_linear = {
    "svm__C": [0.1, 1, 10, 100]
}
param_grid_rbf = {
    "svm__C": [0.1, 1, 10, 100],
    "svm__gamma": [0.001, 0.01, 0.1, 1, "scale"]
}

grid_linear = GridSearchCV(pipe_linear, param_grid_linear, cv=cv)
grid_rbf = GridSearchCV(pipe_rbf, param_grid_rbf, cv=cv)

grid_linear.fit(X_train, y_train)
grid_rbf.fit(X_train, y_train)

print("Best Linear SVM params:", grid_linear.best_params_)
print("Best RBF SVM params:", grid_rbf.best_params_)
print("Linear SVM CV score:", grid_linear.best_score_)
print("RBF SVM CV score:", grid_rbf.best_score_)

# 4. Visualization function for decision boundary
def plot_decision_boundary(model, X, y, title):
    X = X.values
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_df = pd.DataFrame(grid, columns=[feature_pair[0], feature_pair[1]])
    Z = model.predict(grid_df)
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.xlabel(feature_pair[0])
    plt.ylabel(feature_pair[1])
    plt.title(title)
    plt.show()

# 5. Plot decision boundaries for best models
plot_decision_boundary(grid_linear.best_estimator_, X_pair, y, "Linear SVM Decision Boundary")
plot_decision_boundary(grid_rbf.best_estimator_, X_pair, y, "RBF SVM Decision Boundary")
