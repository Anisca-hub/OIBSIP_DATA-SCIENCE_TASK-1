# ==============================
# TASK 1 IRIS FLOWER CLASSIFICATION
# ==============================

# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# ------------------------------
# 1. LOAD DATASET
# ------------------------------
data = pd.read_csv(r"C:\Users\anisc\Downloads\TASK 1\Iris.csv")

# Display first 5 rows
print("Dataset Preview:")
print(data.head())

# Drop Id column
data = data.drop("Id", axis=1)

# ------------------------------
# 2. DATA VISUALIZATION (EDA)
# ------------------------------

# Pair plot to visualize relationships
sns.pairplot(data, hue="Species")
plt.show()

# Sepal Length vs Petal Length
plt.figure()
sns.scatterplot(
    x="SepalLengthCm",
    y="PetalLengthCm",
    hue="Species",
    data=data
)
plt.title("Sepal Length vs Petal Length")
plt.show()

# ------------------------------
# 3. SPLIT FEATURES & TARGET
# ------------------------------
X = data.drop("Species", axis=1)
y = data["Species"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------
# 4. LOGISTIC REGRESSION MODEL
# ------------------------------
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_predictions)

print("\nLogistic Regression Accuracy:", lr_accuracy)

# CONFUSION MATRIX - LOGISTIC REGRESSION
cm_lr = confusion_matrix(y_test, lr_predictions)

disp_lr = ConfusionMatrixDisplay(
    confusion_matrix=cm_lr,
    display_labels=lr_model.classes_
)

disp_lr.plot()
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

# ------------------------------
# 5. K-NEAREST NEIGHBORS MODEL
# ------------------------------
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

knn_predictions = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_predictions)

print("KNN Accuracy:", knn_accuracy)

# CONFUSION MATRIX - KNN
cm_knn = confusion_matrix(y_test, knn_predictions)

disp_knn = ConfusionMatrixDisplay(
    confusion_matrix=cm_knn,
    display_labels=knn_model.classes_
)

disp_knn.plot()
plt.title("Confusion Matrix - KNN")
plt.show()

# ------------------------------
# 6. SAMPLE PREDICTION
# ------------------------------
sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=X.columns)

print("\nSample Prediction using Logistic Regression:", lr_model.predict(sample)[0])

print("Sample Prediction using KNN:", knn_model.predict(sample)[0])