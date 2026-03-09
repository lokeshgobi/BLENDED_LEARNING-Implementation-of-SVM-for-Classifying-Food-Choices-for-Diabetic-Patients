# BLENDED LEARNING
# Implementation of Support Vector Machine for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a Support Vector Machine (SVM) model to classify food items and optimize hyperparameters for better accuracy.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Load the dataset and separate features and target variable.
2.Split the data into training and testing sets and scale the features.
3.Train the Support Vector Machine model using Grid Search to find the best parameters.
4.Predict the test data and evaluate performance using accuracy, classification report, and confusion matrix.
```
## Program:
```
/*
Program to implement SVM for food classification for diabetic patients.
Developed by: LOKESHWARAN G
RegisterNumber:  212225040210
*/
```
```
Program to implement SVM  
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('food_items_binary.csv')
print(data.head())
print(data.columns)
features = ['Calories','Total Fat','Saturated Fat','Sugars','Dietary Fiber','Protein']
target = 'class'
x = data[features]
y = data[target]
x_train, x_test, y_train,  y_test = train_test_split(x, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
svm = SVC()
param_grid = {
    'C': [0.1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale','auto']
}
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
grid_search.fit(x_train,y_train)
best_model = grid_search.best_estimator_
print("Name: LOKESHWARAN G")
print("Register Number:212225040210")
print("Best Parameters:", grid_search.best_params_)
y_pred = best_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Name: LOKESHWARAN G")
print("Register Number:212225040210")
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()                             
```

## OUTPUt:

<img width="915" height="843" alt="Screenshot 2026-03-09 142413" src="https://github.com/user-attachments/assets/5c22e8b2-10a3-49f1-882b-289bf16902ea" />


## Result:
Thus, the SVM model was successfully implemented to classify food items for diabetic patients, with hyperparameter tuning optimizing the model's performance.
