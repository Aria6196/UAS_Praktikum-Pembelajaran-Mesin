#Aria Octavian Hamza
#1227050025
#Praktikum Pembelajaran Mesin A

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load dataset
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

data = pd.read_csv('diabetes.csv')

# 2. Pisahkan fitur dan label
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# 3. Scaling fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split data ke train dan test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Buat dan latih model SVM
model = SVC(kernel='rbf')  # bisa juga 'linear', 'poly', 'sigmoid'
model.fit(X_train, y_train)

# 6. Prediksi dan evaluasi
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
