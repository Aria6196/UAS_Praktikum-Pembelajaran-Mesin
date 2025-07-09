#Devi Mulyana
#1227050035
#Praktikum Pembelajaran Mesin A

#%% Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#%% 1. Load dataset
df = pd.read_csv("diabetes.csv")

#%% 2. Eksplorasi data
print("Dataset Shape:", df.shape)
print(df.head())
print("\nStatistik deskriptif:")
print(df.describe())

#%% 3. Cek missing value (biasanya tidak ada di dataset ini)
print("\nJumlah nilai nol di tiap kolom:")
print((df == 0).sum())  # untuk kolom seperti Glucose, BloodPressure bisa jadi 0 berarti missing

#%% (Opsional) Ganti 0 dengan NaN untuk kolom medis yang tidak mungkin bernilai nol
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)

#%% Bisa pilih: drop NaN atau isi dengan median
df.fillna(df.median(), inplace=True)

#%% 4. Pisahkan fitur dan label
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

#%% 5. Split data training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%% 6. Inisialisasi dan latih model Decision Tree
model = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
model.fit(X_train, y_train)

#%% 7. Prediksi data testing
y_pred = model.predict(X_test)

#%% 8. Evaluasi model
print("\n=== Hasil Evaluasi ===")
print("Akurasi:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#%% 9. Visualisasi Decision Tree
plt.figure(figsize=(50,25))
plot_tree(model, feature_names=X.columns, class_names=["Non-diabetes", "Diabetes"], filled=True)
plt.title("Visualisasi Decision Tree")
plt.show()

# %%
