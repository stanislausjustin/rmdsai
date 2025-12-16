import pandas as pd

df = pd.read_csv('PCOS.csv')
df.head()

X = df.drop('PCOS_Diagnosis', axis=1)
y = df['PCOS_Diagnosis']

print("Shape of features (X):", X.shape)
print("Shape of target (y):", y.shape)

print("\nMissing values per column in features (X):\n")
print(X.isnull().sum())

X = X.drop('id', axis=1)

from sklearn.preprocessing import StandardScaler

#identify numerical features to scale
numerical_features = ['Age', 'BMI', 'Testosterone_Level(ng/dL)', 'Antral_Follicle_Count']

#initialize StandardScaler
scaler = StandardScaler()

#apply scaling to numerical features
X[numerical_features] = scaler.fit_transform(X[numerical_features])

print("Features after dropping 'id' and scaling numerical columns:")
print(X.head())

from sklearn.model_selection import train_test_split

#split the data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

print("\nValue counts for y_train (stratified):")
print(y_train.value_counts(normalize=True))
print("\nValue counts for y_test (stratified):")
print(y_test.value_counts(normalize=True))

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#initialize the SVM classifier
svm_model = SVC(random_state=42)

#train the SVM model
svm_model.fit(X_train, y_train)

#make predictions on the test set
y_pred_svm = svm_model.predict(X_test)

#calculate evaluation metrics
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)

#print the metrics
print(f"SVM Model Performance:\n")
print(f"Accuracy: {accuracy_svm:.4f}")
print(f"Precision: {precision_svm:.4f}")
print(f"Recall: {recall_svm:.4f}")
print(f"F1-Score: {f1_svm:.4f}")

from sklearn.neural_network import MLPClassifier

#initialize the Neural Network classifier
nn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)

#train the Neural Network model
nn_model.fit(X_train, y_train)

#make predictions on the test set
y_pred_nn = nn_model.predict(X_test)

#calculate evaluation metrics
accuracy_nn = accuracy_score(y_test, y_pred_nn)
precision_nn = precision_score(y_test, y_pred_nn)
recall_nn = recall_score(y_test, y_pred_nn)
f1_nn = f1_score(y_test, y_pred_nn)

# print the metrics
print(f"Neural Network Model Performance:\n")
print(f"Accuracy: {accuracy_nn:.4f}")
print(f"Precision: {precision_nn:.4f}")
print(f"Recall: {recall_nn:.4f}")
print(f"F1-Score: {f1_nn:.4f}")

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

nn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42) # Increased max_iter

nn_model.fit(X_train, y_train)

y_pred_nn = nn_model.predict(X_test)

accuracy_nn = accuracy_score(y_test, y_pred_nn)
precision_nn = precision_score(y_test, y_pred_nn)
recall_nn = recall_score(y_test, y_pred_nn)
f1_nn = f1_score(y_test, y_pred_nn)

print(f"Neural Network Model Performance:\n")
print(f"Accuracy: {accuracy_nn:.4f}")
print(f"Precision: {precision_nn:.4f}")
print(f"Recall: {recall_nn:.4f}")
print(f"F1-Score: {f1_nn:.4f}")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

rf_model = RandomForestClassifier(random_state=42)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

print(f"Random Forest Model Performance:\n")
print(f"Accuracy: {accuracy_rf:.4f}")
print(f"Precision: {precision_rf:.4f}")
print(f"Recall: {recall_rf:.4f}")
print(f"F1-Score: {f1_rf:.4f}")

import pandas as pd

metrics_data = {
    'Model': ['SVM', 'Neural Network', 'Random Forest'],
    'Accuracy': [accuracy_svm, accuracy_nn, accuracy_rf],
    'Precision': [precision_svm, precision_nn, precision_rf],
    'Recall': [recall_svm, recall_nn, recall_rf],
    'F1-Score': [f1_svm, f1_nn, f1_rf]
}

performance_df = pd.DataFrame(metrics_data)

print("Model Performance Comparison:")
print(performance_df.round(4))

import matplotlib.pyplot as plt

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()
for i, metric in enumerate(metrics):
    ax = axes[i]
    bars = ax.bar(performance_df['Model'], performance_df[metric], color=['skyblue', 'lightcoral', 'lightgreen'])
    ax.set_title(f'Model {metric} Comparison', fontsize=14)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_ylim(0.9, 1.0)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.005, round(yval, 4), ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()


from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

#SMOTE
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train, y_train)

print("Class distribution after SMOTE:\n")
print(y_smote.value_counts())

#SMOTEENN
smotenn = SMOTEENN(random_state=42)
X_smotenn, y_smotenn = smotenn.fit_resample(X_train, y_train)

print("\nClass distribution after SMOTEENN:\n")
print(y_smotenn.value_counts())


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

smote_svm_model = SVC(random_state=42)

smote_svm_model.fit(X_smotenn, y_smotenn)

y_pred_smotenn_train = smote_svm_model.predict(X_smotenn)

accuracy_smotenn_train = accuracy_score(y_smotenn, y_pred_smotenn_train)
precision_smotenn_train = precision_score(y_smotenn, y_pred_smotenn_train)
recall_smotenn_train = recall_score(y_smotenn, y_pred_smotenn_train)
f1_smotenn_train = f1_score(y_smotenn, y_pred_smotenn_train)

print(f"SVM Model Performance (SMOTENN Training Set):\n")
print(f"Accuracy: {accuracy_smotenn_train:.4f}")
print(f"Precision: {precision_smotenn_train:.4f}")
print(f"Recall: {recall_smotenn_train:.4f}")
print(f"F1-Score: {f1_smotenn_train:.4f}")

y_pred_smotenn_test = smote_svm_model.predict(X_test)

accuracy_smotenn_test = accuracy_score(y_test, y_pred_smotenn_test)
precision_smotenn_test = precision_score(y_test, y_pred_smotenn_test)
recall_smotenn_test = recall_score(y_test, y_pred_smotenn_test)
f1_smotenn_test = f1_score(y_test, y_pred_smotenn_test)

print(f"\nSVM Model Performance (Original Test Set with SMOTENN-trained model):\n")
print(f"Accuracy: {accuracy_smotenn_test:.4f}")
print(f"Precision: {precision_smotenn_test:.4f}")
print(f"Recall: {recall_smotenn_test:.4f}")
print(f"F1-Score: {f1_smotenn_test:.4f}")


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

nn_smotenn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)

nn_smotenn_model.fit(X_smotenn, y_smotenn)

y_pred_smotenn_nn_train = nn_smotenn_model.predict(X_smotenn)

accuracy_smotenn_nn_train = accuracy_score(y_smotenn, y_pred_smotenn_nn_train)
precision_smotenn_nn_train = precision_score(y_smotenn, y_pred_smotenn_nn_train)
recall_smotenn_nn_train = recall_score(y_smotenn, y_pred_smotenn_nn_train)
f1_smotenn_nn_train = f1_score(y_smotenn, y_pred_smotenn_nn_train)

print(f"Neural Network Model Performance (SMOTENN Training Set):\n")
print(f"Accuracy: {accuracy_smotenn_nn_train:.4f}")
print(f"Precision: {precision_smotenn_nn_train:.4f}")
print(f"Recall: {recall_smotenn_nn_train:.4f}")
print(f"F1-Score: {f1_smotenn_nn_train:.4f}")

y_pred_smotenn_nn_test = nn_smotenn_model.predict(X_test)

accuracy_smotenn_nn_test = accuracy_score(y_test, y_pred_smotenn_nn_test)
precision_smotenn_nn_test = precision_score(y_test, y_pred_smotenn_nn_test)
recall_smotenn_nn_test = recall_score(y_test, y_pred_smotenn_nn_test)
f1_smotenn_nn_test = f1_score(y_test, y_pred_smotenn_nn_test)

print(f"\nNeural Network Model Performance (Original Test Set with SMOTENN-trained model):\n")
print(f"Accuracy: {accuracy_smotenn_nn_test:.4f}")
print(f"Precision: {precision_smotenn_nn_test:.4f}")
print(f"Recall: {recall_smotenn_nn_test:.4f}")
print(f"F1-Score: {f1_smotenn_nn_test:.4f}")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

rf_smotenn_model = RandomForestClassifier(random_state=42)

rf_smotenn_model.fit(X_smotenn, y_smotenn)

y_pred_smotenn_rf_train = rf_smotenn_model.predict(X_smotenn)

accuracy_smotenn_rf_train = accuracy_score(y_smotenn, y_pred_smotenn_rf_train)
precision_smotenn_rf_train = precision_score(y_smotenn, y_pred_smotenn_rf_train)
recall_smotenn_rf_train = recall_score(y_smotenn, y_pred_smotenn_rf_train)
f1_smotenn_rf_train = f1_score(y_smotenn, y_pred_smotenn_rf_train)

print(f"Random Forest Model Performance (SMOTENN Training Set):\n")
print(f"Accuracy: {accuracy_smotenn_rf_train:.4f}")
print(f"Precision: {precision_smotenn_rf_train:.4f}")
print(f"Recall: {recall_smotenn_rf_train:.4f}")
print(f"F1-Score: {f1_smotenn_rf_train:.4f}")

y_pred_smotenn_rf_test = rf_smotenn_model.predict(X_test)

accuracy_smotenn_rf_test = accuracy_score(y_test, y_pred_smotenn_rf_test)
precision_smotenn_rf_test = precision_score(y_test, y_pred_smotenn_rf_test)
recall_smotenn_rf_test = recall_score(y_test, y_pred_smotenn_rf_test)
f1_smotenn_rf_test = f1_score(y_test, y_pred_smotenn_rf_test)

print(f"\nRandom Forest Model Performance (Original Test Set with SMOTENN-trained model):\n")
print(f"Accuracy: {accuracy_smotenn_rf_test:.4f}")
print(f"Precision: {precision_smotenn_rf_test:.4f}")
print(f"Recall: {recall_smotenn_rf_test:.4f}")
print(f"F1-Score: {f1_smotenn_rf_test:.4f}")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

feature_importances = rf_smotenn_model.feature_importances_

features_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
})

features_df = features_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=features_df, palette='viridis')
plt.title('Random Forest Feature Importance (SMOTENN)', fontsize=16)
plt.xlabel('Importance', fontsize=12);
plt.ylabel('Feature', fontsize=12);
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

feature_importances = rf_smotenn_model.feature_importances_

features_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
})

features_df = features_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', hue='Feature', data=features_df, palette='viridis', legend=False)
plt.title('Random Forest Feature Importance (SMOTENN)', fontsize=16)
plt.xlabel('Importance', fontsize=12);
plt.ylabel('Feature', fontsize=12);
plt.show()

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

models_to_evaluate = [
    ('SVM', smote_svm_model, y_pred_smotenn_test),
    ('Neural Network', nn_smotenn_model, y_pred_smotenn_nn_test),
    ('Random Forest', rf_smotenn_model, y_pred_smotenn_rf_test)
]

for model_name, model_obj, y_pred in models_to_evaluate:
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.show()

import pandas as pd

metrics_smotenn_data = {
    'Model': [
        'SVM',
        'SVM',
        'Neural Network',
        'Neural Network',
        'Random Forest',
        'Random Forest'
    ],
    'Set': [
        'Training',
        'Testing',
        'Training',
        'Testing',
        'Training',
        'Testing'
    ],
    'Accuracy': [
        accuracy_smotenn_train, accuracy_smotenn_test,
        accuracy_smotenn_nn_train, accuracy_smotenn_nn_test,
        accuracy_smotenn_rf_train, accuracy_smotenn_rf_test
    ],
    'Precision': [
        precision_smotenn_train, precision_smotenn_test,
        precision_smotenn_nn_train, precision_smotenn_nn_test,
        precision_smotenn_rf_train, precision_smotenn_rf_test
    ],
    'Recall': [
        recall_smotenn_train, recall_smotenn_test,
        recall_smotenn_nn_train, recall_smotenn_nn_test,
        recall_smotenn_rf_train, recall_smotenn_rf_test
    ],
    'F1-Score': [
        f1_smotenn_train, f1_smotenn_test,
        f1_smotenn_nn_train, f1_smotenn_nn_test,
        f1_smotenn_rf_train, f1_smotenn_rf_test
    ]
}

performance_smotenn_df = pd.DataFrame(metrics_smotenn_data)

print("Model Performance Comparison with SMOTENN:")
print(performance_smotenn_df.round(4).to_string(index=False))

import matplotlib.pyplot as plt
import pandas as pd

performance_smotenn_test_df = performance_smotenn_df[performance_smotenn_df['Set'] == 'Testing'].copy()

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for i, metric in enumerate(metrics):
    ax = axes[i]
    bars = ax.bar(performance_smotenn_test_df['Model'], performance_smotenn_test_df[metric], color=['skyblue', 'lightcoral', 'lightgreen'])
    ax.set_title(f'Model {metric} Comparison (SMOTENN Test Set)', fontsize=14)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_ylim(0.85, 1.05)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 4), ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()