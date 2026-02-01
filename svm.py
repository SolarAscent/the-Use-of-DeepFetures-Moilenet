import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

X = np.load('./features/features_X.npy')
y = np.load('./features/labels_y.npy')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_classifier = SVC(kernel='linear', C=1.0)
svm_classifier.fit(X_train, y_train)

y_pred = svm_classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accutacity: {acc * 100:.2f}%")

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Cat', 'Predicted Dog'],
            yticklabels=['Actual Cat', 'Actual Dog'])
plt.title('Confusion Matrix (SVM)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()