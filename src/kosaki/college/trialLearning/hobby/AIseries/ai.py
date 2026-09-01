import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# データ準備
X = np.array([[2, 3], [4, 1], [3, 6], [6, 7]])
y = np.array([0, 1, 0, 1])

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# モデルのトレーニング
model = LogisticRegression()
model.fit(X_train, y_train)

# テストデータで評価
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
