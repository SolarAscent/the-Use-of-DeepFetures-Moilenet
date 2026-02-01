import numpy as np
import matplotlib.pyplot as plt

class NewSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.losses = []

    def _get_loss(self, X, y):
        """
        计算当前的 Hinge Loss
        公式: Loss = mean(max(0, 1 - y*(wx - b))) + lambda * ||w||^2
        """
        # 1. 1 - y * (wx - b)
        distances = 1 - y * (np.dot(X, self.w) - self.b)

        # 2.max(0, distance)
        distances[distances < 0] = 0
        hinge_loss = np.mean(distances)

        # 3. lambda * ||w||^2
        reg_loss = self.lambda_param * np.dot(self.w, self.w)

        return hinge_loss + reg_loss

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # 转换标签为 -1 和 1
        y_ = np.where(y <= 0, -1, 1)

        # 初始化参数
        self.w = np.zeros(n_features)
        self.b = 0
        self.losses = []

        for epoch in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]
            if epoch % 50 == 0:
                current_loss = self._get_loss(X, y_)
                self.losses.append(current_loss)
                print(f"Epoch {epoch}: Loss = {current_loss:.4f}")

    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)

X_train = np.load('./features/features_X.npy')
y_train = np.load('./features/labels_y.npy')

indices = np.arange(X_train.shape[0])
np.random.shuffle(indices)
X_train = X_train[indices]
y_train = y_train[indices]

svm = NewSVM(learning_rate=0.001, lambda_param=0.01, n_iters=500)
svm.fit(X_train, y_train)

# 画出 Loss 曲线
plt.plot(svm.losses)
plt.title('Training Loss Curve')
plt.xlabel('Epochs (x50)')
plt.ylabel('Hinge Loss')
plt.show()

X_train = np.load('./features/features_shallow_X.npy')
y_train = np.load('./features/labels_shallow_y.npy')

indices = np.arange(X_train.shape[0])
np.random.shuffle(indices)
X_train = X_train[indices]
y_train = y_train[indices]

X_train = X_train / np.max(np.abs(X_train))

svm = NewSVM(learning_rate=0.0005, lambda_param=0.01, n_iters=1000)
svm.fit(X_train, y_train)

plt.plot(svm.losses)
plt.title('Training Loss: Shallow Features (Layer 4)')
plt.xlabel('Epochs (x50)')
plt.ylabel('Hinge Loss')
plt.show()