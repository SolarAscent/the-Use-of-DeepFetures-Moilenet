# ==========================================
# 运行测试并画图 (浅层特征版)
# ==========================================

# 1. 改这里：加载【浅层特征】(Shallow Features)
# 请确保文件名和你硬盘里的一样，如果你整理过文件夹，记得加上路径 (例如 ./features/...)
print("正在加载浅层特征...")
X_train = np.load('features_shallow_X.npy')
y_train = np.load('labels_shallow_y.npy')

# 2. 依然要打乱数据 (Standard Procedure)
indices = np.arange(X_train.shape[0])
np.random.shuffle(indices)
X_train = X_train[indices]
y_train = y_train[indices]

# 3. 归一化 (关键！)
# 浅层特征的数值范围可能和深层特征不一样（比如很大），这会让 SVM 很难受。
# 我们手动把它们除以最大值，压缩到 -1 到 1 之间，帮助模型收敛。
X_train = X_train / np.max(np.abs(X_train))

# 4. 实例化并训练
# 对于浅层特征，任务变难了，我们可能需要多给点时间 (n_iters=1000)
# 学习率也可以稍微调小一点点，防止震荡
svm = MySVM(learning_rate=0.0005, lambda_param=0.01, n_iters=1000)
svm.fit(X_train, y_train)

# 5. 画图
plt.plot(svm.losses)
plt.title('Training Loss: Shallow Features (Layer 4)')
plt.xlabel('Epochs (x50)')
plt.ylabel('Hinge Loss')
plt.show()