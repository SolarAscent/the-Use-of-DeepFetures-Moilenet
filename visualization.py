import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

X_deep = np.load('./features/features_X.npy')
y_deep = np.load('./features/labels_y.npy')
X_shallow = np.load('./features/features_shallow_X.npy')
y_shallow = np.load('./features/labels_shallow_y.npy')

limit = 1000
X_deep = X_deep[:limit]
y_deep = y_deep[:limit]
X_shallow = X_shallow[:limit]
y_shallow = y_shallow[:limit]

tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
X_deep_2d = tsne.fit_transform(X_deep)
X_shallow_2d = tsne.fit_transform(X_shallow)
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 猫(蓝色), 狗(橙色)
palette = sns.color_palette("bright", 2)

# 左深
sns.scatterplot(x=X_deep_2d[:,0], y=X_deep_2d[:,1], hue=y_deep, palette=palette, ax=axes[0], s=60)
axes[0].set_title('Deep Features,Layer 18) ', fontsize=16)
axes[0].set_xlabel('t-SNE Dimension 1')
axes[0].set_ylabel('t-SNE Dimension 2')
axes[0].legend(title='Class', labels=['Cat', 'Dog'])

# 右浅
sns.scatterplot(x=X_shallow_2d[:,0], y=X_shallow_2d[:,1], hue=y_shallow, palette=palette, ax=axes[1], s=60)
axes[1].set_title('Shallow Features,Layer 4', fontsize=16)
axes[1].set_xlabel('t-SNE Dimension 1')
axes[1].set_ylabel('t-SNE Dimension 2')
axes[1].legend(title='Class', labels=['Cat', 'Dog'])

plt.tight_layout()
plt.show()