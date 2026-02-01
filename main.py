import torch

import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np

transform = transforms.Compose([
    transforms.Resize(256),
    # MobileNet 规定
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

full_dataset = ImageFolder(root='./data/train', transform=transform)
dataloader = DataLoader(full_dataset, batch_size=4, shuffle=True)

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

# MobileNet输出特征向量
model.classifier = nn.Identity()
model = model.to()
model.eval()

all_features = []  # 特征
all_labels = []  # 0cat, 1dog)

with torch.no_grad():
    for i, (images, labels) in enumerate(dataloader):
        images = images.to()
        features = model(images)

        all_features.append(features.cpu().numpy())
        all_labels.append(labels.numpy())

X = np.concatenate(all_features, axis=0)
y = np.concatenate(all_labels, axis=0)

np.save('./features/features_X.npy', X)
np.save('./features/labels_y.npy', y)