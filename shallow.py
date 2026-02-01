import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = ImageFolder(root=r'D:\Codes\Pytorch_Project2\data\train', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

full_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

# 从MobileNetV2的18层取4层)
shallow_layers = list(full_model.features.children())[:4]

model = nn.Sequential(
    *shallow_layers,  # 前几层卷积
    nn.AdaptiveAvgPool2d((1, 1)),  # 强行压缩成 1x1
    nn.Flatten()  # 拉平向量
)

model = model.to(device)
model.eval()

all_features = []
all_labels = []
with torch.no_grad():
    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        features = model(images)

        all_features.append(features.cpu().numpy())
        all_labels.append(labels.numpy())
        if (i + 1) % 5 == 0: print(".", end="")

X_shallow = np.concatenate(all_features, axis=0)
y_shallow = np.concatenate(all_labels, axis=0)

np.save('./features/features_shallow_X.npy', X_shallow)
np.save('./features/labels_shallow_y.npy', y_shallow)