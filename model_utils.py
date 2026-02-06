import time
import numpy as np
import torch
import torchvision
import sklearn.svm, sklearn.metrics
from torchvision import transforms
from tqdm import tqdm
import config

# ---------- DataLoader ----------
def make_loaders():
    tfm = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])
    train_ds = torchvision.datasets.ImageFolder(config.SPLIT_ROOT/'train', transform=tfm)
    val_ds   = torchvision.datasets.ImageFolder(config.SPLIT_ROOT/'val',   transform=tfm)
    test_ds  = torchvision.datasets.ImageFolder(config.SPLIT_ROOT/'test',  transform=tfm)
    train_ld = torch.utils.data.DataLoader(train_ds, batch_size=config.BATCH, shuffle=True, num_workers=2)
    val_ld   = torch.utils.data.DataLoader(val_ds,   batch_size=config.BATCH, shuffle=False, num_workers=2)
    test_ld  = torch.utils.data.DataLoader(test_ds,  batch_size=config.BATCH, shuffle=False, num_workers=2)
    return train_ld, val_ld, test_ld

def build_extractor(cut=3):
    try:
        net = torchvision.models.mobilenet_v2(weights='IMAGENET1K_V1').features.to(config.DEVICE).eval()
    except:
        try:
            net = torchvision.models.mobilenet_v2(pretrained=True).features.to(config.DEVICE).eval()
        except:
            net = torchvision.models.mobilenet_v2(weights=None).features.to(config.DEVICE).eval()
    net = net[:cut]
    for p in net.parameters(): p.requires_grad = False
    return net

def extract(loader, net):
    feats, labels = [], []
    for x, y in tqdm(loader, desc=f'extract cut={len(net)}'):
        x = x.to(config.DEVICE)
        with torch.no_grad():
            f = net(x).mean([2, 3])  # Global Average Pooling
        feats.append(f.cpu())
        labels.append(y)
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()

# ---------- SVM ----------
def compute_hinge_loss(clf, X, y, n_classes=None):
    if n_classes is None:
        n_classes = len(config.CLASS_NAMES)
    try:
        decisions = clf.decision_function(X)
        if decisions.ndim == 1:
            # 2
            y_binary = (y == 1).astype(float) * 2 - 1
            return float(np.mean(np.maximum(0, 1 - y_binary * decisions)))
        # hinge loss for multi-class
        losses = []
        for i in range(n_classes):
            y_binary = (y == i).astype(float) * 2 - 1
            margin = y_binary * decisions[:, i]
            losses.append(np.mean(np.maximum(0, 1 - margin)))
        return float(np.mean(losses))
    except Exception:
        return 0.0

def train_and_evaluate_svm(train_x, train_y, val_x, val_y, test_x, test_y):
    n_classes = len(config.CLASS_NAMES)
    clf = sklearn.svm.LinearSVC(C=1.0, max_iter=5000)
    
    # Cost
    train_start = time.time()
    clf.fit(train_x, train_y)
    train_time = time.time() - train_start
    
    # Prediction
    train_pred = clf.predict(train_x)
    val_start = time.time()
    val_pred = clf.predict(val_x)
    val_time = time.time() - val_start
    test_start = time.time()
    test_pred = clf.predict(test_x)
    test_time = time.time() - test_start
    
    # Accuracy
    train_acc = sklearn.metrics.accuracy_score(train_y, train_pred)
    val_acc   = sklearn.metrics.accuracy_score(val_y,   val_pred)
    test_acc  = sklearn.metrics.accuracy_score(test_y,  test_pred)
    
    # loss
    train_loss = compute_hinge_loss(clf, train_x, train_y, n_classes=n_classes)
    val_loss = compute_hinge_loss(clf, val_x, val_y, n_classes=n_classes)
    test_loss = compute_hinge_loss(clf, test_x, test_y, n_classes=n_classes)
    
    return {
        'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': test_acc,
        'train_loss': train_loss, 'val_loss': val_loss, 'test_loss': test_loss,
        'train_time': train_time, 'val_time': val_time, 'test_time': test_time
    }
