import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import torchvision
from torchvision import transforms
from tqdm import tqdm
import config

# --------------------Data loader-------------------------------------
def make_loaders():
    num_workers = min(os.cpu_count(), 25)
    print(f"DataLoader: {num_workers} workers, Pin_Memory=True, Batch={config.BATCH}")

    tfm = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = torchvision.datasets.ImageFolder(config.SPLIT_ROOT/'train', transform=tfm)
    val_ds   = torchvision.datasets.ImageFolder(config.SPLIT_ROOT/'val',   transform=tfm)
    test_ds  = torchvision.datasets.ImageFolder(config.SPLIT_ROOT/'test',  transform=tfm)

    loader_args = dict(
        batch_size=config.BATCH,
        num_workers=num_workers,
        pin_memory=True,          
        persistent_workers=True,  
        prefetch_factor=4, 
    )

    train_ld = torch.utils.data.DataLoader(train_ds, shuffle=True, **loader_args)
    val_ld   = torch.utils.data.DataLoader(val_ds, shuffle=False, **loader_args)
    test_ld  = torch.utils.data.DataLoader(test_ds, shuffle=False, **loader_args)
    
    return train_ld, val_ld, test_ld

# ---------------------------------------------------------
# ---------------------------------------------------------
def build_vit_model(num_classes):
    print("Loading ViT-B/16 Model")
    try:
        weights = torchvision.models.ViT_B_16_Weights.DEFAULT
        model = torchvision.models.vit_b_16(weights=weights)
    except:
        model = torchvision.models.vit_b_16(pretrained=True)
    
    if hasattr(model, 'heads'):
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
    else:
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
    
    model = model.to(config.DEVICE)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model

# ---------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    
    for inputs, labels in tqdm(loader, desc='Training ViT', leave=True):
        inputs = inputs.to(config.DEVICE, non_blocking=True)
        labels = labels.to(config.DEVICE, non_blocking=True)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    epoch_time = time.time() - start_time
    return running_loss / total, correct / total, epoch_time

def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(config.DEVICE, non_blocking=True)
            labels = labels.to(config.DEVICE, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss / total, correct / total, time.time() - start_time