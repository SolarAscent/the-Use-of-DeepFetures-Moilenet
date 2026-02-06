import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
import config
import data_utils
import deep_model_utils
import visualization_deep

# Epochs
NUM_EPOCHS = 6 
print(f'Using device: {config.DEVICE}')
print(f'Vision Transformer_ViT-B/16')
print(f'Target training Epochs: {NUM_EPOCHS}')

def main():
    if not (config.SPLIT_ROOT/'train').exists():
        data_utils.split_dataset()
    data_utils.augment_dataset()
    
    train_ld, val_ld, test_ld = deep_model_utils.make_loaders()
    
    # init
    model = deep_model_utils.build_vit_model(len(config.CLASS_NAMES))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scaler = GradScaler()
    
    # result
    train_accs, val_accs, test_accs = [], [], []
    train_losses, val_losses, test_losses = [], [], []
    train_times = []
    feature_extract_times = []

    print(f'\nStart training ({NUM_EPOCHS} Epochs)...')
    total_start = time.time()
    
    for epoch in range(NUM_EPOCHS):
        print(f'\n{"-"*40}')
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}')
        t_loss, t_acc, t_time = deep_model_utils.train_one_epoch(
            model, train_ld, criterion, optimizer, scaler
        )
        
        v_loss, v_acc, v_time = deep_model_utils.evaluate(model, val_ld, criterion)
        te_loss, te_acc, te_time = deep_model_utils.evaluate(model, test_ld, criterion)
        
        print(f'  Train Acc: {t_acc:.1%} | Loss: {t_loss:.4f}')
        print(f'  Test  Acc: {te_acc:.1%} | Loss: {te_loss:.4f}')
        print(f'  Time Cost: {t_time:.1f}s')
        
        train_accs.append(t_acc)
        val_accs.append(v_acc)
        test_accs.append(te_acc)
        
        train_losses.append(t_loss)
        val_losses.append(v_loss)
        test_losses.append(te_loss)
        
        train_times.append(t_time)
        feature_extract_times.append(0.0)

    # pic
    original_fig_name = config.RESULT_FIG
    config.RESULT_FIG = f'result_vit_{NUM_EPOCHS}epochs.png'
    
    visualization_deep.plot_summary(
        train_accs, val_accs, test_accs, 
        train_losses, val_losses, test_losses,
        train_times, feature_extract_times
    )
    
    result = {
        'model': 'ViT-B/16 Custom Run',
        'epochs': NUM_EPOCHS,
        'train_accuracy': [round(a, 4) for a in train_accs],
        'test_accuracy':  [round(a, 4) for a in test_accs],
        'train_loss':     [round(a, 4) for a in train_losses],
        'test_loss':      [round(a, 4) for a in test_losses],
        'train_time':     [round(a, 4) for a in train_times]
    }
    
    txt_name = f'result_vit_{NUM_EPOCHS}epochs.txt'
    with open(txt_name, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        
    print(f'\n{"="*60}')
    print('Training Finished!')
    print(f'Picture saved: {config.RESULT_FIG}')
    print(f'Results saved to: {txt_name}')
    print(f'Final Test Accuracy: {test_accs[-1]:.1%}')
    print(f'{"="*60}')
    
    config.RESULT_FIG = original_fig_name

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    main()