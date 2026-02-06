import time
import json
import numpy as np
import config
import data_utils
import model_utils
import visualization

print(f'Using device: {config.DEVICE}')
print(f'MobileNet V2 Total Layers: {config.TOTAL_LAYERS}, will training: {config.CUTS}')

# ---------- main ----------
def main():
    need_split = False
    if not (config.SPLIT_ROOT/'train').exists():
        need_split = True
    else:
        total_files = data_utils.count_images_in_split()
        if total_files == 0:
            print('data_split empty, need to split dataset again')
            need_split = True
        else:
            print(f'Spliting completed already({total_files}pics in total)')
    
    if need_split:
        data_utils.split_dataset()

    data_utils.augment_dataset()

    train_ld, val_ld, test_ld = model_utils.make_loaders()

    train_accs, val_accs, test_accs, dims = [], [], [], []
    train_losses, val_losses, test_losses = [], [], []
    train_times, val_times, test_times = [], [], []
    feature_extract_times = []


    print(f'\nStarting in {len(config.CUTS)} layers...')
    for cut in config.CUTS:
        print(f'\n{"="*60}')
        print(f'Depth cut={cut}/{config.TOTAL_LAYERS}')
        print(f'{"="*60}')
       
        # Feature extraction
        extract_start = time.time()
        net = model_utils.build_extractor(cut)
        print('Feature extraction...')
        train_x, train_y = model_utils.extract(train_ld, net)
        val_x,   val_y   = model_utils.extract(val_ld,   net)
        test_x,  test_y  = model_utils.extract(test_ld,   net)
        extract_time = time.time() - extract_start
        feature_extract_times.append(extract_time)
        
        dim = train_x.shape[1]
        print(f'Feature dimension: {dim}')
        
        
        print('Training SVM classifier')
        results = model_utils.train_and_evaluate_svm(train_x, train_y, val_x, val_y, test_x, test_y)
        
        # Results display
        print(f'\n--- Results ---')
        print(f'Feature extraction time: {extract_time:.2f} seconds')
        print(f'SVM training time:  {results["train_time"]:.2f} seconds (cost)')
        print(f'Prediction time:     val={results["val_time"]:.3f} seconds, test={results["test_time"]:.3f} seconds')
        print(f'\nAccuracy:')
        print(f'  Train: {results["train_acc"]:.4f} ({results["train_acc"]:.1%})')
        print(f'  Val:   {results["val_acc"]:.4f} ({results["val_acc"]:.1%})')
        print(f'  Test:  {results["test_acc"]:.4f} ({results["test_acc"]:.1%})')
        print(f'\nLoss (Hinge Loss):')
        print(f'  Train: {results["train_loss"]:.4f}')
        print(f'  Val:   {results["val_loss"]:.4f}')
        print(f'  Test:  {results["test_loss"]:.4f}')
        
        # save
        train_accs.append(results["train_acc"])
        val_accs.append(results["val_acc"])
        test_accs.append(results["test_acc"])
        train_losses.append(results["train_loss"])
        val_losses.append(results["val_loss"])
        test_losses.append(results["test_loss"])
        train_times.append(results["train_time"])
        val_times.append(results["val_time"])
        test_times.append(results["test_time"])
        dims.append(dim)
        
        progress = (cut - config.CUTS[0] + 1) / len(config.CUTS) * 100
        print(f'\n进度: {progress:.1f}% ({cut - config.CUTS[0] + 1}/{len(config.CUTS)})')

    # Pic
    visualization.plot_summary(train_accs, val_accs, test_accs, train_losses, val_losses, test_losses,
                               train_times, feature_extract_times)

    # Result
    result = {
        'depth_blocks': config.CUTS,
        'feature_dims': dims,
        'train_accuracy': [round(a, 4) for a in train_accs],
        'val_accuracy':   [round(a, 4) for a in val_accs],
        'test_accuracy':  [round(a, 4) for a in test_accs],
        'train_loss':     [round(a, 4) for a in train_losses],
        'val_loss':       [round(a, 4) for a in val_losses],
        'test_loss':      [round(a, 4) for a in test_losses],
        'train_time':     [round(a, 4) for a in train_times],
        'val_time':       [round(a, 4) for a in val_times],
        'test_time':      [round(a, 4) for a in test_times],
        'feature_extract_time': [round(a, 4) for a in feature_extract_times]
    }
    with open(config.RESULT_TXT, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f'Result saved to {config.RESULT_TXT}')
    
    best_test_idx = np.argmax(test_accs)
    best_layer = config.CUTS[best_test_idx]
    print(f'\n{"="*60}')
    print('Best Result:')
    print(f'  Depth: {best_layer}')
    print(f'  Test Accuracy: {test_accs[best_test_idx]:.4f} ({test_accs[best_test_idx]:.1%})')
    print(f'  Test Loss: {test_losses[best_test_idx]:.4f}')
    print(f'  Training time: {train_times[best_test_idx]:.2f} seconds')
    print(f'  Feature dimension: {dims[best_test_idx]}')
    print(f'{"="*60}')
    print('\nCompleted')

if __name__ == '__main__':
    main()
