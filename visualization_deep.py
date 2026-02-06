import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import config
'''
This categorization is extremely unreasonable and serves only as a temporary measure.
'''
def set_smart_xticks(ax, data_len, cuts):
    # set x ticks 
    if data_len <= 10:
        # Epoch
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.set_xlabel('Training Epochs', fontsize=11, fontweight='bold')
        # 1 to data_len
        ax.set_xticks(range(1, data_len + 1))
    else:
        # Layer Depth
        ax.set_xlabel('Layer Depth (Complexity)', fontsize=11, fontweight='bold')
        if len(cuts) > 10:
            ax.set_xticks(cuts[::max(1, len(cuts)//10)])
        else:
            ax.set_xticks(cuts)

def plot_summary(train_accs, val_accs, test_accs, train_losses, val_losses, test_losses,
                 train_times, feature_extract_times):
    data_len = len(train_accs)
    is_vit = data_len < 10
    
    if is_vit:
        x_axis = list(range(1, data_len + 1))
        title_suffix = "(ViT Training Progress)"
    else:
        x_axis = config.CUTS
        title_suffix = "(MobileNet Feature Depth)"

    print(f'\n{"="*60}')
    print(f'Pic generating (Data in total {data_len} , pattern {title_suffix})')
    print(f'{"="*60}')
    
    fig = plt.figure(figsize=(16, 12))
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'
    
    # -----------------Accuracy----------------
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(x_axis, train_accs, marker='o', label='Train Acc', linewidth=2, markersize=6, alpha=0.8)
    ax1.plot(x_axis, val_accs, marker='s', label='Val Acc', linewidth=2, markersize=6, alpha=0.8)
    ax1.plot(x_axis, test_accs, marker='^', label='Test Acc', linewidth=2, markersize=8, alpha=0.9, color='#d62728') # 红色突出Test
    
    ax1.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax1.set_title(f'Accuracy Curve {title_suffix}', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    set_smart_xticks(ax1, data_len, config.CUTS)
    
    best_idx = np.argmax(test_accs)
    best_val = test_accs[best_idx]
    best_x = x_axis[best_idx]
    ax1.annotate(f'Best: {best_val:.1%}', xy=(best_x, best_val), xytext=(best_x, best_val-0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05), horizontalalignment='center')

    # ----------------Loss ----------------
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(x_axis, train_losses, marker='o', label='Train Loss', linewidth=2, markersize=6, alpha=0.8, color='#1f77b4')
    ax2.plot(x_axis, val_losses, marker='s', label='Val Loss', linewidth=2, markersize=6, alpha=0.8, color='#ff7f0e')
    ax2.plot(x_axis, test_losses, marker='^', label='Test Loss', linewidth=2, markersize=6, alpha=0.8, color='#2ca02c')
    
    ax2.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax2.set_title(f'Loss Curve {title_suffix}', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    set_smart_xticks(ax2, data_len, config.CUTS)

    # ---------------cost----------------------
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(x_axis, train_times, marker='o', label='Training Time (s)', linewidth=2, markersize=6, color='#d62728')
    
    if sum(feature_extract_times) > 0.1:
        ax3.plot(x_axis, feature_extract_times, marker='s', label='Feature Extraction (s)', linewidth=2, markersize=6, color='#9467bd')
    
    ax3.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax3.set_title('Computational Cost', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    set_smart_xticks(ax3, data_len, config.CUTS)

    # --------------total----------------------
    ax4 = plt.subplot(2, 2, 4)
    ax4_twin = ax4.twinx()
    
    line1 = ax4.plot(x_axis, test_accs, marker='^', label='Test Accuracy', linewidth=3, 
                     markersize=8, color='#2ca02c') 
    ax4.set_ylabel('Test Accuracy', fontsize=11, fontweight='bold', color='#2ca02c')
    ax4.tick_params(axis='y', labelcolor='#2ca02c')
    
    line2 = ax4_twin.plot(x_axis, train_times, marker='o', label='Training Cost', 
                          linewidth=2, markersize=6, color='#d62728', linestyle='--') 
    ax4_twin.set_ylabel('Time (s)', fontsize=11, fontweight='bold', color='#d62728')
    ax4_twin.tick_params(axis='y', labelcolor='#d62728')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='center right', fontsize=10)
    
    ax4.set_title('Efficiency Trade-off: Accuracy vs Cost', fontsize=12, fontweight='bold')
    set_smart_xticks(ax4, data_len, config.CUTS)
    
    plt.tight_layout()
    plt.savefig(config.RESULT_FIG, dpi=300, bbox_inches='tight')
    print(f'Pic saved to {config.RESULT_FIG}')
    plt.close()