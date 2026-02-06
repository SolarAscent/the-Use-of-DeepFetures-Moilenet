import matplotlib.pyplot as plt
import config

def set_xticks(ax, cuts):
    # set x ticks
    if len(cuts) > 10:
        ax.set_xticks(cuts[::max(1, len(cuts)//10)])
    else:
        ax.set_xticks(cuts)

def plot_summary(train_accs, val_accs, test_accs, train_losses, val_losses, test_losses,
                 train_times, feature_extract_times):
    print(f'\n{"="*60}')
    print('Generating pic...')
    print(f'{"="*60}')
    
    fig = plt.figure(figsize=(16, 12))
    plt.rcParams['font.size'] = 10
    
    # -----------------Accuracy----------------
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(config.CUTS, train_accs, marker='o', label='Train', linewidth=2, markersize=4, alpha=0.8)
    ax1.plot(config.CUTS, val_accs, marker='s', label='Val', linewidth=2, markersize=4, alpha=0.8)
    ax1.plot(config.CUTS, test_accs, marker='^', label='Test', linewidth=2, markersize=4, alpha=0.8)
    ax1.set_xlabel('Layer Depth', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax1.set_title('Accuracy vs Layer Depth', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(alpha=0.3, linestyle='--')
    set_xticks(ax1, config.CUTS)
    
    # ----------------Loss ----------------
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(config.CUTS, train_losses, marker='o', label='Train Loss', linewidth=2, markersize=4, alpha=0.8, color='#1f77b4')
    ax2.plot(config.CUTS, val_losses, marker='s', label='Val Loss', linewidth=2, markersize=4, alpha=0.8, color='#ff7f0e')
    ax2.plot(config.CUTS, test_losses, marker='^', label='Test Loss', linewidth=2, markersize=4, alpha=0.8, color='#2ca02c')
    ax2.set_xlabel('Layer Depth', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Hinge Loss', fontsize=11, fontweight='bold')
    ax2.set_title('Loss vs Layer Depth', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(alpha=0.3, linestyle='--')
    set_xticks(ax2, config.CUTS)
    
    # ---------------cost----------------------
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(config.CUTS, train_times, marker='o', label='Training Time', linewidth=2, markersize=4, alpha=0.8, color='#d62728')
    ax3.plot(config.CUTS, feature_extract_times, marker='s', label='Feature Extraction Time', linewidth=2, markersize=4, alpha=0.8, color='#9467bd')
    ax3.set_xlabel('Layer Depth', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax3.set_title('Training Cost vs Layer Depth', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(alpha=0.3, linestyle='--')
    set_xticks(ax3, config.CUTS)
    
    # --------------total----------------------
    ax4 = plt.subplot(2, 2, 4)
    ax4_twin = ax4.twinx()
    
    line1 = ax4.plot(config.CUTS, test_accs, marker='o', label='Test Accuracy', linewidth=2.5, 
                     markersize=5, alpha=0.9, color='#2ca02c')
    ax4.set_xlabel('Layer Depth', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Test Accuracy', fontsize=11, fontweight='bold', color='#2ca02c')
    ax4.tick_params(axis='y', labelcolor='#2ca02c')
    
    line2 = ax4_twin.plot(config.CUTS, train_times, marker='s', label='Training Time (Cost)', 
                          linewidth=2.5, markersize=5, alpha=0.9, color='#d62728', linestyle='--')
    ax4_twin.set_ylabel('Training Time (seconds)', fontsize=11, fontweight='bold', color='#d62728')
    ax4_twin.tick_params(axis='y', labelcolor='#d62728')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='best', fontsize=9)
    ax4.set_title('Accuracy vs Cost (Dual Y-axis)', fontsize=12, fontweight='bold')
    ax4.grid(alpha=0.3, linestyle='--')
    set_xticks(ax4, config.CUTS)
    
    plt.tight_layout()
    plt.savefig(config.RESULT_FIG, dpi=300, bbox_inches='tight')
    print(f'Pic saved to {config.RESULT_FIG}')
