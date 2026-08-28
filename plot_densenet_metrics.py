import re
import matplotlib.pyplot as plt
import os

def parse_and_plot_log(log_path):
    epochs = []
    train_loss = []
    val_loss = []
    val_auroc = []
    val_f1 = []
    
    with open(log_path, 'r', encoding='utf-8') as f:
        # Fallback to standard utf-8, but might need utf-16 if powershell did it
        try:
            content = f.read()
        except UnicodeDecodeError:
            with open(log_path, 'r', encoding='utf-16') as f2:
                content = f2.read()
                
    pattern = r"Epoch \[(\d+)/\d+\] Train Loss: ([\d.]+) Val Loss: ([\d.]+) Val AUROC: ([\d.]+) Val F1: ([\d.]+)"
    matches = re.findall(pattern, content)
    
    for match in matches:
        epochs.append(int(match[0]))
        train_loss.append(float(match[1]))
        val_loss.append(float(match[2]))
        val_auroc.append(float(match[3]))
        val_f1.append(float(match[4]))
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(epochs, train_loss, label='Train Loss', color='blue', linewidth=2)
    ax1.plot(epochs, val_loss, label='Val Loss', color='red', linewidth=2)
    ax1.set_title('Loss Curve (DenseNet201)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('BCEWithLogits Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.plot(epochs, val_auroc, label='Val AUROC', color='green', linewidth=2)
    ax2.plot(epochs, val_f1, label='Val F1', color='orange', linewidth=2)
    ax2.set_title('Performance Metrics (Macro)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('densenet201_metrics.png', dpi=300)
    print("Saved plot to densenet201_metrics.png")

if __name__ == "__main__":
    parse_and_plot_log(r"c:\Users\shmso\UCD\Spring\Data challanges\374245_train.log")
