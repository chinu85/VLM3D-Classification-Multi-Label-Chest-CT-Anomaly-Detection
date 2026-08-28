import re
import matplotlib.pyplot as plt
import os
import numpy as np

def parse_log(log_path):
    epochs = []
    train_loss = []
    val_loss = []
    val_auroc = []
    val_f1 = []
    
    if not os.path.exists(log_path):
        print(f"Warning: File {log_path} not found.")
        return None
        
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        
    pattern = r"Epoch \[(\d+)/\d+\] Train Loss: ([\d.]+) Val Loss: ([\d.]+) Val AUROC: ([\d.]+) Val F1: ([\d.]+)"
    matches = re.findall(pattern, content)
    
    for match in matches:
        epochs.append(int(match[0]))
        train_loss.append(float(match[1]))
        val_loss.append(float(match[2]))
        val_auroc.append(float(match[3]))
        val_f1.append(float(match[4]))
        
    return {
        'epochs': epochs,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_auroc': val_auroc,
        'val_f1': val_f1
    }

def main():
    log_baseline = "374245_train.log"
    log_stage3_interrupted = "374567_train.log"
    log_stage3_final = "381849_train.log"
    
    data_baseline = parse_log(log_baseline)
    data_interrupted = parse_log(log_stage3_interrupted)
    data_final = parse_log(log_stage3_final)
    
    # Styled plot settings
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Plot Validation AUROC Comparison
    if data_baseline:
        ax1.plot(data_baseline['epochs'], data_baseline['val_auroc'], 
                 label='Baseline (2.0mm Spacing, BCE Loss, No TTA)', 
                 color='#888888', linestyle='--', linewidth=1.5)
    if data_interrupted:
        ax1.plot(data_interrupted['epochs'], data_interrupted['val_auroc'], 
                 label='Stage 3 (1.5mm, ASL, TTA) - Job Interrupted (82 Epochs)', 
                 color='#ff7f0e', linestyle='-.', linewidth=2.0)
    if data_final:
        ax1.plot(data_final['epochs'], data_final['val_auroc'], 
                 label='Stage 3 Final (1.5mm, ASL, TTA) - Fully Trained (100 Epochs)', 
                 color='#1f77b4', linestyle='-', linewidth=2.5)
        
        # Highlight best epoch
        best_idx = np.argmax(data_final['val_auroc'])
        best_epoch = data_final['epochs'][best_idx]
        best_val = data_final['val_auroc'][best_idx]
        ax1.scatter(best_epoch, best_val, color='#d62728', s=100, zorder=5, 
                    label=f'Peak AUROC: {best_val:.4f} (Epoch {best_epoch})')
        ax1.annotate(f"Peak: {best_val:.4f}", 
                     xy=(best_epoch, best_val), 
                     xytext=(best_epoch - 15, best_val - 0.015),
                     arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6))
        
    ax1.set_title('Validation AUROC Progression (Macro)', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Macro AUROC', fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
    ax1.set_ylim(0.70, 0.81)
    
    # 2. Plot Validation F1 Comparison
    if data_baseline:
        ax2.plot(data_baseline['epochs'], data_baseline['val_f1'], 
                 label='Baseline (2.0mm Spacing, BCE Loss, No TTA)', 
                 color='#888888', linestyle='--', linewidth=1.5)
    if data_interrupted:
        ax2.plot(data_interrupted['epochs'], data_interrupted['val_f1'], 
                 label='Stage 3 (1.5mm, ASL, TTA) - Job Interrupted (82 Epochs)', 
                 color='#ff7f0e', linestyle='-.', linewidth=2.0)
    if data_final:
        ax2.plot(data_final['epochs'], data_final['val_f1'], 
                 label='Stage 3 Final (1.5mm, ASL, TTA) - Fully Trained (100 Epochs)', 
                 color='#2ca02c', linestyle='-', linewidth=2.5)
        
        # Highlight best epoch for F1 or final best overall
        # Let's find best F1
        best_f1_idx = np.argmax(data_final['val_f1'])
        best_f1_epoch = data_final['epochs'][best_f1_idx]
        best_f1_val = data_final['val_f1'][best_f1_idx]
        ax2.scatter(best_f1_epoch, best_f1_val, color='#d62728', s=100, zorder=5,
                    label=f'Peak F1: {best_f1_val:.4f} (Epoch {best_f1_epoch})')
        ax2.annotate(f"Peak: {best_f1_val:.4f}", 
                     xy=(best_f1_epoch, best_f1_val), 
                     xytext=(best_f1_epoch - 15, best_f1_val - 0.015),
                     arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6))
        
    ax2.set_title('Validation F1-Score Progression (Macro)', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Macro F1-Score', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
    ax2.set_ylim(0.40, 0.50)
    
    plt.suptitle('VLM3D Frontier Upgrade Performance Comparison', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    comparison_img_path = "stage3_comparison_metrics.png"
    plt.savefig(comparison_img_path, dpi=300)
    print(f"Saved beautiful comparison plot to: {os.path.abspath(comparison_img_path)}")

if __name__ == "__main__":
    main()
