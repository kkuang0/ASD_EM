import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from typing import List, Tuple
import seaborn as sns
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize

def plot_sample_images(dataset, num_samples: int = 8, save_path: str = None):
    """Plot sample images from the dataset"""
    
    # Get random samples
    indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    
    cols = min(4, num_samples)
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    if cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, idx in enumerate(indices):
        row, col = i // cols, i % cols
        
        image, labels = dataset[idx]
        
        # Convert tensor to numpy for plotting
        if isinstance(image, torch.Tensor):
            # Denormalize if necessary
            image_np = image.permute(1, 2, 0).numpy()
            # Assuming ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_np = std * image_np + mean
            image_np = np.clip(image_np, 0, 1)
        else:
            image_np = np.array(image)
        
        axes[row, col].imshow(image_np)
        axes[row, col].axis('off')
        
        # Create title with labels
        title_parts = []
        for task, label in labels.items():
            title_parts.append(f"{task}: {label}")
        axes[row, col].set_title(", ".join(title_parts), fontsize=8)
    
    # Hide empty subplots
    for i in range(num_samples, rows * cols):
        row, col = i // cols, i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def create_results_summary(results_df, save_path: str = None):
    """Create a comprehensive results summary plot"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Results by fold (bar plot)
    fold_data = results_df[results_df['fold'].str.isdigit().fillna(False)]
    task_cols = [col for col in fold_data.columns if col not in ['fold', 'average']]
    
    x = range(len(fold_data))
    width = 0.8 / len(task_cols)
    
    for i, task in enumerate(task_cols):
        axes[0, 0].bar([xi + i*width for xi in x], fold_data[task], 
                      width, label=task, alpha=0.8)
    
    axes[0, 0].set_xlabel('Fold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Results by Fold')
    axes[0, 0].set_xticks([xi + width*(len(task_cols)-1)/2 for xi in x])
    axes[0, 0].set_xticklabels(fold_data['fold'])
    axes[0, 0].legend()
    
    # Box plot of results distribution
    box_data = [fold_data[task].values for task in task_cols]
    axes[0, 1].boxplot(box_data, labels=task_cols)
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Results Distribution')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Mean results with error bars
    mean_row = results_df[results_df['fold'] == 'mean'].iloc[0]
    std_row = results_df[results_df['fold'] == 'std'].iloc[0]
    
    means = [mean_row[task] for task in task_cols]
    stds = [std_row[task] for task in task_cols]
    
    axes[1, 0].bar(task_cols, means, yerr=stds, capsize=5, alpha=0.8)
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Mean Results ± Std')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Summary statistics table
    summary_data = results_df[results_df['fold'].isin(['mean', 'std'])]
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    table = axes[1, 1].table(cellText=summary_data.round(4).values,
                           colLabels=summary_data.columns,
                           cellLoc='center',
                           loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    axes[1, 1].set_title('Summary Statistics')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_roc_curves(labels_dict, probabilities_dict, task_names):
    """Plot ROC curves for each task (binary) or one-vs-rest (multi-class)"""
    fig, axes = plt.subplots(1, len(task_names), figsize=(5*len(task_names), 4))
    if len(task_names) == 1:
        axes = [axes]
    
    for idx, task in enumerate(task_names):
        # Convert inputs to numpy arrays for easier indexing
        labels = np.asarray(labels_dict[task])
        probs = np.asarray(probabilities_dict[task])
        
        if probs.shape[1] == 2:  # Binary classification
            fpr, tpr, _ = roc_curve(labels, probs[:, 1])  # use positive class prob
            auc_score = auc(fpr, tpr)
            axes[idx].plot(fpr, tpr, label=f'AUC = {auc_score:.3f}')
            axes[idx].plot([0, 1], [0, 1], 'k--', alpha=0.5)
            axes[idx].set_xlabel('False Positive Rate')
            axes[idx].set_ylabel('True Positive Rate')
            axes[idx].set_title(f'ROC Curve - {task}')
            axes[idx].legend()
            
        else:  # Multi-class classification (one-vs-rest)
            for class_idx in range(probs.shape[1]):
                # Convert to binary: current class vs all others
                binary_labels = (labels == class_idx).astype(int)
                fpr, tpr, _ = roc_curve(binary_labels, probs[:, class_idx])
                auc_score = auc(fpr, tpr)
                axes[idx].plot(fpr, tpr, label=f'Class {class_idx} AUC = {auc_score:.3f}')
            
            axes[idx].plot([0, 1], [0, 1], 'k--', alpha=0.5)
            axes[idx].set_xlabel('False Positive Rate')
            axes[idx].set_ylabel('True Positive Rate')
            axes[idx].set_title(f'ROC Curves - {task} (One-vs-Rest)')
            axes[idx].legend()
    
    plt.tight_layout()
    return fig


def plot_pr_curves(labels_dict, probabilities_dict, task_names):
    """Plot PR curves for each task (binary) or one-vs-rest (multi-class)"""
    fig, axes = plt.subplots(1, len(task_names), figsize=(5*len(task_names), 4))
    if len(task_names) == 1:
        axes = [axes]
    
    for idx, task in enumerate(task_names):
        # Convert inputs to numpy arrays for easier indexing
        labels = np.asarray(labels_dict[task])
        probs = np.asarray(probabilities_dict[task])
        
        if probs.shape[1] == 2:  # Binary classification
            precision, recall, _ = precision_recall_curve(labels, probs[:, 1])
            ap_score = average_precision_score(labels, probs[:, 1])
            axes[idx].plot(recall, precision, label=f'AP = {ap_score:.3f}')
            axes[idx].set_xlabel('Recall')
            axes[idx].set_ylabel('Precision')
            axes[idx].set_title(f'PR Curve - {task}')
            axes[idx].legend()
            
        else:  # Multi-class classification (one-vs-rest)
            for class_idx in range(probs.shape[1]):
                # Convert to binary: current class vs all others
                binary_labels = (labels == class_idx).astype(int)
                precision, recall, _ = precision_recall_curve(binary_labels, probs[:, class_idx])
                ap_score = average_precision_score(binary_labels, probs[:, class_idx])
                axes[idx].plot(recall, precision, label=f'Class {class_idx} AP = {ap_score:.3f}')
            
            axes[idx].set_xlabel('Recall')
            axes[idx].set_ylabel('Precision')
            axes[idx].set_title(f'PR Curves - {task} (One-vs-Rest)')
            axes[idx].legend()
    
    plt.tight_layout()
    return fig
