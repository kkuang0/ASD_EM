import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import pandas as pd

def calculate_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Calculate various classification metrics"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    return {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, average='weighted', zero_division=0),
        'recall': recall_score(labels, predictions, average='weighted', zero_division=0),
        'f1': f1_score(labels, predictions, average='weighted', zero_division=0)
    }

def plot_training_curves(log_dir: str, save_path: str = None):
    """Plot training curves from TensorBoard logs"""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    # Get scalar tags
    tags = event_acc.Tags()['scalars']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    # Plot loss curves
    if 'Loss/Train' in tags and 'Loss/Val' in tags:
        train_loss = event_acc.Scalars('Loss/Train')
        val_loss = event_acc.Scalars('Loss/Val')
        
        train_steps = [x.step for x in train_loss]
        train_values = [x.value for x in train_loss]
        val_steps = [x.step for x in val_loss]
        val_values = [x.value for x in val_loss]
        
        axes[0].plot(train_steps, train_values, label='Train')
        axes[0].plot(val_steps, val_values, label='Validation')
        axes[0].set_title('Loss Curves')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
    
    # Plot accuracy curves for each task
    accuracy_tags = [tag for tag in tags if tag.startswith('Accuracy/Val_') and tag != 'Accuracy/Val_Average']
    
    for i, tag in enumerate(accuracy_tags[:3]):  # Max 3 tasks
        if i < 3:
            accuracy_data = event_acc.Scalars(tag)
            steps = [x.step for x in accuracy_data]
            values = [x.value for x in accuracy_data]
            
            task_name = tag.split('_')[-1]
            axes[i+1].plot(steps, values)
            axes[i+1].set_title(f'{task_name.capitalize()} Accuracy')
            axes[i+1].set_xlabel('Epoch')
            axes[i+1].set_ylabel('Accuracy')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def create_results_dataframe(all_fold_results: List[Dict[str, float]], task_names: List[str]) -> pd.DataFrame:
    """Create a DataFrame summarizing cross-validation results"""
    
    data = []
    for fold_idx, fold_results in enumerate(all_fold_results):
        row = {'fold': fold_idx + 1}
        row.update(fold_results)
        row['average'] = np.mean(list(fold_results.values()))
        data.append(row)
    
    # Add summary statistics
    df = pd.DataFrame(data)
    
    summary_row = {'fold': 'mean'}
    for col in df.columns[1:]:
        summary_row[col] = df[col].mean()
    data.append(summary_row)
    
    summary_row = {'fold': 'std'}
    for col in df.columns[1:]:
        summary_row[col] = df[col].std()
    data.append(summary_row)
    
    return pd.DataFrame(data)