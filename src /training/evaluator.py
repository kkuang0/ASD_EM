import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    """Model evaluation utilities"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        self.task_names = list(config.num_classes.keys())
    
    def evaluate(self, model, dataloader, detailed: bool = False) -> Dict:
        """Evaluate model on given dataloader"""
        model.eval()
        all_preds = {task: [] for task in self.task_names}
        all_labels = {task: [] for task in self.task_names}
        all_probs = {task: [] for task in self.task_names}
        total_loss = 0.0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = {k: v.to(self.device) for k, v in labels.items()}
                
                outputs = model(images)
                
                # Calculate loss
                task_losses = {}
                for task in self.task_names:
                    task_losses[task] = F.cross_entropy(outputs[task], labels[task])
                
                batch_loss = sum(
                    self.config.task_weights.get(task, 1.0) * task_losses[task] 
                    for task in self.task_names
                )
                total_loss += batch_loss.item()
                
                # Collect predictions, probabilities, and labels
                for task in self.task_names:
                    probs = F.softmax(outputs[task], dim=1)
                    preds = torch.argmax(outputs[task], dim=1)
                    
                    all_preds[task].extend(preds.cpu().numpy())
                    all_labels[task].extend(labels[task].cpu().numpy())
                    all_probs[task].extend(probs.cpu().numpy())
        
        # Calculate metrics
        results = {
            'loss': total_loss / len(dataloader),
            'accuracies': {},
            'f1_scores': {}
        }
        
        for task in self.task_names:
            # Accuracy
            correct = np.array(all_preds[task]) == np.array(all_labels[task])
            results['accuracies'][task] = correct.mean()
            
            # F1 score
            f1 = f1_score(all_labels[task], all_preds[task], average='weighted')
            results['f1_scores'][task] = f1
        
        if detailed:
            results['predictions'] = all_preds
            results['labels'] = all_labels
            results['probabilities'] = all_probs
            results['classification_reports'] = {}
            results['confusion_matrices'] = {}
            
            for task in self.task_names:
                # Classification report
                results['classification_reports'][task] = classification_report(
                    all_labels[task], all_preds[task], output_dict=True
                )
                
                # Confusion matrix
                results['confusion_matrices'][task] = confusion_matrix(
                    all_labels[task], all_preds[task]
                )
        
        return results
    
    def plot_confusion_matrices(self, results: Dict, save_path: Optional[str] = None):
        """Plot confusion matrices for all tasks"""
        if 'confusion_matrices' not in results:
            raise ValueError("Results must contain confusion matrices")
        
        n_tasks = len(self.task_names)
        fig, axes = plt.subplots(1, n_tasks, figsize=(5*n_tasks, 4))
        
        if n_tasks == 1:
            axes = [axes]
        
        for i, task in enumerate(self.task_names):
            cm = results['confusion_matrices'][task]
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
            axes[i].set_title(f'{task.capitalize()} Confusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def print_detailed_results(self, results: Dict):
        """Print detailed evaluation results"""
        print("\n" + "="*60)
        print("DETAILED EVALUATION RESULTS")
        print("="*60)
        
        print(f"Overall Loss: {results['loss']:.4f}")
        
        for task in self.task_names:
            print(f"\n{task.upper()} TASK:")
            print(f"  Accuracy: {results['accuracies'][task]:.4f}")
            print(f"  F1 Score: {results['f1_scores'][task]:.4f}")
            
            if 'classification_reports' in results:
                print(f"\n  Classification Report:")
                report = results['classification_reports'][task]
                for class_name, metrics in report.items():
                    if isinstance(metrics, dict):
                        print(f"    {class_name}: precision={metrics['precision']:.3f}, "
                              f"recall={metrics['recall']:.3f}, f1={metrics['f1-score']:.3f}")