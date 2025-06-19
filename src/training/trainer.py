import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import Counter

from ..data import EMAxonDataset, get_em_transforms, get_backbone_image_size, create_splits
from ..models import MultiTaskCNN
from ..utils import setup_logging, calculate_metrics
from .evaluator import ModelEvaluator
from .losses import MultiTaskLoss  


class CNNTrainer:
    """Main trainer class for multi-task CNN"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        self.task_names = list(config.num_classes.keys())
        
        # Create output directories
        os.makedirs(os.path.join(config.output_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, 'logs'), exist_ok=True)
        
        # Get appropriate image size for backbone
        self.image_size = get_backbone_image_size(config.backbone)
        print(f"Using image size {self.image_size} for backbone {config.backbone}")
        
        self.class_weights = None

        # Initialize evaluator
        self.evaluator = ModelEvaluator(config)
        
        # Initialize loss function
        self._setup_loss_function()
        
    def _setup_loss_function(self):
        """Setup the consolidated multi-task loss function"""
        # Define loss types per task (can be customized)
        loss_types = getattr(self.config, 'loss_types', {
            task: 'ce' for task in self.task_names  # Default to cross-entropy
        })
        
        # If using focal loss for imbalanced tasks
        if hasattr(self.config, 'use_focal_loss') and self.config.use_focal_loss:
            loss_types = {task: 'focal' for task in self.task_names}
        
        # If using label smoothing
        if hasattr(self.config, 'use_label_smoothing') and self.config.use_label_smoothing:
            loss_types = {task: 'smoothed_ce' for task in self.task_names}
        
        self.loss_types = loss_types
        
    def create_model(self) -> MultiTaskCNN:
        """Create and initialize the model"""
        model = MultiTaskCNN(
            backbone_name=self.config.backbone,
            num_classes_dict=self.config.num_classes,
            dropout_rate=self.config.dropout_rate,
            feature_dim=self.config.feature_dim
        )
        return model.to(self.device)
    
    def calculate_class_weights(self, train_dataset):
        """Calculate balanced class weights for each task"""
        
        class_weights = {}
        task_labels = {task: [] for task in self.task_names}
        
        # Collect all labels
        for _, labels in train_dataset:
            for task, label in labels.items():
                task_labels[task].append(label.item() if torch.is_tensor(label) else label)
        
        # Calculate weights for each task
        for task, labels in task_labels.items():
            counter = Counter(labels)
            total = len(labels)
            
            # Calculate inverse frequency weights
            weights = torch.zeros(len(counter))
            for class_id, count in counter.items():
                weights[class_id] = total / (len(counter) * count)
            
            class_weights[task] = weights.to(self.device)
            print(f"{task.capitalize()} class distribution: {counter}")
            print(f"{task.capitalize()} weights: {weights}")
        
        return class_weights
    
    def create_dataloaders(self, train_df, val_df, test_df):
        """Create data loaders with updated transforms"""
        
        # Create datasets with updated transforms
        train_dataset = EMAxonDataset(
            train_df, 
            transform=get_em_transforms(image_size=self.image_size, is_training=True)
        )
        val_dataset = EMAxonDataset(
            val_df, 
            transform=get_em_transforms(image_size=self.image_size, is_training=False)
        )
        test_dataset = EMAxonDataset(
            test_df, 
            transform=get_em_transforms(image_size=self.image_size, is_training=False)
        )
        
        # Calculate class weights
        self.class_weights = self.calculate_class_weights(train_dataset)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size,
            shuffle=True, 
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size,
            shuffle=False, 
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config.batch_size,
            shuffle=False, 
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        return train_loader, val_loader, test_loader
    
    def create_loss_function(self) -> MultiTaskLoss:
        """Create the consolidated multi-task loss function"""
        return MultiTaskLoss(
            task_weights=self.config.task_weights,
            loss_types=self.loss_types,
            class_weights=self.class_weights,
            label_smoothing=getattr(self.config, 'label_smoothing', 0.0),
            focal_alpha=getattr(self.config, 'focal_alpha', 1.0),
            focal_gamma=getattr(self.config, 'focal_gamma', 2.0),
            normalize_weights=getattr(self.config, 'normalize_task_weights', True)
        )
    
    def calculate_loss(self, outputs: Dict[str, torch.Tensor], 
                      labels: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Calculate loss using consolidated loss function"""
        if not hasattr(self, 'criterion'):
            self.criterion = self.create_loss_function()
        
        return self.criterion(outputs, labels)
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                   optimizer: torch.optim.Optimizer, scaler: GradScaler,
                   epoch: int, writer: SummaryWriter) -> float:
        """Enhanced training step with gradient clipping"""
        model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = {k: v.to(self.device, non_blocking=True) for k, v in labels.items()}
            
            optimizer.zero_grad()
            
            with autocast("cuda", enabled=self.config.mixed_precision):
                outputs = model(images)
                loss, task_losses = self.calculate_loss(outputs, labels)
            
            if self.config.mixed_precision:
                scaler.scale(loss).backward()
                
                # Gradient clipping
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    getattr(self.config, 'max_grad_norm', 1.0)
                )
                
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    getattr(self.config, 'max_grad_norm', 1.0)
                )
                
                optimizer.step()
            
            total_loss += loss.item()
            
            # Enhanced logging with gradient norms
            if batch_idx % getattr(self.config, 'log_every_n_batches', 50) == 0:
                step = epoch * num_batches + batch_idx
                writer.add_scalar('Loss/Train_Batch', loss.item(), step)
                writer.add_scalar('GradNorm/Train_Batch', grad_norm.item(), step)
                
                for task, task_loss in task_losses.items():
                    writer.add_scalar(f'Loss/Train_{task}_Batch', task_loss.item(), step)
                
                print(f"Epoch {epoch+1}/{self.config.epochs} - "
                      f"Batch {batch_idx}/{num_batches} - "
                      f"Loss: {loss.item():.4f} - "
                      f"GradNorm: {grad_norm.item():.4f}")
        
        return total_loss / num_batches
    
    def train_fold(self, fold_idx: int, train_df, val_df, test_df) -> Dict[str, float]:
        """Train a single fold"""
        print(f"\n{'='*50}")
        print(f"Training Fold {fold_idx + 1}")
        print(f"{'='*50}")
        
        # Create data loaders
        train_loader, val_loader, test_loader = self.create_dataloaders(train_df, val_df, test_df)
        
        # Create model
        model = self.create_model()
        
        # Improved optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=self.config.lr,
            weight_decay=getattr(self.config, 'weight_decay', 0.01),
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Cosine annealing with warm restarts
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=getattr(self.config, 'scheduler_T0', 10),
            T_mult=getattr(self.config, 'scheduler_T_mult', 2),
            eta_min=getattr(self.config, 'scheduler_eta_min', 1e-6)
        )
        
        # Mixed precision scaler
        scaler = GradScaler(enabled=getattr(self.config, 'mixed_precision', True))
        
        # Logging
        log_dir = os.path.join(
            self.config.output_dir, 'logs',
            f"fold_{fold_idx+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        writer = SummaryWriter(log_dir)
        
        # Training variables
        best_val_score = 0.0
        best_model_state = None
        patience_counter = 0
        max_patience = getattr(self.config, 'early_stopping_patience', float('inf'))
        
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")
        print(f"Image size: {self.image_size}")
        print(f"Loss types: {self.loss_types}")
        
        # Training loop
        for epoch in range(self.config.epochs):
            # Train
            train_loss = self.train_epoch(model, train_loader, optimizer, scaler, epoch, writer)
            
            # Validate
            val_metrics = self.evaluator.evaluate(model, val_loader)
            val_loss = val_metrics['loss']
            val_accuracies = val_metrics['accuracies']
            
            # Learning rate scheduling
            scheduler.step()
            
            # Calculate average validation accuracy
            avg_val_acc = np.mean(list(val_accuracies.values()))
            
            # Early stopping and best model saving
            if avg_val_acc > best_val_score:
                best_val_score = avg_val_acc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Logging
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Val', val_loss, epoch)
            writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
            
            for task, acc in val_accuracies.items():
                writer.add_scalar(f'Accuracy/Val_{task}', acc, epoch)
            writer.add_scalar('Accuracy/Val_Average', avg_val_acc, epoch)
            
            print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
            print(f"Val Accuracies: {val_accuracies}")
            print(f"Average Val Accuracy: {avg_val_acc:.4f} - Best: {best_val_score:.4f}")
            
            # Early stopping
            if patience_counter >= max_patience:
                print(f"Early stopping triggered after {patience_counter} epochs without improvement")
                break
        
        # Load best model and evaluate on test set
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        test_metrics = self.evaluator.evaluate(model, test_loader, detailed=True)
        
        print(f"\nFold {fold_idx+1} Final Results:")
        print(f"Test Accuracies: {test_metrics['accuracies']}")
        print(f"Average Test Accuracy: {np.mean(list(test_metrics['accuracies'].values())):.4f}")
        
        # Save model
        if getattr(self.config, 'save_best_model', True):
            model_save_path = os.path.join(
                self.config.output_dir, 'models', 
                f"fold_{fold_idx+1}_best_model.pth"
            )
            torch.save({
                'model_state_dict': best_model_state,
                'config': self.config,
                'test_metrics': test_metrics,
                'fold_idx': fold_idx,
                'image_size': self.image_size,
                'loss_types': self.loss_types
            }, model_save_path)
        
        writer.close()
        return test_metrics['accuracies']
    
    def train_cross_validation(self) -> Dict[str, List[float]]:
        """Train with cross-validation"""
        print(f"Starting cross-validation with {self.config.n_splits} folds")
        print(f"Using device: {self.device}")
        print(f"Backbone: {self.config.backbone}")
        print(f"Image size: {self.image_size}")
        
        # Create data splits
        folds, test_df = create_splits(
            self.config.csv_path, 
            n_splits=self.config.n_splits,
            holdout_frac=getattr(self.config, 'holdout_frac', 0.2),
            seed=getattr(self.config, 'random_seed', 42)
        )
        
        # Store results from all folds
        all_fold_results = []
        
        # Train each fold
        for fold_idx, (train_df, val_df) in enumerate(folds):
            fold_results = self.train_fold(fold_idx, train_df, val_df, test_df)
            all_fold_results.append(fold_results)
        
        # Calculate cross-validation statistics
        self._print_cv_results(all_fold_results)
        
        return all_fold_results
    
    def _print_cv_results(self, all_fold_results: List[Dict[str, float]]):
        """Print cross-validation results summary"""
        print(f"\n{'='*50}")
        print("CROSS-VALIDATION RESULTS")
        print(f"{'='*50}")
        
        for task in self.task_names:
            task_scores = [fold_result[task] for fold_result in all_fold_results]
            mean_score = np.mean(task_scores)
            std_score = np.std(task_scores)
            print(f"{task.upper()}: {mean_score:.4f} ± {std_score:.4f}")
        
        # Overall average
        overall_scores = []
        for fold_result in all_fold_results:
            overall_scores.append(np.mean(list(fold_result.values())))
        
        print(f"OVERALL: {np.mean(overall_scores):.4f} ± {np.std(overall_scores):.4f}")
        
    def update_task_weights_dynamically(self, epoch: int, val_metrics: Dict[str, float]):
        """
        Optional: Update task weights based on validation performance
        Can be called during training to adaptively balance tasks
        """
        if hasattr(self.config, 'dynamic_task_weighting') and self.config.dynamic_task_weighting:
            # Example: Increase weight for poorly performing tasks
            new_weights = {}
            for task in self.task_names:
                # Lower accuracy = higher weight
                performance_factor = 1.0 - val_metrics.get(task, 0.0)
                new_weights[task] = self.config.task_weights[task] * (1.0 + performance_factor)
            
            # Update the loss function's task weights
            if hasattr(self, 'criterion'):
                self.criterion.update_task_weights(new_weights, normalize=True)
                print(f"Updated task weights: {self.criterion.get_task_weights()}")